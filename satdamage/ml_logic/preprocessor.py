import os
import json
import random
import numpy as np
import tensorflow as tf
import rasterio
from satdamage.params import *
from pathlib import Path
from PIL import Image
from shapely.geometry import shape
from shapely import wkt as shapely_wkt
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional


"""
Module for preprocessing satellite imagery data from xView2 dataset.

This module handles loading, cropping, and preparing building damage detection samples
from pre- and post-disaster images. It includes functions for parsing annotations,
extracting building crops, splitting data, and building TensorFlow datasets.
"""


# ─────────────────────────────────────────────
# 1. PARSING DES ANNOTATIONS JSON xView2
# ─────────────────────────────────────────────

def load_json_buildings(json_path: str) -> List[Dict]:

    with open(json_path, "r") as f:
        data = json.load(f)

    buildings = []
    features  = data.get("features", {}).get("xy", [])

    for feature in features:
        props  = feature.get("properties", {})
        damage = props.get("subtype", "un-classified")

        geom = None

        # ── Format WKT (le plus courant dans xView2)
        wkt_str = feature.get("wkt", "")
        if wkt_str:
            try:
                geom = shapely_wkt.loads(wkt_str)
            except Exception:
                pass

        # ── Format GeoJSON (fallback)
        if geom is None:
            try:
                geom = shape(feature.get("geometry", {}))
            except Exception:
                continue

        if geom is None or geom.is_empty or not geom.is_valid:
            continue

        buildings.append({"polygon": geom, "damage": damage})

    return buildings



# ─────────────────────────────────────────────
# 2. EXTRACTION DES CROPS DE BÂTIMENTS
# ─────────────────────────────────────────────

def polygon_to_pixel_bbox(
    polygon,
    image_width:  int,
    image_height: int,
    padding:      int = CROP_PADDING
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert a shapely polygon to a pixel bounding box with padding and clipping.

    Args:
        polygon: Shapely polygon object.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        padding (int, optional): Padding to add to the bounding box. Defaults to CROP_PADDING.

    Returns:
        Optional[Tuple[int, int, int, int]]: Bounding box as (x_min, y_min, x_max, y_max), or None if too small.
    """
    minx, miny, maxx, maxy = polygon.bounds

    x_min = max(0, int(minx) - padding)
    y_min = max(0, int(miny) - padding)
    x_max = min(image_width,  int(maxx) + padding)
    y_max = min(image_height, int(maxy) + padding)

    # Ignorer les bâtiments trop petits (bruit d'annotation)
    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        return None

    return x_min, y_min, x_max, y_max


def crop_building(
    image:       np.ndarray,
    bbox:        Tuple[int, int, int, int],
    target_size: Tuple[int, int] = CROP_SIZE
) -> np.ndarray:

    x_min, y_min, x_max, y_max = bbox
    crop     = image[y_min:y_max, x_min:x_max, :]

    # Scale each band
    def scale_band(band):
        p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
        if p98 == p2:
            return np.zeros_like(band, dtype=np.float32)
        return np.clip((band.astype(np.float32) - p2) / (p98 - p2), 0, 1)

    rgb = np.stack([scale_band(crop[:,:,0]), scale_band(crop[:,:,1]), scale_band(crop[:,:,2])], axis=-1)

    pil_crop = Image.fromarray((rgb * 255).astype(np.uint8)).resize(target_size, Image.Resampling.LANCZOS)
    return np.array(pil_crop)



# ─────────────────────────────────────────────
# 3. TRAITEMENT D'UNE PAIRE D'IMAGES
# ─────────────────────────────────────────────

def process_image_pair(
    pre_path: str,
    post_path: str,
    label_post_path: str,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:

    def _load_image(path: str) -> np.ndarray:
        """Loads PNG or TIFF as RGB uint8."""
        p = Path(path)
        if p.suffix.lower() in ['.tif', '.tiff']:
            with rasterio.open(path) as src:
                data = src.read()
                rgb = data[:3] if data.shape[0] >= 3 else np.repeat(data[[0]], 3, axis=0)
                img = rgb.transpose(1, 2, 0)
                return np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
        return np.array(Image.open(path).convert("RGB"))

    pre_img = _load_image(pre_path)
    post_img = _load_image(post_path)
    h, w = pre_img.shape[:2]

    buildings = load_json_buildings(label_post_path)
    samples = []

    for building in buildings:
        damage = building["damage"]
        label = DAMAGE_TO_BINARY.get(damage, None)
        if label is None:
            continue
        bbox = polygon_to_pixel_bbox(building["polygon"], w, h)
        if bbox is None:
            continue
        pre_crop = crop_building(pre_img, bbox)
        post_crop = crop_building(post_img, bbox)
        samples.append((pre_crop, post_crop, label))

    return samples



# ─────────────────────────────────────────────
# 4. SCAN DES PAIRES D'IMAGES xView2
# ─────────────────────────────────────────────

def find_image_pairs(
    xview2_root: str
) -> List[Dict[str, str]]:
    pairs = []
    root = Path(xview2_root)

    # Recursively find all directories named "images" under root
    images_dirs = list(root.rglob("images"))

    if not images_dirs:
        print(f"[WARN] Aucun dossier 'images' trouvé sous : {root}")
        return pairs

    for img_dir in images_dirs:
        # Assume "labels" is a sibling directory to "images"
        label_dir = img_dir.parent / "labels"

        if not label_dir.exists() or not label_dir.is_dir():
            print(f"[WARN] Dossier 'labels' manquant ou invalide pour : {img_dir}")
            continue

        # Chercher les fichiers post_disaster en PNG ou TIFF dans ce dossier images
        post_images = sorted(
            list(img_dir.glob("*_post_disaster.png")) +
            list(img_dir.glob("*_post_disaster.tif")) +
            list(img_dir.glob("*_post_disaster.tiff"))
        )

        for post_img_path in post_images:
            stem = post_img_path.stem   # ex: hurricane-florence_00000001_post_disaster
            ext = post_img_path.suffix  # ex: .png ou .tif ou .tiff

            pre_stem = stem.replace("_post_disaster", "_pre_disaster")
            pre_img_path = img_dir / f"{pre_stem}{ext}"
            post_label_path = label_dir / f"{stem}.json"

            if not pre_img_path.exists():
                print(f"[WARN] Image pré-disaster manquante : {pre_img_path}")
                continue
            if not post_img_path.exists():
                print(f"[WARN] Image post-disaster manquante : {post_img_path}")
                continue
            if not post_label_path.exists():
                print(f"[WARN] Annotation JSON manquante : {post_label_path}")
                continue

            # Nom de l'événement : tout sauf les deux derniers segments
            # "hurricane-florence_00000001_post_disaster" → "hurricane-florence"
            event = "_".join(stem.split("_")[:-2])

            pairs.append({
                "pre_img": str(pre_img_path),
                "post_img": str(post_img_path),
                "post_label": str(post_label_path),
                "event": event,
            })

    print(f"[INFO] {len(pairs)} paires d'images trouvées dans tous les sous-dossiers")
    return pairs


# ─────────────────────────────────────────────
# 5. EXTRACTION DE TOUS LES SAMPLES
# ─────────────────────────────────────────────

def build_all_samples(
    pairs: List[Dict[str, str]],
    verbose: bool = True
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:

    image_pairs = []
    labels = []
    errors = 0

    for i, pair in enumerate(pairs):
        if verbose and (i+1) % 10 == 0:
            print(f"Processing {i+1}/{len(pairs)}: {pair['event']}")
        try:
            samples = process_image_pair(
                pair["pre_img"], pair["post_img"], pair["post_label"]
            )
            for pre_crop, post_crop, label in samples:
                image_pairs.append((pre_crop, post_crop))
                labels.append(label)
        except Exception:
            errors += 1

    if verbose:
        dist = Counter(labels)
        total = len(labels)
        print(f"Samples: {total}, Errors: {errors}, Undamaged: {dist.get(0, 0)}, Damaged: {dist.get(1, 0)}")

    return image_pairs, labels


# ─────────────────────────────────────────────
# 6. SPLIT TRAIN / VAL / TEST
# ─────────────────────────────────────────────

def pairs_split(
    pairs:       List[Dict[str, str]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed:        int   = RANDOM_SEED
) -> Tuple[List, List, List]:

    n       = len(pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]

    print(f"\n[INFO] Split :")
    print(f"  Train : {len(train_pairs):>5} paires d'images")
    print(f"  Val   : {len(val_pairs):>5} paires d'images")
    print(f"  Test  : {len(test_pairs):>5} paires d'images")

    return train_pairs, val_pairs, test_pairs


# ─────────────────────────────────────────────
# 7. PRÉPROCESSING & DATA PIPELINE
# ─────────────────────────────────────────────

def preprocess_pair(pre_image, post_image, label):
    """Concatenates and normalizes pre/post images."""
    pre = tf.cast(pre_image, tf.float32) / 255.0
    post = tf.cast(post_image, tf.float32) / 255.0
    pre = tf.image.resize(pre, CROP_SIZE)
    post = tf.image.resize(post, CROP_SIZE)
    combined = tf.concat([pre, post], axis=-1)
    return combined, label

def augment(image, label):
    """Applies basic augmentations."""
    image = tf.image.random_flip_left_right(image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def build_dataset(image_pairs, labels, training=False, batch_size=32):

    pre_images = np.array([p[0] for p in image_pairs])
    post_images = np.array([p[1] for p in image_pairs])
    labels_arr = np.array(labels, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices(((pre_images, post_images), labels_arr))
    ds = ds.map(lambda imgs, lbl: preprocess_pair(imgs[0], imgs[1], lbl), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

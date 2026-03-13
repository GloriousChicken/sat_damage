import os
from io import BytesIO
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
from collections import Counter
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from google.cloud import storage
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


"""
Module for preprocessing satellite imagery data from xView2 dataset.

This module handles loading, cropping, and preparing building damage detection samples
from pre- and post-disaster images. It includes functions for parsing annotations,
extracting building crops, splitting data, and building TensorFlow datasets.
"""


if MODEL_TARGET == "gcs":
    CLIENT = storage.Client()
    BUCKET = CLIENT.bucket(BUCKET_NAME)


# ─────────────────────────────────────────────
# 1. PARSING DES ANNOTATIONS JSON xView2
# ─────────────────────────────────────────────

def load_json_buildings(json_path: str) -> List[Dict]:
    """
    Loads building annotations from a JSON file and returns a list of building dicts with 'polygon' and 'damage' keys.
    Supports both local files and GCS paths based on MODEL_TARGET.
    """
    if MODEL_TARGET == "local":
        with open(json_path, "r") as f:
            data = json.load(f)
    elif MODEL_TARGET == "gcs":
        blob = BUCKET.blob(json_path)
        data = json.loads(blob.download_as_text())
    else:
        print(f"[WARN] Unsupported MODEL_TARGET: {MODEL_TARGET}")
        return []

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
    """
    Crop a building from the image using the bounding box and apply percentile-based normalization.
    Args:
        image (np.ndarray): Input image as a HxWxC array.
        bbox (Tuple[int, int, int, int]): Bounding box as (x_min, y_min, x_max, y_max).
        target_size (Tuple[int, int], optional): Desired output size (width, height). Defaults to CROP_SIZE.
    Returns:
        np.ndarray: Cropped and normalized image as a target_size array."""

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
    """
    Processes a single image pair and its annotations to extract building crops and labels.
    Returns a list of tuples: (pre_crop, post_crop, label) for each building
    """

    def _load_image(path: str) -> np.ndarray:
        """
        Loads PNG or TIFF as RGB uint8.
        """
        p = Path(path)
        if p.suffix.lower() in ['.tif', '.tiff']:
            if MODEL_TARGET == "local":
                with rasterio.open(path) as src:
                    data = src.read()
                    rgb = data[:3] if data.shape[0] >= 3 else np.repeat(data[[0]], 3, axis=0)
                    img = rgb.transpose(1, 2, 0)
                    return np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
            else:
                blob = BUCKET.blob(path)
                bytes_data = blob.download_as_bytes()
                with rasterio.MemoryFile(bytes_data) as memfile:
                    with memfile.open() as src:
                        data = src.read()
                        rgb = data[:3] if data.shape[0] >= 3 else np.repeat(data[[0]], 3, axis=0)
                        img = rgb.transpose(1, 2, 0)
                        return np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
        else:
            if MODEL_TARGET == "local":
                return np.array(Image.open(path).convert("RGB"))
            else:
                blob = BUCKET.blob(path)
                bytes_data = blob.download_as_bytes()
                return np.array(Image.open(BytesIO(bytes_data)).convert("RGB"))

        return np.zeros((256, 256, 3), dtype=np.uint8)

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
# 4-1. SCAN DES PAIRES D'IMAGES xView2 EN LOCAL
# ─────────────────────────────────────────────

def find_image_pairs(
    xview2_root: str
) -> List[Dict[str, str]]:
    """
    Scans the xView2 dataset directory to find all valid image pairs and their corresponding labels.
    Returns a list of dicts with keys: 'pre_img', 'post_img', 'post_label', 'event'.
    The function looks for "images" directories, finds post-disaster images,
    and checks for corresponding pre-disaster images and label JSONs.
    """
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
                print(f"[WARN] Fichier image 'pre_disaster' manquant ou invalide pour : {stem}")
                continue
            if not post_label_path.exists():
                print(f"[WARN] Fichier 'label' manquant ou invalide pour : {stem}")
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
# 4-2. SCAN DES PAIRES D'IMAGES xView2 SUR GCS
# ─────────────────────────────────────────────

def find_image_pairs_gcs(
    prefix: str = ""
) -> List[Dict[str, str]]:
    """
    Scans the xView2 dataset on a GCS bucket to find all valid image pairs and labels.
    Returns a list of dicts with keys: 'pre_img', 'post_img', 'post_label', 'event'.

    Args:
        prefix: Optional prefix to narrow the scan (e.g. "train/" or "")
    """
    pairs = []

    # Lister tous les blobs du bucket sous le prefix donné
    all_blobs = set(
        blob.name for blob in CLIENT.list_blobs(BUCKET_NAME, prefix=prefix)
    )

    if not all_blobs:
        print(f"[WARN] Aucun fichier trouvé sous gs://{BUCKET_NAME}/{prefix}")
        return pairs

    # Filtrer uniquement les images post_disaster
    post_blobs = sorted([
        b for b in all_blobs
        if "_post_disaster" in b and b.endswith((".png", ".tif", ".tiff"))
        and "/images/" in b
    ])

    for post_blob_name in post_blobs:
        # ex: "train/hurricane-florence/images/hurricane-florence_00000001_post_disaster.png"
        parts = post_blob_name.rsplit("/", 1)   # ["train/.../images", "filename.png"]
        img_dir_prefix = parts[0]               # "train/.../images"
        filename = parts[1]                     # "hurricane-florence_00000001_post_disaster.png"

        stem, ext = filename.rsplit(".", 1)
        ext = f".{ext}"

        # Construire les chemins pre_disaster et label
        pre_stem = stem.replace("_post_disaster", "_pre_disaster")
        label_dir_prefix = img_dir_prefix.replace("/images", "/labels")

        pre_blob_name   = f"{img_dir_prefix}/{pre_stem}{ext}"
        label_blob_name = f"{label_dir_prefix}/{stem}.json"

        # Vérifier l'existence dans le set de blobs
        if pre_blob_name not in all_blobs:
            print(f"[WARN] Image pre_disaster manquante : {pre_blob_name}")
            continue
        if label_blob_name not in all_blobs:
            print(f"[WARN] Label manquant : {label_blob_name}")
            continue

        # Nom de l'événement extrait du stem
        event = "_".join(stem.split("_")[:-2])

        pairs.append({
            "pre_img":    pre_blob_name,
            "post_img":   post_blob_name,
            "post_label": label_blob_name,
            "event":      event
        })

    print(f"[INFO] {len(pairs)} paires trouvées dans gs://{BUCKET_NAME}/{prefix}")
    return pairs

# ─────────────────────────────────────────────
# 5. EXTRACTION DE TOUS LES SAMPLES
# ─────────────────────────────────────────────

def _process_pair(pair: Dict[str, str]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int], int]:
    """
    Worker function: processes a single image pair and returns its samples.
    """
    try:
        samples = process_image_pair(
            pair["pre_img"], pair["post_img"], pair["post_label"]
        )
        image_pairs = [(pre_crop, post_crop) for pre_crop, post_crop, _ in samples]
        labels = [label for _, _, label in samples]
        return image_pairs, labels, 0
    except Exception:
        return [], [], 1


def build_all_samples(
    pairs: List[Dict[str, str]],
    verbose: bool = True,
    max_workers: int = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """
    Processes all image pairs in parallel and aggregates the results.

    Args:
        pairs: List of image pair dicts to process.
        verbose: Whether to print progress and summary statistics.
        max_workers: Maximum number of parallel workers (defaults to number of CPU cores).
    Returns:
        image_pairs: List of (pre_crop, post_crop) tuples.
        labels: List of corresponding binary labels."""
    image_pairs = []
    labels = []
    errors = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(_process_pair, pair): pair for pair in pairs}

        for future in as_completed(future_to_pair):
            pair_image_pairs, pair_labels, pair_errors = future.result()
            image_pairs.extend(pair_image_pairs)
            labels.extend(pair_labels)
            errors += pair_errors

            completed += 1
            if verbose and completed % 10 == 0:
                pair = future_to_pair[future]
                print(f"Processed {completed}/{len(pairs)}: {pair['event']}")

    if verbose:
        dist = Counter(labels)
        total = len(labels)
        print(f"Samples: {total}, Errors: {errors}, Undamaged: {dist.get(0, 0)}, Damaged: {dist.get(1, 0)}")

    return image_pairs, labels


# ─────────────────────────────────────────────
# 6. SPLIT TRAIN / VAL / TEST
# ─────────────────────────────────────────────

def split_samples(
    samples: List[Tuple[np.ndarray, np.ndarray]],  # List of (pre_crop, post_crop)
    labels: List[int],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], List[int], List[int], List[int]]:
    """
    Stratified split of samples (crops) into train, val, test sets.
    Preserves label distribution using stratification.

    Args:
        samples: List of (pre_crop, post_crop) tuples.
        labels: Corresponding list of labels.
        train_ratio: Fraction for train.
        val_ratio: Fraction for val.
        seed: Random seed for reproducibility.

    Returns:
        train_samples, val_samples, test_samples, train_labels, val_labels, test_labels
    """

    # First split: train + (val + test)
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples, labels,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=seed
    )

    # Second split: val and test from the remainder
    val_ratio_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))  # Normalize val_ratio for the temp set
    val_samples, test_samples, val_labels, test_labels = train_test_split(
        temp_samples, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=seed
    )

    print(f"\n[INFO] Stratified Split (Crop-Level):")
    print(f"  Train : {len(train_samples):>6} crops (Undamaged: {train_labels.count(0)}, Damaged: {train_labels.count(1)})")
    print(f"  Val   : {len(val_samples):>6} crops (Undamaged: {val_labels.count(0)}, Damaged: {val_labels.count(1)})")
    print(f"  Test  : {len(test_samples):>6} crops (Undamaged: {test_labels.count(0)}, Damaged: {test_labels.count(1)})")

    return train_samples, val_samples, test_samples, train_labels, val_labels, test_labels


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
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)  # prevent out-of-range values after brightness
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def balance_dataset(image_pairs, labels, majority_ratio=2):
    """Balances the dataset by oversampling the minority class and undersampling the majority class."""

    # Reshape image_pairs into a 2D array for imblearn
    X = np.array([np.concatenate([p[0].flatten(), p[1].flatten()]) for p in image_pairs])
    y = np.array(labels)
    class_counts = Counter(y)
    print(f"Before balancing: {class_counts}")
    majority_class = class_counts.most_common(1)[0][0]
    # Minority classes: all other classes except the majority
    minority_classes = class_counts.keys() - {majority_class}

    # Oversample minority class
    strategy_oversampling = {cls: min(class_counts[cls]*2, class_counts[majority_class]) for cls in minority_classes}
    ros = RandomOverSampler(sampling_strategy=strategy_oversampling, random_state=RANDOM_SEED)
    X_ros, y_ros = ros.fit_resample(X, y)

    # Undersample majority class
    strategy_undersampling = {majority_class: int(sum(strategy_oversampling.values())*majority_ratio)}
    rus = RandomUnderSampler(sampling_strategy=strategy_undersampling, random_state=RANDOM_SEED)
    X_balanced, y_balanced = rus.fit_resample(X_ros, y_ros)

    # Reconstruct image pairs from flattened arrays
    crop_size = image_pairs[0][0].shape
    flat_size = np.prod(crop_size)
    balanced_pairs = [(X_balanced[i, :flat_size].reshape(crop_size), X_balanced[i, flat_size:].reshape(crop_size)) for i in range(len(X_balanced))]
    balanced_labels = list(y_balanced)
    print(f"After balancing: {Counter(balanced_labels)}")
    return balanced_pairs, balanced_labels


def build_dataset(image_pairs, labels, training=False, batch_size=32, balance=True, majority_ratio=2.0):

    if training and balance:
        # Balance the dataset by oversampling the minority class and undersampling the majority class.
        # This should work for multiple classes, but here we only have 2.
        image_pairs, labels = balance_dataset(image_pairs, labels, majority_ratio=majority_ratio)

    pre_images = np.array([p[0] for p in image_pairs])
    post_images = np.array([p[1] for p in image_pairs])
    labels_arr = np.array(labels, dtype=np.float32).reshape(-1, 1)

    ds = tf.data.Dataset.from_tensor_slices(((pre_images, post_images), labels_arr))
    ds = ds.map(lambda imgs, lbl: preprocess_pair(imgs[0], imgs[1], lbl), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

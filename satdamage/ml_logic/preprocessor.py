import os
from io import BytesIO
import json
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
Merged preprocessor: lazy disk-based pipeline (no OOM) + team augmentation improvements.
Pipeline:
    1. find_image_pairs()       → list of (pre, post, label) paths
    2. split_pairs_by_event()   → train/val/test split by disaster event (no leakage)
    3. extract_crops_to_disk()  → saves PNGs to CROPS_DIR/{train,val,test}/{0,1}/
    4. build_dataset_from_dir() → lazy tf.data.Dataset, loads batch by batch from disk
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
    features = data.get("features", {}).get("xy", [])

    for feature in features:
        props  = feature.get("properties", {})
        damage = props.get("subtype", "un-classified")

        geom = None
        wkt_str = feature.get("wkt", "")
        if wkt_str:
            try:
                geom = shapely_wkt.loads(wkt_str)
            except Exception:
                pass

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
# 2. EXTRACTION D'UN CROP
# ─────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray:
    """Load PNG or TIFF as RGB uint8."""
    p = Path(path)
    if p.suffix.lower() in ['.tif', '.tiff']:
        with rasterio.open(path) as src:
            data = src.read()
            rgb = data[:3] if data.shape[0] >= 3 else np.repeat(data[[0]], 3, axis=0)
            img = rgb.transpose(1, 2, 0)
            return np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
    return np.array(Image.open(path).convert("RGB"))


def _scale_band(band: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
    if p98 == p2:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band.astype(np.float32) - p2) / (p98 - p2), 0, 1)


def _crop_and_scale(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    crop = image[y_min:y_max, x_min:x_max, :]
    rgb  = np.stack([_scale_band(crop[:,:,i]) for i in range(3)], axis=-1)
    pil  = Image.fromarray((rgb * 255).astype(np.uint8))
    pil  = pil.resize(CROP_SIZE, Image.Resampling.LANCZOS)
    return np.array(pil)


def _polygon_to_bbox(polygon, w, h, padding=CROP_PADDING):
    minx, miny, maxx, maxy = polygon.bounds
    x_min = max(0, int(minx) - padding)
    y_min = max(0, int(miny) - padding)
    x_max = min(w, int(maxx) + padding)
    y_max = min(h, int(maxy) + padding)
    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        return None
    return x_min, y_min, x_max, y_max


def polygon_to_pixel_bbox(polygon, image_width: int, image_height: int, padding: int = CROP_PADDING) -> Optional[Tuple[int, int, int, int]]:
    """Convert a shapely polygon to a pixel bounding box with padding and clipping."""
    minx, miny, maxx, maxy = polygon.bounds
    x_min = max(0, int(minx) - padding)
    y_min = max(0, int(miny) - padding)
    x_max = min(image_width,  int(maxx) + padding)
    y_max = min(image_height, int(maxy) + padding)
    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        return None
    return x_min, y_min, x_max, y_max


def crop_building(
    image:       np.ndarray,
    bbox:        Tuple[int, int, int, int],
    target_size: Tuple[int, int] = CROP_SIZE
) -> np.ndarray:
    """Crop a building from the image using the bounding box and apply percentile-based normalization."""
    x_min, y_min, x_max, y_max = bbox
    crop = image[y_min:y_max, x_min:x_max, :]

    def scale_band(band):
        p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
        if p98 == p2:
            return np.zeros_like(band, dtype=np.float32)
        return np.clip((band.astype(np.float32) - p2) / (p98 - p2), 0, 1)

    rgb = np.stack([scale_band(crop[:,:,0]), scale_band(crop[:,:,1]), scale_band(crop[:,:,2])], axis=-1)
    pil_crop = Image.fromarray((rgb * 255).astype(np.uint8)).resize(target_size, Image.Resampling.LANCZOS)
    return np.array(pil_crop)


# ─────────────────────────────────────────────
# 3. WORKER: PROCESS ONE PAIR → SAVE PNGs
# ─────────────────────────────────────────────

def _save_pair_crops(args):
    """
    Worker function: processes one image pair and saves crops as PNGs.
    Returns list of (png_path, label) tuples.
    """
    pre_path, post_path, label_path, out_dir, pair_idx = args
    results = []
    errors  = 0

    try:
        pre_img  = _load_image(pre_path)
        post_img = _load_image(post_path)
        h, w = pre_img.shape[:2]
        buildings = load_json_buildings(label_path)

        for b_idx, building in enumerate(buildings):
            damage = building["damage"]
            label  = DAMAGE_TO_BINARY.get(damage, None)
            if label is None:
                continue

            bbox = _polygon_to_bbox(building["polygon"], w, h)
            if bbox is None:
                continue

            pre_crop  = _crop_and_scale(pre_img,  bbox)
            post_crop = _crop_and_scale(post_img, bbox)

            # Stack pre+post horizontally into a single 128x256 PNG
            combined = np.concatenate([pre_crop, post_crop], axis=1)  # (128, 256, 3)

            label_dir = Path(out_dir) / str(label)
            label_dir.mkdir(parents=True, exist_ok=True)

            fname = f"{pair_idx:06d}_{b_idx:04d}.png"
            fpath = label_dir / fname
            Image.fromarray(combined).save(fpath)
            results.append((str(fpath), label))

    except Exception as e:
        errors = 1

    return results, errors


# ─────────────────────────────────────────────
# 4. SCAN DES PAIRES D'IMAGES
# ─────────────────────────────────────────────

def process_image_pair(
    pre_path: str,
    post_path: str,
    label_pre_path: str,
    label_post_path: str,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Processes a single image pair and its annotations to extract building crops and labels.
    Returns a list of tuples: (pre_crop, post_crop, label) for each building.
    Used by the in-memory pipeline (build_all_samples).
    """
    pre_img  = _load_image(pre_path)
    post_img = _load_image(post_path)
    h, w = pre_img.shape[:2]

    buildings_pre  = load_json_buildings(label_pre_path)
    buildings_post = load_json_buildings(label_post_path)
    samples = []

    for building_pre, building_post in zip(buildings_pre, buildings_post):
        damage = building_post["damage"]
        label  = DAMAGE_TO_BINARY.get(damage, None)
        if label is None:
            continue
        bbox_pre  = polygon_to_pixel_bbox(building_pre["polygon"],  w, h)
        bbox_post = polygon_to_pixel_bbox(building_post["polygon"], w, h)
        if bbox_pre is None or bbox_post is None:
            continue
        pre_crop  = crop_building(pre_img,  bbox_pre)
        post_crop = crop_building(post_img, bbox_post)
        samples.append((pre_crop, post_crop, label))

    return samples


# ─────────────────────────────────────────────
# 4-1. SCAN DES PAIRES D'IMAGES xView2 EN LOCAL
# ─────────────────────────────────────────────

def find_image_pairs(xview2_root: str) -> List[Dict[str, str]]:
    """
    Scans the xView2 dataset directory to find all valid image pairs and their corresponding labels.
    Returns a list of dicts with keys: 'pre_img', 'post_img', 'pre_label', 'post_label', 'event'.
    """
    pairs = []
    root  = Path(xview2_root)

    images_dirs = list(root.rglob("images"))
    if not images_dirs:
        print(f"[WARN] Aucun dossier 'images' trouve sous : {root}")
        return pairs

    for img_dir in images_dirs:
        label_dir = img_dir.parent / "labels"
        if not label_dir.exists():
            continue

        post_images = sorted(
            list(img_dir.glob("*_post_disaster.png")) +
            list(img_dir.glob("*_post_disaster.tif")) +
            list(img_dir.glob("*_post_disaster.tiff"))
        )

        for post_img_path in post_images:
            stem = post_img_path.stem
            ext  = post_img_path.suffix

            pre_stem        = stem.replace("_post_disaster", "_pre_disaster")
            pre_img_path    = img_dir / f"{pre_stem}{ext}"
            pre_label_path  = label_dir / f"{pre_stem}.json"
            post_label_path = label_dir / f"{stem}.json"

            if not pre_img_path.exists():
                print(f"[WARN] Image pre_disaster manquante : {stem}")
                continue
            if not pre_label_path.exists():
                print(f"[WARN] Label pre_disaster manquant : {pre_stem}")
                continue
            if not post_label_path.exists():
                print(f"[WARN] Label post_disaster manquant : {stem}")
                continue

            event = "_".join(stem.split("_")[:-2])
            pairs.append({
                "pre_img":    str(pre_img_path),
                "post_img":   str(post_img_path),
                "pre_label":  str(pre_label_path),
                "post_label": str(post_label_path),
                "event":      event,
            })

    print(f"[INFO] {len(pairs)} paires d'images trouvees")
    return pairs


# ─────────────────────────────────────────────
# 5. SPLIT PAIRS BY EVENT (no leakage)
# ─────────────────────────────────────────────

def split_pairs_by_event(
    pairs: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed:        int   = RANDOM_SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    from collections import defaultdict
    import random

    event_pairs = defaultdict(list)
    for p in pairs:
        event_pairs[p["event"]].append(p)

    events = list(event_pairs.keys())
    random.seed(seed)
    random.shuffle(events)

    n = len(events)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    train_events = events[:n_train]
    val_events   = events[n_train:n_train + n_val]
    test_events  = events[n_train + n_val:]

    train_pairs = [p for e in train_events for p in event_pairs[e]]
    val_pairs   = [p for e in val_events   for p in event_pairs[e]]
    test_pairs  = [p for e in test_events  for p in event_pairs[e]]

    print(f"[INFO] Event split -> Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)} pairs")
    return train_pairs, val_pairs, test_pairs


# ─────────────────────────────────────────────
# 4-2. SCAN DES PAIRES D'IMAGES xView2 SUR GCS
# ─────────────────────────────────────────────

def find_image_pairs_gcs(prefix: str = "") -> List[Dict[str, str]]:
    """
    Scans the xView2 dataset on a GCS bucket to find all valid image pairs and labels.
    Returns a list of dicts with keys: 'pre_img', 'post_img', 'pre_label', 'post_label', 'event'.
    """
    pairs = []

    all_blobs = set(
        blob.name for blob in CLIENT.list_blobs(BUCKET_NAME, prefix=prefix)
    )

    if not all_blobs:
        print(f"[WARN] Aucun fichier trouvé sous gs://{BUCKET_NAME}/{prefix}")
        return pairs

    post_blobs = sorted([
        b for b in all_blobs
        if "_post_disaster" in b and b.endswith((".png", ".tif", ".tiff"))
        and "/images/" in b
    ])

    for post_img_blob_name in post_blobs:
        parts = post_img_blob_name.rsplit("/", 1)
        img_dir_prefix = parts[0]
        filename = parts[1]

        stem, ext = filename.rsplit(".", 1)
        ext = f".{ext}"

        pre_stem             = stem.replace("_post_disaster", "_pre_disaster")
        label_dir_prefix     = img_dir_prefix.replace("/images", "/labels")
        pre_img_blob_name    = f"{img_dir_prefix}/{pre_stem}{ext}"
        pre_label_blob_name  = f"{label_dir_prefix}/{pre_stem}.json"
        post_label_blob_name = f"{label_dir_prefix}/{stem}.json"

        if pre_img_blob_name not in all_blobs:
            print(f"[WARN] Image pre_disaster manquante : {pre_img_blob_name}")
            continue
        if pre_label_blob_name not in all_blobs:
            print(f"[WARN] Label manquant : {pre_label_blob_name}")
            continue
        if post_label_blob_name not in all_blobs:
            print(f"[WARN] Label manquant : {post_label_blob_name}")
            continue

        event = "_".join(stem.split("_")[:-2])
        pairs.append({
            "pre_img":    pre_img_blob_name,
            "post_img":   post_img_blob_name,
            "pre_label":  pre_label_blob_name,
            "post_label": post_label_blob_name,
            "event":      event
        })

    print(f"[INFO] {len(pairs)} paires trouvées dans gs://{BUCKET_NAME}/{prefix}")
    return pairs


# ─────────────────────────────────────────────
# 6. EXTRACT ALL CROPS TO DISK (lazy pipeline — OOM fix)
# ─────────────────────────────────────────────

def extract_crops_to_disk(
    pairs:       List[Dict],
    out_dir:     str,
    split_name:  str,
    max_workers: int = 8,
    verbose:     bool = True,
) -> Tuple[int, int]:
    """
    Saves building crops as PNGs to out_dir/split_name/{0,1}/.
    Idempotent — skips if crops already exist.
    """
    split_dir = Path(out_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    existing = list(split_dir.rglob("*.png"))
    if existing:
        print(f"[INFO] {split_name}: {len(existing)} crops already on disk, skipping extraction.")
        return len(existing), 0

    args_list = [
        (p["pre_img"], p["post_img"], p["post_label"], str(split_dir), idx)
        for idx, p in enumerate(pairs)
    ]

    total_crops  = 0
    total_errors = 0
    completed    = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_save_pair_crops, args): args for args in args_list}

        for future in as_completed(futures):
            results, errors = future.result()
            total_crops  += len(results)
            total_errors += errors
            completed    += 1

            if verbose and completed % 50 == 0:
                print(f"  [{split_name}] {completed}/{len(pairs)} pairs processed — {total_crops} crops saved")

    n0 = len(list((split_dir / "0").glob("*.png"))) if (split_dir / "0").exists() else 0
    n1 = len(list((split_dir / "1").glob("*.png"))) if (split_dir / "1").exists() else 0
    print(f"  [{split_name}] Done: {total_crops} crops — Undamaged: {n0} | Damaged: {n1} | Errors: {total_errors}")

    return total_crops, total_errors


# ─────────────────────────────────────────────
# 7. AUGMENTATION
# ─────────────────────────────────────────────

def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies random augmentations. Used by the lazy disk pipeline."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ─────────────────────────────────────────────
# 8. BUILD LAZY tf.data.Dataset FROM DISK
# ─────────────────────────────────────────────

def _parse_image(path: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load a saved combined (128x256) PNG and split back into pre+post as 6-channel.
    Returns zeros + label=-1 for corrupted/malformed files (filtered downstream).
    """
    _sentinel = (tf.zeros((*CROP_SIZE, 6), dtype=tf.float32), tf.constant(-1.0))
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)

        # Guard: must be 3-D with positive height and width
        if img.shape.rank != 3 or img.shape[0] == 0 or img.shape[1] == 0:
            return _sentinel

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, [CROP_SIZE[0], CROP_SIZE[1] * 2])

        pre  = img[:, :CROP_SIZE[1], :]
        post = img[:, CROP_SIZE[1]:, :]

        combined = tf.concat([pre, post], axis=-1)
        combined = tf.image.resize(combined, CROP_SIZE)
        return combined, tf.cast(label, tf.float32)
    except Exception:
        return _sentinel


def build_dataset_from_dir(
    split_dir:  str,
    training:   bool = False,
    batch_size: int  = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Builds a lazy tf.data.Dataset from PNG paths on disk.
    Never loads all images into memory at once.
    """
    split_path = Path(split_dir)
    paths, labels = [], []

    for label in [0, 1]:
        label_dir = split_path / str(label)
        if not label_dir.exists():
            continue
        for png in label_dir.glob("*.png"):
            paths.append(str(png))
            labels.append(label)

    if not paths:
        raise FileNotFoundError(f"No crops found in {split_dir}")

    n0, n1 = labels.count(0), labels.count(1)
    print(f"[INFO] {split_path.name}: {len(paths)} crops — Undamaged: {n0} | Damaged: {n1}")

    if training:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)) \
                .shuffle(len(paths), seed=RANDOM_SEED) \
                .repeat()
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(
        lambda p, l: tf.py_function(
            func=lambda p, l: _parse_image(p.numpy().decode(), int(l.numpy())),
            inp=[p, l],
            Tout=[tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.filter(lambda img, lbl: lbl >= 0)

    # Hard runtime shape guard — drop any element where py_function returned
    # a malformed tensor (C++ errors can bypass Python try/except).
    _expected_shape = tf.constant([CROP_SIZE[0], CROP_SIZE[1], 6], dtype=tf.int32)
    ds = ds.filter(lambda img, lbl: tf.reduce_all(tf.equal(tf.shape(img), _expected_shape)))

    ds = ds.map(
        lambda img, lbl: (
            tf.ensure_shape(img, (*CROP_SIZE, 6)),
            tf.ensure_shape(lbl, ())
        )
    )

    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if training:
        import math
        ds.steps_per_epoch = math.ceil(len(paths) / batch_size)
    return ds


# ─────────────────────────────────────────────
# 9. CLASS WEIGHTS
# ─────────────────────────────────────────────

def compute_class_weights_from_dir(split_dir: str) -> Dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight
    split_path = Path(split_dir)
    labels = []
    for label in [0, 1]:
        label_dir = split_path / str(label)
        if label_dir.exists():
            n = len(list(label_dir.glob("*.png")))
            labels.extend([label] * n)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=np.array(labels)
    )
    return {0: float(weights[0]), 1: float(weights[1])}
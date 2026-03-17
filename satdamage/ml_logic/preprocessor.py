import os
from io import BytesIO
import json
import shutil
import numpy as np
import tensorflow as tf
import rasterio
from pathlib import Path
from PIL import Image
from shapely.geometry import shape
from shapely import wkt as shapely_wkt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from typing import Any, List, Tuple, Dict, Optional, TypeVar
from concurrent.futures import ProcessPoolExecutor, as_completed
from google.cloud import storage
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from satdamage.params import *

"""
Merged preprocessor: lazy disk-based pipeline (no OOM) + team augmentation improvements.
Pipeline:
    1. find_image_pairs()       → list of (pre, post, label) paths
    2. split_pairs_by_event()   → train/val/test split by disaster event (no leakage)
    3. extract_crops_to_disk()  → saves PNGs to CROPS_DIR/{train,val,test}/{0,1}/
    4. build_dataset_from_dir() → lazy tf.data.Dataset, loads batch by batch from disk
"""


SampleType = TypeVar("SampleType")


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


def _polygon_to_bbox(polygon, width: int, height: int, padding: int = CROP_PADDING) -> Optional[Tuple[int, int, int, int]]:
    """Convert a shapely polygon to a pixel bounding box with padding and clipping."""
    minx, miny, maxx, maxy = polygon.bounds
    x_min = max(0, int(minx) - padding)
    y_min = max(0, int(miny) - padding)
    x_max = min(width,  int(maxx) + padding)
    y_max = min(height, int(maxy) + padding)
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
            label = DAMAGE_TO_CLASS.get(damage, None) if MODEL_MODE == "multiclass" else DAMAGE_TO_BINARY.get(damage, None)
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
        label = DAMAGE_TO_CLASS.get(damage, None) if MODEL_MODE == "multiclass" else DAMAGE_TO_BINARY.get(damage, None)
        if label is None:
            continue
        bbox_pre  = _polygon_to_bbox(building_pre["polygon"],  w, h)
        bbox_post = _polygon_to_bbox(building_post["polygon"], w, h)
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

            if verbose and completed % 10 == 0:
                print(f"  [{split_name}] {completed}/{len(pairs)} pairs processed — {total_crops} crops saved")

    n_list = []
    for i in range(NUM_CLASSES if MODEL_MODE == "multiclass" else 2):
        n_list.append(len(list((split_dir/str(i)).glob("*.png"))) if (split_dir/str(i)).exists() else 0)
    class_summary = " - ".join(f"Class {cls}: {i}" for cls, i in enumerate(n_list))
    print(f"[{split_name}] Done: {total_crops} crops - {class_summary} - Errors: {total_errors}")

    return total_crops, total_errors


# ─────────────────────────────────────────────
# 7. AUGMENTATION
# ─────────────────────────────────────────────

def split_samples(
    samples: List[SampleType],
    labels: List[int],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED
) -> Tuple[List[SampleType], List[SampleType], List[SampleType], List[int], List[int], List[int]]:
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
    counts = Counter(train_labels)
    class_summary = " | ".join(f"Class {cls}: {counts.get(cls, 0)}" for cls in sorted(counts.keys()))
    print(f"Train : {len(train_samples):>6} crops - {class_summary}")
    counts = Counter(val_labels)
    class_summary = " | ".join(f"Class {cls}: {counts.get(cls, 0)}" for cls in sorted(counts.keys()))
    print(f"Val   : {len(val_samples):>6} crops - {class_summary}")
    counts = Counter(test_labels)
    class_summary = " | ".join(f"Class {cls}: {counts.get(cls, 0)}" for cls in sorted(counts.keys()))
    print(f"Test  : {len(test_samples):>6} crops - {class_summary}")

    return train_samples, val_samples, test_samples, train_labels, val_labels, test_labels


def split_crops_dir_stratified(
    source_dir: str,
    out_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> Dict[str, int]:
    """
    Stratified split of already-extracted PNG crops on disk into train/val/test.
    Moves files from source_dir/{label} to out_dir/{train,val,test}/{label}.
    """
    source_path = Path(source_dir)
    out_path = Path(out_dir)
    split_names = ("train", "val", "test")

    existing_counts = {
        split_name: len(list((out_path / split_name).rglob("*.png")))
        for split_name in split_names
    }
    if any(existing_counts.values()):
        print(
            "[INFO] Stratified crop split already present on disk, skipping split. "
            f"Train: {existing_counts['train']} | Val: {existing_counts['val']} | Test: {existing_counts['test']}"
        )
        return existing_counts

    if not source_path.exists():
        raise FileNotFoundError(f"Source crops directory not found: {source_dir}")

    paths, labels = [], []
    for label_dir in sorted(
        [path for path in source_path.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    ):
        for png in label_dir.glob("*.png"):
            paths.append(png)
            labels.append(int(label_dir.name))

    if not paths:
        raise FileNotFoundError(f"No crops found in {source_dir}")

    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_samples(
        samples=paths,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    split_data = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    for split_name, (split_paths, split_labels) in split_data.items():
        for src_path, label in zip(split_paths, split_labels):
            dst_dir = out_path / split_name / str(label)
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_dir / src_path.name))

    for label_dir in source_path.iterdir():
        if label_dir.is_dir():
            shutil.rmtree(label_dir, ignore_errors=True)

    try:
        source_path.rmdir()
    except OSError:
        pass

    final_counts = {
        split_name: len(list((out_path / split_name).rglob("*.png")))
        for split_name in split_names
    }
    print(
        "[INFO] Disk split complete -> "
        f"Train: {final_counts['train']} | Val: {final_counts['val']} | Test: {final_counts['test']}"
    )
    return final_counts


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
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def balance_dataset(samples, labels, majority_ratio: float = 2.0):
    """Balances samples by oversampling minority labels then undersampling the majority label.

    Works with generic sample containers (e.g. image-pairs in memory or file paths).
    """
    y = np.array(labels)
    class_counts = Counter(y)
    class_summary = " | ".join(
        f"Class {cls}: {class_counts.get(cls, 0)}"
        for cls in sorted(class_counts.keys())
    )
    print(f"\n** Before balancing: **\n {class_summary}")

    if len(class_counts) <= 1:
        return samples, labels

    majority_class = class_counts.most_common(1)[0][0]
    minority_classes = [cls for cls in class_counts if cls != majority_class]
    minority_total_before = sum(class_counts[cls] for cls in minority_classes)

    if minority_total_before * 10 < class_counts[majority_class]:
        strategy_oversampling = {
            cls: min(class_counts[cls] * 2, class_counts[majority_class])
            for cls in minority_classes
            if class_counts[cls] < class_counts[majority_class]
        }
    else:
        strategy_oversampling = {}

    sample_idx = np.arange(len(samples)).reshape(-1, 1)
    X_cur, y_cur = sample_idx, y

    if strategy_oversampling:
        over_strategy: Any = strategy_oversampling
        ros = RandomOverSampler(
            sampling_strategy=over_strategy,
            random_state=RANDOM_SEED,
        )
        ros_out = ros.fit_resample(X_cur, y_cur)
        X_cur, y_cur = ros_out[0], ros_out[1]
    else:
        print("\n** No oversampling applied (minority classes not extremely underrepresented) **")

    counts_after_over = Counter(y_cur)
    minority_total = sum(v for k, v in counts_after_over.items() if k != majority_class)
    target_majority = int(minority_total * majority_ratio)

    if target_majority > 0 and counts_after_over[majority_class] > target_majority:
        under_strategy: Any = {majority_class: target_majority}
        rus = RandomUnderSampler(
            sampling_strategy=under_strategy,
            random_state=RANDOM_SEED,
        )
        rus_out = rus.fit_resample(X_cur, y_cur)
        X_cur, y_cur = rus_out[0], rus_out[1]
    else:
        print("\n** No undersampling applied (majority class not extremely overrepresented after oversampling) **")

    balanced_samples = [samples[i] for i in X_cur.flatten()]
    balanced_labels = list(y_cur)

    balanced_class_counts = Counter(balanced_labels)
    class_summary = " | ".join(
        f"Class {cls}: {balanced_class_counts.get(cls, 0)}"
        for cls in sorted(balanced_class_counts.keys())
    )
    print(f"\n** After balancing: **\n {class_summary}\n")
    return balanced_samples, balanced_labels


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

    label_values = list(range(NUM_CLASSES)) if MODEL_MODE == "multiclass" else [0, 1]
    for label in label_values:
        label_dir = split_path / str(label)
        if not label_dir.exists():
            continue
        for png in label_dir.glob("*.png"):
            paths.append(str(png))
            labels.append(label)

    if not paths:
        raise FileNotFoundError(f"No crops found in {split_dir}")

    counts = Counter(labels)
    class_summary = " | ".join(
        f"Class {cls}: {counts.get(cls, 0)}"
        for cls in sorted(counts.keys())
    )
    print(f"[INFO] {split_path.name}: {len(paths)} crops — {class_summary}")

    if training :
        majority_count = max(counts.values())
        minority_total = sum(counts.values()) - majority_count
        # if minority_total * 3 < majority_count:
        paths, labels = balance_dataset(paths, labels, majority_ratio=BALANCE_MAJORITY_RATIO)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)) \
                .shuffle(len(paths), reshuffle_each_iteration=True)
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

    if MODEL_MODE == "multiclass":
        ds = ds.map(
            lambda img, lbl: (
                tf.ensure_shape(img, (*CROP_SIZE, 6)),
                tf.ensure_shape(
                    tf.one_hot(tf.cast(lbl, tf.int32), depth=NUM_CLASSES),
                    (NUM_CLASSES,)
                )
            )
        )
    else:
        ds = ds.map(
            lambda img, lbl: (
                tf.ensure_shape(img, (*CROP_SIZE, 6)),
                tf.ensure_shape(tf.reshape(lbl, (1,)), (1,))
            )
        )

    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        # .repeat() makes the stream infinite so the dataset never raises StopIteration
        # mid-epoch; .take(n_steps) gives Keras a known finite cardinality per epoch
        # so no "ran out of data" warning fires.  A fresh iterator is created each epoch
        # (Keras resets when steps_per_epoch is inferred, not explicit).
        n_steps = max(1, len(paths) // batch_size)
        ds = ds.repeat().batch(batch_size).take(n_steps).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────
# 9. CLASS WEIGHTS
# ─────────────────────────────────────────────

def compute_class_weights_from_dir(split_dir: str) -> Dict[int, float]:
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

# ─────────────────────────────────────────────
# 10. SPLIT PAIRS BY EVENT (no leakage)
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

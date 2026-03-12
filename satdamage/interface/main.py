import os
import numpy as np
from pathlib import Path
from satdamage.ml_logic.preprocessor import (
    find_image_pairs,
    split_pairs_by_event,
    extract_crops_to_disk,
    build_dataset_from_dir,
    compute_class_weights_from_dir,
)
from satdamage.ml_logic.model import train, evaluate
from satdamage.params import DATA_DIR

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CROPS_DIR   = os.environ.get("CROPS_DIR", "/home/pierre/crops")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 8))


# ─────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────

def build_xview2_datasets(xview2_root: str, crops_dir: str):
    """
    Full pipeline:
        1. Scan image pairs
        2. Split by event (no leakage)
        3. Extract crops to disk (idempotent — skips if already done)
        4. Compute class weights from disk
        5. Build lazy tf.data.Dataset
    """
    print("=" * 55)
    print("  SatDamage — xView2 Dataset Builder (Lazy)")
    print("=" * 55)

    # ── 1. Scan
    print("\n[1/5] Scan des paires d'images...")
    all_pairs = find_image_pairs(xview2_root)
    if not all_pairs:
        raise FileNotFoundError(f"Aucune paire trouvee dans {xview2_root}.")

    # ── 2. Split by event
    print("\n[2/5] Split par evenement (sans data leakage)...")
    train_pairs, val_pairs, test_pairs = split_pairs_by_event(all_pairs)

    # ── 3. Extract crops to disk
    print("\n[3/5] Extraction des crops vers le disque...")
    extract_crops_to_disk(train_pairs, crops_dir, "train", max_workers=MAX_WORKERS)
    extract_crops_to_disk(val_pairs,   crops_dir, "val",   max_workers=MAX_WORKERS)
    extract_crops_to_disk(test_pairs,  crops_dir, "test",  max_workers=MAX_WORKERS)

    # ── 4. Class weights
    print("\n[4/5] Calcul des class weights...")
    class_weights = compute_class_weights_from_dir(str(Path(crops_dir) / "train"))
    print(f"  class_weight[0] (undamaged) = {class_weights[0]:.3f}")
    print(f"  class_weight[1] (damaged)   = {class_weights[1]:.3f}")

    # ── 5. Build lazy datasets
    print("\n[5/5] Construction des tf.data.Dataset (lazy)...")
    train_ds = build_dataset_from_dir(str(Path(crops_dir) / "train"), training=True)
    val_ds   = build_dataset_from_dir(str(Path(crops_dir) / "val"),   training=False)
    test_ds  = build_dataset_from_dir(str(Path(crops_dir) / "test"),  training=False)

    print("\n" + "=" * 55)
    print("  Datasets prets (chargement lazy depuis disque)")
    print("=" * 55)

    return train_ds, val_ds, test_ds, class_weights


# ─────────────────────────────────────────────
# POINT D'ENTREE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    train_ds, val_ds, test_ds, class_weights = build_xview2_datasets(
        xview2_root=DATA_DIR,
        crops_dir=CROPS_DIR,
    )

    model, history = train(train_ds, val_ds, class_weights=class_weights)
    evaluate(model, test_ds)
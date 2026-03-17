import time
import gc
from pathlib import Path
import numpy as np
import tensorflow as tf

from satdamage.params import *
from satdamage.ml_logic.registry import *
from satdamage.ml_logic.model import train, evaluate, train_efficientnet
from satdamage.ml_logic.preprocessor import (
    find_image_pairs,
    find_image_pairs_gcs,
    extract_crops_to_disk,
    split_crops_dir_stratified,
    build_dataset_from_dir
)

# ─────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────

def build_xview2_datasets(xview2_root: str, crops_dir: str):
    """
    Full pipeline:
        1. Scan image pairs
        2. Extract all crops to disk (idempotent — skips if already done)
        3. Stratified split of saved crops into train/val/test
        4. Compute class weights from disk
        5. Build lazy tf.data.Dataset
    """
    print("=" * 55)
    print("  SatDamage — xView2 Dataset Builder (Lazy)")
    print("=" * 55)

    # ── 1. Scan
    start_time = time.time()
    print("\n[1/5] Scan des paires d'images...")
    print(f"  MODEL TARGET : {MODEL_TARGET}\n  DATA SOURCE : {xview2_root}")
    if MODEL_TARGET == "local":
        all_pairs = find_image_pairs(xview2_root)
    else:
        all_pairs = find_image_pairs_gcs(xview2_root)

    if not all_pairs:
        raise FileNotFoundError(
            f"Aucune paire trouvée dans {xview2_root}. "
            "Vérifiez la structure du dossier."
        )
    end_time = time.time()
    print(f"Temps : {end_time - start_time:.2f} secondes")

    # ── 2. Extract all crops to disk once
    print("\n[2/5] Extraction de tous les crops vers le disque...")
    start_time = time.time()
    extract_crops_to_disk(all_pairs, crops_dir, "all", max_workers=MAX_WORKERS)
    end_time = time.time()
    print(f"Temps : {end_time - start_time:.2f} secondes")

    # ── 3. Stratified split on saved crops to preserve label distribution
    print("\n[3/5] Split stratifie des crops sauvegardes...")
    start_time = time.time()
    split_crops_dir_stratified(
        source_dir=str(Path(crops_dir) / "all"),
        out_dir=crops_dir,
    )
    end_time = time.time()
    print(f"Temps : {end_time - start_time:.2f} secondes")

    # # ── 4. Class weights — passed to model.fit() to compensate for imbalance
    # print("\n[4/5] Distribution des classes (class_weight)...")
    # class_weights = compute_class_weights_from_dir(str(Path(crops_dir) / "train"))
    # print(f"  class_weight[0] (undamaged) = {class_weights[0]:.3f}")
    # print(f"  class_weight[1] (damaged)   = {class_weights[1]:.3f}")

    # ── 5. Build lazy tf.data.Datasets (never loads all images into memory)
    print("\n[5/5] Construction des tf.data.Dataset (lazy)...")
    start_time = time.time()
    train_ds = build_dataset_from_dir(str(Path(crops_dir) / "train"), training=True)
    val_ds   = build_dataset_from_dir(str(Path(crops_dir) / "val"),   training=False)
    test_ds  = build_dataset_from_dir(str(Path(crops_dir) / "test"),  training=False)
    end_time = time.time()
    print(f"Temps : {end_time - start_time:.2f} secondes")

    print("=" * 55)
    print("  Datasets prets (chargement lazy depuis disque)")
    print("=" * 55)
    return train_ds, val_ds, test_ds

# ─────────────────────────────────────────────
# POINT D'ENTREE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if DATA_DIR is None:
        raise ValueError("DATA_DIR environment variable must be set.")

    # If CROPS_DIR is not empty, delete it to ensure a clean slate (idempotent — won't fail if it doesn't exist)
    if CROPS_DIR and Path(CROPS_DIR).exists():
        print(f"Cleaning up existing crops directory at {CROPS_DIR}...")
        for item in Path(CROPS_DIR).glob("*"):
            if item.is_dir():
                tf.io.gfile.rmtree(str(item))
            else:
                item.unlink()

    # ── 1. Build datasets
    train_ds, val_ds, test_ds = build_xview2_datasets(
        xview2_root=DATA_DIR,
        crops_dir=CROPS_DIR,
    )

     # ── 2. Train & Evaluate
    if MODEL_ARCHITECTURE == "efficientnet":
        model, history_warmup, history_finetune = train_efficientnet(train_ds, val_ds)
    elif MODEL_ARCHITECTURE in ("cnn_dual", "cnn_concat"):
        model, history = train(train_ds, val_ds)
    else:
        raise ValueError(f"Architecture inconnue : {MODEL_ARCHITECTURE}")

    print("Training done \n")
    evaluate(model, test_ds)

    metrics = {
        "f1_damaged":           0.0,    # fill after training
        "precision_damaged":    0.0,
        "recall_damaged":       0.0,
        "accuracy":             0.0,
        "val_auc":              0.0,
        "best_epoch":           0,
        "dataset":              "xBD challenge full",
        "model":                f"EfficientNetV2B0 binary v1" if MODEL_ARCHITECTURE == "efficientnet" else "CNN dual-stream siamese binary v5",
        "crop_size":            128,
        "notes":                "EfficientNetV2B0 2-phase warmup+finetune, focal loss, no class_weight" if MODEL_ARCHITECTURE == "efficientnet" else "dual-stream residual+SE, binary, crop-level stratified split"
    }
    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/run_{MODEL_ARCHITECTURE}_v1.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*31}\n****    GREAT SUCCESS !    ****\n{'='*31}\n")

    # explicit cleanup to avoid AtomicFunction __del__ noise at interpreter shutdown
    del model, train_ds, val_ds, test_ds
    tf.keras.backend.clear_session()
    gc.collect()

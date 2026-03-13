import os
import numpy as np
import time
import gc
import tensorflow as tf
from pathlib import Path
from satdamage.ml_logic.preprocessor import (
    find_image_pairs,
    find_image_pairs_gcs,
    split_pairs_by_event,
    extract_crops_to_disk,
    build_dataset_from_dir,
    compute_class_weights_from_dir,
)
from satdamage.ml_logic.model import train, evaluate, train_efficientnet
from satdamage.params import MODEL_TARGET, DATA_DIR, BATCH_SIZE, MODEL_ARCHITECTURE
from satdamage.ml_logic.registry import *

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
    start_time = time.time()
    print("\n[1/5] Scan des paires d'images...")
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

    # ── 2. Split by event (no data leakage between disaster events)
    print("\n[2/5] Split par evenement (sans data leakage)...")
    train_pairs, val_pairs, test_pairs = split_pairs_by_event(all_pairs)

    # ── 3. Extract crops to disk (lazy pipeline — idempotent)
    print("\n[3/5] Extraction des crops vers le disque...")
    extract_crops_to_disk(train_pairs, crops_dir, "train", max_workers=MAX_WORKERS)
    extract_crops_to_disk(val_pairs,   crops_dir, "val",   max_workers=MAX_WORKERS)
    extract_crops_to_disk(test_pairs,  crops_dir, "test",  max_workers=MAX_WORKERS)

    # ── 4. Class weights (informatif uniquement — le rééquilibrage est fait
    #       par balanced sampling dans build_dataset_from_dir)
    print("\n[4/5] Distribution des classes (info)...")
    class_weights = compute_class_weights_from_dir(str(Path(crops_dir) / "train"))
    print(f"  class_weight[0] (undamaged) = {class_weights[0]:.3f}  (sklearn balanced, pour info)")
    print(f"  class_weight[1] (damaged)   = {class_weights[1]:.3f}  (balanced sampling actif)")

    # ── 5. Build lazy tf.data.Datasets (never loads all images into memory)
    print("\n[5/5] Construction des tf.data.Dataset (lazy)...")
    train_ds = build_dataset_from_dir(str(Path(crops_dir) / "train"), training=True)
    val_ds   = build_dataset_from_dir(str(Path(crops_dir) / "val"),   training=False)
    test_ds  = build_dataset_from_dir(str(Path(crops_dir) / "test"),  training=False)

    print("\n" + "=" * 55)
    print("  Datasets prets (chargement lazy depuis disque)")
    print("=" * 55)

    # steps_per_epoch: 2 passes through the minority class per epoch
    n_damaged = len(list((Path(crops_dir) / "train" / "1").glob("*.png")))
    steps_per_epoch = (2 * n_damaged) // BATCH_SIZE
    print(f"[INFO] steps_per_epoch = {steps_per_epoch} (2 × {n_damaged} damaged / {BATCH_SIZE})")

    return train_ds, val_ds, test_ds, steps_per_epoch


# ─────────────────────────────────────────────
# POINT D'ENTREE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    train_ds, val_ds, test_ds, steps_per_epoch = build_xview2_datasets(
        xview2_root=DATA_DIR,
        crops_dir=CROPS_DIR,
    )

    # ── Lancement de l'entraînement
    if MODEL_ARCHITECTURE == "efficientnet":
        model, history_warmup, history_finetune = train_efficientnet(train_ds, val_ds)
    elif MODEL_ARCHITECTURE in ("cnn_dual", "cnn_concat"):
        model, history = train(train_ds, val_ds, steps_per_epoch=steps_per_epoch)
    else:
        raise ValueError(f"Architecture inconnue : {MODEL_ARCHITECTURE}")

    print("Train done \n")

    evaluate(model, test_ds)
    print(f"\n{'='*31}\n****    GREAT SUCCESS !    ****\n{'='*31}\n")

    # explicit cleanup to avoid AtomicFunction __del__ noise at interpreter shutdown
    del model, train_ds, val_ds, test_ds
    tf.keras.backend.clear_session()
    gc.collect()

import time
import gc
from pathlib import Path
import numpy as np
import tensorflow as tf

from google.cloud import storage
from satdamage.params import (
    MODEL_TARGET,
    MODEL_ARCHITECTURE,
    MODEL_FILENAME,
    DATA_DIR,
    CROPS_DIR,
    BUCKET_NAME,
    MAX_WORKERS
)
from satdamage.ml_logic.registry import load_model_light
from satdamage.ml_logic.model import evaluate_light
from satdamage.ml_logic.preprocessor import (
    find_image_pairs,
    find_image_pairs_gcs,
    extract_crops_to_disk,
    build_dataset_from_dir
)

# ─────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────

def build_xview2_datasets_light(xview2_root: str, crops_dir: str):
    """
    Full pipeline:
        1. Scan image pairs
        2. Extract all crops to disk (idempotent — skips if already done)
        3. Build lazy tf.data.Dataset
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
    extract_crops_to_disk(all_pairs, crops_dir, "test", max_workers=MAX_WORKERS)
    end_time = time.time()
    print(f"Temps : {end_time - start_time:.2f} secondes")

    # ── 3. Build lazy tf.data.Datasets (never loads all images into memory)
    print("\n[3/5] Construction des tf.data.Dataset (lazy)...")
    start_time = time.time()
    test_dataset  = build_dataset_from_dir(str(Path(crops_dir) / "test"),  training=False)
    end_time = time.time()
    print(f"Temps : {end_time - start_time:.2f} secondes")

    print("=" * 55)
    print("  Datasets prets (chargement lazy depuis disque)")
    print("=" * 55)
    return test_dataset

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
    test_ds = build_xview2_datasets_light(
        xview2_root=DATA_DIR,
        crops_dir=CROPS_DIR,
    )

    # ── 2. Load model
    start_time = time.time()
    model = load_model_light(model_name=MODEL_FILENAME)
    end_time = time.time()
    print(f"Temps : {end_time - start_time:.2f} secondes")

    # ── 3. Evaluate
    start_time = time.time()
    metrics_light, metrics = evaluate_light(model, test_ds)
    end_time = time.time()
    print(f"Temps : {end_time - start_time:.2f} secondes")

    start_time = time.time()
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    # Save metrics in the cloud
    if metrics is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_filename = f"run_{MODEL_ARCHITECTURE}_{timestamp}.json"
        blob = bucket.blob(f"metrics/{metrics_filename}")
        blob.upload_from_string(
            data=json.dumps(metrics),
            content_type="application/json"
        )
        print("✅ Results saved to GCS")
    else:
        print("⚠️ No metrics to save")
    end_time = time.time()
    print(f"Temps : {end_time - start_time:.2f} secondes")

    # explicit cleanup to avoid AtomicFunction __del__ noise at interpreter shutdown
    del model, test_ds
    tf.keras.backend.clear_session()
    gc.collect()

    print(f"\n{'='*31}\n****    GREAT SUCCESS !    ****\n{'='*31}\n")

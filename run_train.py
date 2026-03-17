#!/usr/bin/env python3
"""
Training script optimized for Mac M5 with Apple Silicon GPU.
Usage: python run_train.py
"""

import os
import sys
import time
import gc
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from satdamage.ml_logic.gpu_config import setup_gpu
from satdamage.ml_logic.preprocessor import (
    find_image_pairs,
    extract_crops_to_disk,
    split_crops_dir_stratified,
    build_dataset_from_dir,
)
from satdamage.ml_logic.model import train, evaluate, train_efficientnet
from satdamage.params import (
    TRAIN_DATA_DIR, 
    TEST_DATA_DIR,
    CROPS_DIR, 
    MODEL_ARCHITECTURE, 
    MAX_WORKERS,
    BATCH_SIZE,
)


def main():
    print("=" * 60)
    print("  SatDamage Training - Mac M5 Optimized")
    print("=" * 60)
    
    # Setup GPU/M5 optimization
    setup_gpu()
    
    # Determine data directory
    xview2_root = TRAIN_DATA_DIR if Path(TRAIN_DATA_DIR).exists() else None
    
    if xview2_root is None:
        raise FileNotFoundError(f"Training data not found at {TRAIN_DATA_DIR}")
    
    print(f"\n📁 Data directory: {xview2_root}")
    print(f"📁 Crops directory: {CROPS_DIR}")
    print(f"🖥️  Model: {MODEL_ARCHITECTURE}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔧 Max workers: {MAX_WORKERS}")
    
    # Build datasets
    print("\n" + "=" * 60)
    print("  Building datasets...")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Scan image pairs
    print("\n[1/5] Scanning image pairs...")
    all_pairs = find_image_pairs(xview2_root)
    if not all_pairs:
        raise FileNotFoundError(f"No image pairs found in {xview2_root}")
    print(f"Found {len(all_pairs)} image pairs")
    
    # 2. Extract crops to disk
    print("\n[2/5] Extracting crops to disk...")
    extract_crops_to_disk(all_pairs, CROPS_DIR, "all", max_workers=MAX_WORKERS)
    
    # 3. Stratified split
    print("\n[3/5] Splitting into train/val/test...")
    split_crops_dir_stratified(
        source_dir=str(Path(CROPS_DIR) / "all"),
        out_dir=CROPS_DIR,
    )
    
    # 4. Build tf.data.Datasets
    print("\n[4/5] Building TensorFlow datasets...")
    train_ds = build_dataset_from_dir(str(Path(CROPS_DIR) / "train"), training=True)
    val_ds = build_dataset_from_dir(str(Path(CROPS_DIR) / "val"), training=False)
    test_ds = build_dataset_from_dir(str(Path(CROPS_DIR) / "test"), training=False)
    
    prep_time = time.time() - start_time
    print(f"\nDataset preparation completed in {prep_time:.1f}s")
    
    # 5. Train
    print("\n" + "=" * 60)
    print("  Starting training...")
    print("=" * 60)
    
    if MODEL_ARCHITECTURE == "efficientnet":
        model, history_warmup, history_finetune = train_efficientnet(train_ds, val_ds)
    elif MODEL_ARCHITECTURE in ("cnn_dual", "cnn_concat"):
        model, history = train(train_ds, val_ds)
    else:
        raise ValueError(f"Unknown architecture: {MODEL_ARCHITECTURE}")
    
    # 6. Evaluate
    print("\n" + "=" * 60)
    print("  Evaluating model...")
    print("=" * 60)
    results = evaluate(model, test_ds)
    
    print("\n" + "=" * 60)
    print("  ✅ Training complete!")
    print("=" * 60)
    
    # Cleanup
    del model, train_ds, val_ds, test_ds
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()

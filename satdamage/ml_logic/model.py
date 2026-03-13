"""
CNN Binaire - Classification de dommages de bâtiments
Dataset: xView2 (xDB)
Classes: 0 = non endommagé | 1 = endommagé
Input:   paires d'images pré/post-catastrophe (6 canaux)
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from satdamage.params import *

# ─────────────────────────────────────────────
# 1. BLOC CONVOLUTIONNEL DE BASE
# ─────────────────────────────────────────────

def conv_block(x, filters, kernel_size=3, use_bn=True, dropout=0.0, name=None):
    """
    Bloc Conv → BatchNorm → ReLU → (Dropout optionnel)
    Brique élémentaire du réseau.
    """
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name=f"{name}_conv" if name else None
    )(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop" if name else None)(x)
    return x


# ─────────────────────────────────────────────
# 2. ARCHITECTURE CNN PRINCIPALE
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 2.1 6 CHANNELS CONCATENATION
# ─────────────────────────────────────────────

def build_damage_cnn_concat(input_shape=(128, 128, 6)):
    """
    CNN à 4 blocs convolutionnels pour classification binaire.

    Architecture :
        Input (128×128×6)
            ↓
        Bloc 1 : Conv(32) → Conv(32) → MaxPool → Dropout
            ↓
        Bloc 2 : Conv(64) → Conv(64) → MaxPool → Dropout
            ↓
        Bloc 3 : Conv(128) → Conv(128) → MaxPool → Dropout
            ↓
        Bloc 4 : Conv(256) → Conv(256) → GlobalAvgPool
            ↓
        Dense(256) → BN → ReLU → Dropout
            ↓
        Dense(128) → BN → ReLU → Dropout
            ↓
        Output(1) → Sigmoid  [0 = intact, 1 = endommagé]
    """
    inputs = layers.Input(shape=input_shape, name="input_pre_post")

    # ── Bloc 1 — Extraction de features basses fréquences (32 filtres)
    x = conv_block(inputs, 32, name="b1_c1")
    x = conv_block(x, 32, name="b1_c2")
    x = layers.MaxPooling2D(pool_size=2, name="b1_pool")(x)
    x = layers.Dropout(0.25, name="b1_drop")(x)
    # Sortie : 64×64×32

    # ── Bloc 2 — Features intermédiaires (64 filtres)
    x = conv_block(x, 64, name="b2_c1")
    x = conv_block(x, 64, name="b2_c2")
    x = layers.MaxPooling2D(pool_size=2, name="b2_pool")(x)
    x = layers.Dropout(0.25, name="b2_drop")(x)
    # Sortie : 32×32×64

    # ── Bloc 3 — Features hautes fréquences (128 filtres)
    x = conv_block(x, 128, name="b3_c1")
    x = conv_block(x, 128, name="b3_c2")
    x = layers.MaxPooling2D(pool_size=2, name="b3_pool")(x)
    x = layers.Dropout(0.35, name="b3_drop")(x)
    # Sortie : 16×16×128

    # ── Bloc 4 — Features sémantiques profondes (256 filtres)
    x = conv_block(x, 256, name="b4_c1")
    x = conv_block(x, 256, name="b4_c2")
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    # Sortie : vecteur 256-dim (remplace Flatten → moins de params, moins d'overfit)

    # ── Head de classification
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(DROPOUT_RATE, name="fc1_drop")(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.ReLU(name="fc2_relu")(x)
    x = layers.Dropout(0.3, name="fc2_drop")(x)

    # Sortie binaire
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="CNN_DamageClassifier")
    return model



# ─────────────────────────────────────────────
# 2.2 DUAL STREAM
# ─────────────────────────────────────────────

#### HELPER BLOCKS ####

def se_block(x, reduction=16, name="se"):
    """Squeeze-and-Excitation: recalibrates channel importance."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Dense(max(1, filters // reduction), activation="relu",  name=f"{name}_fc1")(se)
    se = layers.Dense(filters,                      activation="sigmoid", name=f"{name}_fc2")(se)
    se = layers.Reshape((1, 1, filters), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def res_block(x, filters, dropout=0.25, name="res"):
    """Conv → BN → ReLU → Conv → BN → SE → Add (residual)."""
    shortcut = x
    x = conv_block(x, filters, name=f"{name}_c1")
    x = conv_block(x, filters, name=f"{name}_c2")
    x = se_block(x, name=f"{name}_se")
    # Project shortcut to matching filter depth if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, padding="same",
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            name=f"{name}_proj"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


#### DUAL STREAM ARCHITECTURE ####

def build_damage_cnn_dual(input_shape=(128, 128, 6)):
    """
    Dual-stream CNN with residual blocks and SE attention for binary damage classification.

    Architecture :
        Input (128×128×6)  →  split into Pre (×3) and Post (×3)
              ↓                              ↓
        Shared-weight siamese encoder (3 residual stages + SE)
              ↓                              ↓
        Merge: Concat(pre_feat, post_feat, |post_feat - pre_feat|)
              ↓
        Fusion block : ResBlock(256) → GlobalAvgPool
              ↓
        Dense(256) → BN → ReLU → Dropout
              ↓
        Dense(128) → BN → ReLU → Dropout
              ↓
        Output(1) → Sigmoid
    """
    inputs = layers.Input(shape=input_shape, name="input_pre_post")

    # ── Split 6-channel input into pre / post
    pre  = inputs[:, :, :, :3]
    post = inputs[:, :, :, 3:]

    # ── Siamese encoder (identical structure, independent weights for pre and post)
    def encoder(x, prefix):
        # Stage 1 — 32 filters, 64×64
        x = res_block(x, 32,  dropout=0.25, name=f"{prefix}_s1")
        x = layers.MaxPooling2D(pool_size=2, name=f"{prefix}_pool1")(x)

        # Stage 2 — 64 filters, 32×32
        x = res_block(x, 64,  dropout=0.25, name=f"{prefix}_s2")
        x = layers.MaxPooling2D(pool_size=2, name=f"{prefix}_pool2")(x)

        # Stage 3 — 128 filters, 16×16
        x = res_block(x, 128, dropout=0.35, name=f"{prefix}_s3")
        x = layers.MaxPooling2D(pool_size=2, name=f"{prefix}_pool3")(x)
        return x  # (batch, 16, 16, 128)

    pre_feat  = encoder(pre,  "pre")
    post_feat = encoder(post, "post")

    # ── Explicit change signal: |post - pre|
    diff = layers.Subtract(name="subtract")([post_feat, pre_feat])
    diff = tf.keras.ops.abs(diff)

    # ── Merge: concatenate all three feature maps → (batch, 16, 16, 384)
    merged = layers.Concatenate(name="merge")([pre_feat, post_feat, diff])

    # ── Fusion block — 256 filters with residual
    x = res_block(merged, 256, dropout=0.35, name="fusion")
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    # → (batch, 256)

    # ── Classification head
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(DROPOUT_RATE, name="fc1_drop")(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.ReLU(name="fc2_relu")(x)
    x = layers.Dropout(0.3, name="fc2_drop")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs, outputs, name="CNN_DamageClassifier_DualStream")

# ─────────────────────────────────────────────
# 3. COMPILATION & CALLBACKS
# ─────────────────────────────────────────────

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE
        ),
        # loss=tf.keras.losses.BinaryCrossentropy(),
        loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=1.0),  # Focal Loss pour mieux gérer le déséquilibre
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
            tf.keras.metrics.F1Score(name="f1", threshold=0.5, average="micro")
            ]
    )
    return model


def get_callbacks():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    return [
        # Arrêt si pas d'amélioration sur la val_loss
        EarlyStopping(
            monitor="val_auc_pr",
            mode="max",
            min_delta=1e-3,
            patience=12,
            start_from_epoch=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Réduction du LR si plateau
        ReduceLROnPlateau(
            monitor="val_auc_pr",
            mode="max",
            factor=0.5,
            patience=4,
            min_delta=5e-4,
            cooldown=1,
            min_lr=1e-6,
            verbose=1
        ),
        # Sauvegarde du meilleur modèle
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_auc_pr",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        # TensorBoard
        TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1
        ),
    ]



# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train(train_ds, val_ds, class_weights=None):
    if MODEL_ARCHITECTURE=="cnn_concat":
        model = build_damage_cnn_concat()
    elif MODEL_ARCHITECTURE=="cnn_dual":
        model = build_damage_cnn_dual()

    model = compile_model(model)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )
    return model, history


# ─────────────────────────────────────────────
# 5. ÉVALUATION
# ─────────────────────────────────────────────

def evaluate(model, test_ds, threshold=0.5):
    """
    Évalue le modèle et affiche la matrice de confusion + F1.
    """

    y_true, y_pred_prob = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_prob.extend(preds.flatten())
        y_true.extend(labels.numpy().flatten())

    y_pred = (np.array(y_pred_prob) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)

    print("\n── Rapport de classification ──")
    print(classification_report(
        y_true, y_pred,
        target_names=["non-endommagé", "endommagé"]
    ))

    print("── Matrice de confusion ──")
    print(confusion_matrix(y_true, y_pred))

    f1 = f1_score(y_true, y_pred)
    print(f"\nF1-score (endommagé) : {f1:.4f}")

    return y_pred, y_pred_prob

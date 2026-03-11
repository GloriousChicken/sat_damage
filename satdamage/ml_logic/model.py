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


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

class Config:
    # Données
    IMAGE_SIZE     = (128, 128)        # Taille de resize des crops bâtiments
    INPUT_CHANNELS = 6                 # 3 (pré) + 3 (post)
    NUM_CLASSES    = 1                 # Binaire → sigmoid
    BATCH_SIZE     = 32
    EPOCHS         = 50

    # Chemins xView2
    TRAIN_DIR = "data/train"
    VAL_DIR   = "data/val"
    TEST_DIR  = "data/test"

    # Entraînement
    LEARNING_RATE  = 1e-3
    WEIGHT_DECAY   = 1e-4
    DROPOUT_RATE   = 0.5

    # Sauvegarde
    CHECKPOINT_PATH = "checkpoints/cnn_damage_best.keras"
    LOG_DIR         = "logs/cnn_damage"


# ─────────────────────────────────────────────
# 2. BLOC CONVOLUTIONNEL DE BASE
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
        kernel_regularizer=regularizers.l2(Config.WEIGHT_DECAY),
        name=f"{name}_conv" if name else None
    )(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop" if name else None)(x)
    return x


# ─────────────────────────────────────────────
# 3. ARCHITECTURE CNN PRINCIPALE
# ─────────────────────────────────────────────

def build_damage_cnn(input_shape=(128, 128, 6)):
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
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(Config.WEIGHT_DECAY), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name="fc1_drop")(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(Config.WEIGHT_DECAY), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.ReLU(name="fc2_relu")(x)
    x = layers.Dropout(0.3, name="fc2_drop")(x)

    # Sortie binaire
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="CNN_DamageClassifier")
    return model



# ─────────────────────────────────────────────
# 6. COMPILATION & CALLBACKS
# ─────────────────────────────────────────────

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=Config.LEARNING_RATE
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )
    return model


def get_callbacks():
    os.makedirs(os.path.dirname(Config.CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    return [
        # Arrêt si pas d'amélioration sur la val_loss
        EarlyStopping(
            monitor="val_auc",
            patience=8,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        # Réduction du LR si plateau
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        # Sauvegarde du meilleur modèle
        ModelCheckpoint(
            filepath=Config.CHECKPOINT_PATH,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        # TensorBoard
        TensorBoard(
            log_dir=Config.LOG_DIR,
            histogram_freq=1
        ),
    ]



# ─────────────────────────────────────────────
# 7. ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train(train_ds, val_ds, class_weights=None):
    model = build_damage_cnn()
    model = compile_model(model)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )
    return model, history


# ─────────────────────────────────────────────
# 8. ÉVALUATION
# ─────────────────────────────────────────────

def evaluate(model, test_ds, threshold=0.5):
    """
    Évalue le modèle et affiche la matrice de confusion + F1.
    """

    y_true, y_pred_prob = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_prob.extend(preds.flatten())
        y_true.extend(labels.numpy())

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

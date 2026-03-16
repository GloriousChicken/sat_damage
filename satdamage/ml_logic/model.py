"""
CNN Binaire - Classification de dommages de bâtiments
Dataset: xView2 (xDB)
Classes: 0 = non endommagé | 1 = endommagé
Input:   paires d'images pré/post-catastrophe (6 canaux)
"""

import numpy as np
import os
from datetime import datetime
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from satdamage.params import *


class BinaryF1Score(tf.keras.metrics.Metric):
    """F1 metric compatible with binary sigmoid output (None, 1) and flat labels (None,)."""
    def __init__(self, threshold=0.5, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall    = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * p * r / (p + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


# ─────────────────────────────────────────────
# 1. ARCHITECTURE EfficientNetV2B0
# ─────────────────────────────────────────────

def build_damage_efficientnet(input_shape=(128, 128, 6), freeze_backbone = True):
    """
    Construit le modèle EfficientNetV2B0.
    Architecture complète :
    ─────────────────────────────────────────────────────────
    Input (128×128×6)
        ↓
    Conv1×1 + BN + ReLU          ← projection 6→3 canaux (18 params)
        ↓
    EfficientNetV2B0 backbone    ← pré-entraîné ImageNet (~5.9M params)
      (include_top=False)          feature map : 4×4×1280
        ↓
    GlobalAveragePooling2D       ← vecteur 1280-dim
        ↓
    BatchNormalization
        ↓
    Dropout(0.4)
        ↓
    Dense(256) → ReLU → Dropout(0.3)
        ↓
    Dense(1) → Sigmoid           ← P(endommagé) ∈ [0, 1]
    ─────────────────────────────────────────────────────────
    Args:
        input_shape      : (H, W, C) — C=6 pour paires pré/post
        freeze_backbone  : True  = backbone gelé    (phase warm-up)
                           False = backbone dégelé  (fine-tuning)
    """
    inputs = layers.Input(shape=input_shape, name="input_pre_post_6ch")

    # ── Étape 1 : Projection 6 → 3 canaux
    """
    Projette l'input 6 canaux (pré+post) vers 3 canaux via Conv1×1.
    """
    x = layers.Conv2D(
        filters     = 3,
        kernel_size = 1,
        padding     = "same",
        use_bias    = False,
        name        = "proj_conv1x1"
    )(inputs)
    x = layers.BatchNormalization(name="proj_bn")(x)
    x = layers.ReLU(name="proj_relu")(x)

    # ── Étape 2 : Backbone EfficientNetV2B0
    backbone = EfficientNetV2B0(
        include_top           = False,
        weights               = "imagenet",
        input_shape           = (input_shape[0], input_shape[1], 3),
        include_preprocessing = False,
        # include_preprocessing=True : le backbone applique sa propre
        # normalisation [0,255] → [-1,1].
    )
    backbone.trainable = not freeze_backbone
    x = backbone(x, training=not freeze_backbone)
    # Sortie backbone : (batch, 4, 4, 1280) pour input 128×128

    # ── Étape 3 : Head de classification binaire
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.4, name="head_drop")(x)

    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="fc1_drop")(x)

    if MODEL_MODE == "multiclass":
        # Sortie softmax 4 classes
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    else:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)



    return Model(inputs, outputs,
                 name="EfficientNetV2B0_DamageClassifier")

# ─────────────────────────────────────────────
# 2. ENTRAÎNEMENT EN 2 PHASES
# ─────────────────────────────────────────────

def train_efficientnet(train_ds, val_ds, class_weights=None):
    """
    Entraîne EfficientNetV2B0 en deux phases successives.

    Phase 1 — Warm-up (backbone gelé, 10 epochs)
    ──────────────────────────────────────────────
    Seuls la couche de projection Conv1×1 et le head Dense s'entraînent.
    Le backbone conserve ses poids ImageNet intacts.
    Objectif : initialiser correctement le head avant de toucher
    au backbone — sans cette phase, les gradients aléatoires du head
    risquent de détériorer les features pré-entraînées.

    Phase 2 — Fine-tuning (40 dernières couches dégelées, 30 epochs)
    ──────────────────────────────────────────────────────────────────
    Learning rate réduit (5e-5) pour adapter doucement le backbone
    aux images satellites sans "oublier" ImageNet (catastrophic forgetting).
    AdamW avec weight_decay pour régulariser les nouveaux poids.

    Retourne : (model, history_warmup, history_finetune)
    """
    # ── Phase 1 : Warm-up
    print("=" * 55)
    print("  Phase 1 — Warm-up  (backbone gelé)")
    print(f"  AdamW lr={LR_WARMUP} | "
          f"{EPOCHS_WARMUP} epochs max")
    print("=" * 55)

    model = build_damage_efficientnet(freeze_backbone=True)
    model = compile_model(model, LR_WARMUP)
    model.summary()

    history_warmup = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS_WARMUP,
        callbacks       = get_callbacks(phase="warmup"),
        verbose         = 2,
    )

    # ── Phase 2 : Fine-tuning
    print("\n" + "=" * 55)
    print("  Phase 2 — Fine-tuning  (backbone partiellement dégelé)")
    print(f"  AdamW lr={LR_FINETUNE} | "
          f"{EPOCHS_FINETUNE} epochs max")
    print("=" * 55)

    # Localiser le backbone dans le modèle
    backbone_layer = next(
        (l for l in model.layers if "efficientnetv2" in l.name.lower()),
        None
    )

    if backbone_layer is not None:
        backbone_layer.trainable = True

        # Geler toutes les couches SAUF les N dernières
        n_layers     = len(backbone_layer.layers)
        freeze_until = n_layers - UNFREEZE_LAYERS

        for i, layer in enumerate(backbone_layer.layers):
            layer.trainable = (i >= freeze_until)

        n_frozen    = sum(1 for l in backbone_layer.layers if not l.trainable)
        n_trainable = sum(1 for l in backbone_layer.layers if l.trainable)
        print(f"  Backbone : {n_frozen} couches gelées | "
              f"{n_trainable} dégelées")
    else:
        print("  [WARN] Backbone non trouvé — fine-tuning du modèle entier")
        model.trainable = True

    # Recompiler obligatoire après modification de trainable
    model = compile_model(model, LR_FINETUNE)

    history_finetune = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS_FINETUNE,
        callbacks       = get_callbacks(phase="finetune"),
        verbose         = 2,
    )

    return model, history_warmup, history_finetune

# ─────────────────────────────────────────────
# 1. ARCHITECTURE CNN PRINCIPALE & BLOC CONVOLUTIONNEL DE BASE
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
# 1.1 6 CHANNELS CONCATENATION
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
    x = layers.Dropout(0.5, name="fc1_drop")(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.ReLU(name="fc2_relu")(x)
    x = layers.Dropout(0.3, name="fc2_drop")(x)

    if MODEL_MODE == "multiclass":
        # Sortie softmax 4 classes
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    else:
        # Sortie binaire
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="CNN_DamageClassifier")
    return model



# ─────────────────────────────────────────────
# 1.2 DUAL STREAM
# ─────────────────────────────────────────────

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
        x = res_block(x, 32,  dropout=0.35, name=f"{prefix}_s1")
        x = layers.MaxPooling2D(pool_size=2, name=f"{prefix}_pool1")(x)

        # Stage 2 — 64 filters, 32×32
        x = res_block(x, 64,  dropout=0.35, name=f"{prefix}_s2")
        x = layers.MaxPooling2D(pool_size=2, name=f"{prefix}_pool2")(x)

        # Stage 3 — 128 filters, 16×16
        x = res_block(x, 128, dropout=0.45, name=f"{prefix}_s3")
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
    x = layers.Dropout(0.5, name="fc1_drop")(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.ReLU(name="fc2_relu")(x)
    x = layers.Dropout(0.3, name="fc2_drop")(x)

    if MODEL_MODE == "multiclass":
        # Sortie softmax 4 classes
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    else:
        # Sortie binaire
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs, outputs, name="CNN_DualStream_DamageClassifier")

# ─────────────────────────────────────────────
# 2. COMPILATION & CALLBACKS
# ─────────────────────────────────────────────

def compile_model(model: Model, learning_rate: float) -> Model:
    """
    Compile avec AdamW et métriques.
    Appelée deux fois pour EfficientNet : une fois au warm-up, une fois au fine-tuning.
    """
    opt = tf.keras.optimizers.AdamW(
        learning_rate = learning_rate,
        weight_decay  = 1e-2,
        beta_1        = 0.9,
        beta_2        = 0.999,
        epsilon       = 1e-7,
    )
    met = [
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        # AUC-PR est plus informatif qu'AUC-ROC sur données déséquilibrées
        tf.keras.metrics.F1Score(name="f1", threshold=0.5, average="micro")
    ]
    if MODEL_MODE == "multiclass":
        model.compile(
            optimizer = opt,
            loss      = tf.keras.losses.CategoricalFocalCrossentropy(gamma=1.0, from_logits=False),
            metrics   = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")] + met
        )
    else:
        model.compile(
            optimizer = opt,
            # loss      = tf.keras.losses.BinaryCrossentropy(),
            loss      = tf.keras.losses.BinaryFocalCrossentropy(gamma=FOCAL_GAMMA, label_smoothing=0.05),
            metrics   = [tf.keras.metrics.BinaryAccuracy(name="accuracy")] + met
        )
    return model

def get_callbacks(phase: str = "warmup"):
    """
    Callbacks adaptés à chaque phase d'entraînement.
    Args:
        phase : "warmup" ou "finetune" only used for efficientnet
    """
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    if MODEL_ARCHITECTURE == "efficientnet":
        log_dir = os.path.join(LOG_DIR, phase)
        patience_es = 6  if phase == "warmup" else 10
        patience_lr = 3  if phase == "warmup" else 5
        ckpt_path = CHECKPOINT_PATH
        monitor = "val_auc_pr"
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = CHECKPOINT_PATH.replace(".keras", f"_{run_id}.keras")
        log_dir = LOG_DIR
        patience_es = 12
        patience_lr = 4
        monitor = "val_f1"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[Checkpoint] {ckpt_path}")

    return [
        EarlyStopping(
            monitor=monitor,
            mode="max",
            min_delta=1e-3,
            patience=20,           # was 12 — more room to train
            start_from_epoch=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.7,
            patience=6,            # was 4 — give more room before cutting LR
            min_delta=5e-4,
            cooldown=1,
            min_lr=1e-6,
            mode="max",
            verbose=1
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]

# ─────────────────────────────────────────────
# 3. ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train(train_ds, val_ds, class_weight=None):
    """
    Entraîne le modèle CNN. Balanced sampling replaced by class_weight —
    real distribution during training, gradient re-weighting compensates for imbalance.
    """
    if MODEL_ARCHITECTURE == "cnn_concat":
        model = build_damage_cnn_concat()
    elif MODEL_ARCHITECTURE == "cnn_dual":
        model = build_damage_cnn_dual()
    else:
        raise ValueError(f"Architecture CNN inconnue: {MODEL_ARCHITECTURE}. Utilisez 'cnn_concat' ou 'cnn_dual'.")

    model = compile_model(model, LEARNING_RATE)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=get_callbacks(),
        verbose=2          # was 1; \r updates break tail -f
    )
    return model, history


# ─────────────────────────────────────────────
# 4. ÉVALUATION
# ─────────────────────────────────────────────

def find_best_threshold(y_true, y_pred_prob):
    """
    Parcourt les seuils de 0.10 à 0.90 et retourne celui qui maximise le F1
    sur l'ensemble fourni (typiquement le jeu de validation).
    """
    thresholds = np.arange(0.10, 0.91, 0.05)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (np.array(y_pred_prob) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"[Threshold sweep] Meilleur seuil : {best_t:.2f} → F1 = {best_f1:.4f}")
    return float(best_t)


def evaluate(model, test_ds, threshold=None):
    """
    Évalue le modèle et affiche la matrice de confusion + F1.
    Si threshold=None, recherche automatiquement le seuil optimal sur test_ds
    avant d'afficher le rapport final.
    """
    if MODEL_MODE == "multiclass":
        y_true, y_prob_all = [], []
        for images, labels in test_ds:
            probs = model.predict(images, verbose=0)   # shape (batch, 4)
            y_prob_all.extend(probs)
            y_true.extend(labels.numpy())

        y_prob_all = np.array(y_prob_all)
        y_pred     = np.argmax(y_prob_all, axis=1)
        y_true     = np.array(y_true, dtype=int)

        print(f"\n── Rapport de classification {model.name} ──")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print("── Matrice de confusion ──")
        print(f"  TN={tn:>5}  FP={fp:>5}")
        print(f"  FN={fn:>5}  TP={tp:>5}")
        f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"F1 macro    : {f1_macro:.4f}   ← indicateur principal (classes déséquilibrées)")
        print(f"F1 weighted : {f1_weighted:.4f}")

        return {
            "y_true":       y_true,
            "y_pred":       y_pred,
            "y_prob":       y_prob_all,
            "f1_macro":     f1_macro,
            "f1_weighted":  f1_weighted,
        }

    else:
        y_true, y_prob = [], []
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_prob.extend(preds.flatten())
            y_true.extend(labels.numpy())

        y_pred = (np.array(y_prob) >= threshold).astype(int)
        y_true = np.array(y_true).astype(int)

        print(f"\n── Rapport de classification {model.name} ──")
        print(classification_report(y_true, y_pred, target_names=["no-damage", "damaged"]))

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print("── Matrice de confusion ──")
        print(f"  TN={tn:>5}  FP={fp:>5}")
        print(f"  FN={fn:>5}  TP={tp:>5}")
        print(f"\nF1-score  (damaged) : {f1_score(y_true, y_pred):.4f}")
        print(f"Threshold utilisé    : {threshold}")

        return y_pred, y_prob

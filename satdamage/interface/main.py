import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from satdamage.ml_logic.preprocessor import build_dataset, find_image_pairs, split_samples, build_all_samples
from satdamage.ml_logic.model import train, evaluate, train_efficientnet, evaluate_efficientnet
from satdamage.params import DATA_DIR

# ─────────────────────────────────────────────
# 1. GESTION DU DÉSÉQUILIBRE DE CLASSES
# ─────────────────────────────────────────────

def compute_class_weights(labels):
    """
    xView2 contient beaucoup plus de bâtiments non endommagés.
    Les class weights compensent ce déséquilibre.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels
    )
    return {0: weights[0], 1: weights[1]}



# ─────────────────────────────────────────────
# 2. PIPELINE COMPLET
# ─────────────────────────────────────────────

def build_xview2_datasets(xview2_root: str):
    """
    Pipeline complet xView2 → tf.data.Dataset.

    Étapes :
        1. Scan des paires d'images pré/post
        2. Split par événement (sans data leakage)
        3. Extraction des crops de bâtiments
        4. Calcul des class weights
        5. Construction des tf.data.Dataset

    Retourne :
        train_ds, val_ds, test_ds  : tf.data.Dataset
        class_weights              : dict {0: w0, 1: w1}

    Exemple d'utilisation :
        train_ds, val_ds, test_ds, cw = build_xview2_datasets("xview2/")
        model, history = train(train_ds, val_ds, class_weight=cw)
        evaluate(model, test_ds)
    """

    print("=" * 55)
    print("  xView2 Dataset Builder")
    print("=" * 55)

    # ── 1. Scan
    print("\n[1/5] Scan des paires d'images...")
    all_pairs = find_image_pairs(xview2_root)
    if not all_pairs:
        raise FileNotFoundError(
            f"Aucune paire trouvée dans {xview2_root}. "
            "Vérifiez la structure du dossier."
        )

    # ── 2. Extraction de TOUS les crops (avant le split)
    print("\n[2/5] Extraction de tous les crops...")
    all_samples, all_labels = build_all_samples(all_pairs[:100], verbose=True)
    if not all_samples:
        raise ValueError("Aucun crop extrait. Vérifiez les données.")

    # ── 3. Split stratifié des crops
    print("\n[3/5] Split stratifié train / val / test...")
    train_samples, val_samples, test_samples, train_labels, val_labels, test_labels = split_samples(
        all_samples, all_labels
    )

    # ── 4. Class weights
    print("\n[4/5] Calcul des class weights...")
    class_weights = compute_class_weights(train_labels)
    print(f"  class_weight[0] = {class_weights[0]:.3f}")
    print(f"  class_weight[1] = {class_weights[1]:.3f}")

    # ── 5. tf.data.Dataset
    print("\n[5/5] Construction des tf.data.Dataset...")
    train_ds = build_dataset(train_samples, train_labels, training=True)
    val_ds   = build_dataset(val_samples,   val_labels,   training=False)
    test_ds  = build_dataset(test_samples,  test_labels,  training=False)

    print("\n" + "=" * 55)
    print("  Datasets prêts !")
    print(f"  Train  : {len(train_labels):>6} bâtiments")
    print(f"  Val    : {len(val_labels):>6} bâtiments")
    print(f"  Test   : {len(test_labels):>6} bâtiments")
    print("=" * 55)

    return train_ds, val_ds, test_ds, class_weights



# ─────────────────────────────────────────────
# 3. POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Pipeline automatique complet
    train_ds, val_ds, test_ds, class_weights = build_xview2_datasets(
        xview2_root=DATA_DIR
    )

    # ── Lancement de l'entraînement
    # CNN PRINCIPALE
    # model, history = train(train_ds, val_ds, class_weights=class_weights)
    # evaluate(model, test_ds)
    # EFFICIENTNET
    model, history_warmup, history_finetune = train_efficientnet(train_ds, val_ds, class_weights=class_weights)
    evaluate(model, test_ds)

    # ── Debug pas à pas (décommenter si besoin)
    #
    # all_pairs = find_image_pairs("xview2/")
    # train_pairs, val_pairs, test_pairs = stratified_event_split(all_pairs)
    # train_img_pairs, train_labels = build_all_samples(train_pairs[:5])
    # visualize_samples(train_img_pairs, train_labels, n=6)

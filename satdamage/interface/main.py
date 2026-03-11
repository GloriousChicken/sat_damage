import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from satdamage.ml_logic.preprocessor import build_dataset, find_image_pairs, pairs_split, build_all_samples
from satdamage.ml_logic.model import train, evaluate


# ─────────────────────────────────────────────
# 5. GESTION DU DÉSÉQUILIBRE DE CLASSES
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
# 8. PIPELINE COMPLET
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

    # ── 2. Split par événement
    print("\n[2/5] Split train / val / test...")
    train_pairs, val_pairs, test_pairs = pairs_split(all_pairs)

    # ── 3. Extraction des crops
    print("\n[3/5] Extraction des crops — Train...")
    train_img_pairs, train_labels = build_all_samples(train_pairs)

    print("\n      Extraction des crops — Val...")
    val_img_pairs,   val_labels   = build_all_samples(val_pairs,  verbose=False)

    print("\n      Extraction des crops — Test...")
    test_img_pairs,  test_labels  = build_all_samples(test_pairs, verbose=False)

    # ── 4. Class weights
    print("\n[4/5] Calcul des class weights...")
    class_weights = compute_class_weights(train_labels)
    print(f"  class_weight[0] = {class_weights[0]:.3f}")
    print(f"  class_weight[1] = {class_weights[1]:.3f}")

    # ── 5. tf.data.Dataset
    print("\n[5/5] Construction des tf.data.Dataset...")
    train_ds = build_dataset(train_img_pairs, train_labels, training=False)
    val_ds   = build_dataset(val_img_pairs,   val_labels,   training=False)
    test_ds  = build_dataset(test_img_pairs,  test_labels,  training=False)

    print("\n" + "=" * 55)
    print("  Datasets prêts !")
    print(f"  Train  : {len(train_labels):>6} bâtiments")
    print(f"  Val    : {len(val_labels):>6} bâtiments")
    print(f"  Test   : {len(test_labels):>6} bâtiments")
    print("=" * 55)

    return train_ds, val_ds, test_ds, class_weights



# ─────────────────────────────────────────────
# 10. POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Pipeline automatique complet
    train_ds, val_ds, test_ds, class_weights = build_xview2_datasets(
        xview2_root="data/balanced_samples"
    )

    # ── Lancement de l'entraînement

    model, history = train(train_ds, val_ds, class_weights=class_weights)
    evaluate(model, test_ds)

    # ── Debug pas à pas (décommenter si besoin)
    #
    # all_pairs = find_image_pairs("xview2/")
    # train_pairs, val_pairs, test_pairs = stratified_event_split(all_pairs)
    # train_img_pairs, train_labels = build_all_samples(train_pairs[:5])
    # visualize_samples(train_img_pairs, train_labels, n=6)

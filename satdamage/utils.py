
# ─────────────────────────────────────────────
# 9. VISUALISATION DE DEBUG
# ─────────────────────────────────────────────

def visualize_samples(
    image_pairs: List[Tuple],
    labels:      List[int],
    n:           int = 6,
    save_path:   str = "sample_crops.png"
):
    """
    Affiche n paires de crops (pré | post) avec leur label.
    Utile pour vérifier la qualité des extractions avant entraînement.
    """
    import matplotlib.pyplot as plt

    n       = min(n, len(labels))
    indices = random.sample(range(len(labels)), n)

    fig, axes = plt.subplots(n, 2, figsize=(7, n * 2.5))
    if n == 1:
        axes = [axes]

    label_names = {0: "non-endommagé", 1: "endommagé"}
    colors      = {0: "green",         1: "red"}

    for row, idx in enumerate(indices):
        pre_crop, post_crop = image_pairs[idx]
        lbl = labels[idx]

        axes[row][0].imshow(pre_crop)
        axes[row][0].set_title("Pré-disaster", fontsize=9)
        axes[row][0].axis("off")

        axes[row][1].imshow(post_crop)
        axes[row][1].set_title(
            f"Post — {label_names[lbl]}",
            color=colors[lbl], fontsize=9, fontweight="bold"
        )
        axes[row][1].axis("off")

    plt.suptitle("Échantillons extraits de xView2", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Visualisation sauvegardée → {save_path}")

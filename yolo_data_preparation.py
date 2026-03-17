"""
yolo26_data_preparation.py
===========================
Prépare les données xView2 (images POST-disaster uniquement)
pour entraîner deux modèles YOLO26 distincts :

  ┌─────────────────────────────────────────────────────────────────┐
  │  MODÈLE 1 — YOLO26-seg  (segmentation des bâtiments)           │
  │    Input  : image post-disaster complète                        │
  │    Output : masques de segmentation des bâtiments               │
  │    Labels : polygones JSON → format YOLO segmentation           │
  │    Classe : 1 seule classe "building" (id=0)                    │
  └─────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────┐
  │  MODÈLE 2 — YOLO26-cls  (classification du niveau de dommage)  │
  │    Input  : crop d'un bâtiment individuel (post-disaster)       │
  │    Output : no-damage / minor-damage / major-damage / destroyed │
  │    Labels : crops extraits via polygones JSON + label "subtype" │
  │    Classe : 4 classes de dommages                               │
  └─────────────────────────────────────────────────────────────────┘

Structure de sortie sur disque :
──────────────────────────────────────────────────────────────────────
yolo26_dataset/
├── segmentation/                    ← YOLO26-seg
│   ├── images/
│   │   ├── train/  hurricane-florence_00000001_post_disaster.png
│   │   ├── val/
│   │   └── test/
│   ├── labels/                      ← format YOLO seg (.txt)
│   │   ├── train/  hurricane-florence_00000001_post_disaster.txt
│   │   ├── val/
│   │   └── test/
│   └── dataset.yaml                 ← config YOLO
│
└── classification/                  ← YOLO26-cls
    ├── train/
    │   ├── no-damage/       crop_001.png
    │   ├── minor-damage/    crop_002.png
    │   ├── major-damage/    crop_003.png
    │   └── destroyed/       crop_004.png
    ├── val/
    │   └── ...
    └── test/
        └── ...

Format YOLO segmentation (un fichier .txt par image) :
    class_id  x1 y1 x2 y2 ... xn yn   (coordonnées normalisées [0,1])
    0  0.412 0.318 0.445 0.318 0.445 0.356 0.412 0.356
    0  0.512 0.418 ...
    (une ligne par bâtiment, class_id=0 pour "building")

Format YOLO classification (structure de dossiers) :
    train/no-damage/img_001.png
    train/minor-damage/img_002.png
    ...

Pré-requis :
    pip install ultralytics shapely pillow
"""

import json
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from shapely import wkt as shapely_wkt
from shapely.geometry import shape as shapely_shape
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════

class PrepConfig:
    # ── Sources xView2
    XVIEW2_ROOT    = "data"
    SOURCE_SPLITS  = ["train", "tier3"]

    # ── Sortie
    OUTPUT_ROOT    = "yolo_dataset"
    SEG_DIR        = "yolo_dataset/segmentation"
    CLS_DIR        = "yolo_dataset/classification"

    # ── Split
    TRAIN_RATIO    = 0.70
    VAL_RATIO      = 0.15
    RANDOM_SEED    = 42

    # ── Segmentation
    # Une seule classe pour YOLO26-seg : "building"
    SEG_CLASS_ID   = 0
    SEG_CLASS_NAME = "building"

    # ── Classification
    # 4 classes de dommages
    CLASS_NAMES    = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    DAMAGE_TO_LABEL = {
        "no-damage":     0,
        "minor-damage":  1,
        "major-damage":  2,
        "destroyed":     3
    }

    # ── Crops de bâtiments
    CROP_PADDING   = 10     # pixels de marge autour du bbox
    CROP_MIN_SIZE  = 20     # taille minimale du crop en pixels (sinon ignoré)

    # ── Polygones
    # Nombre minimal de points pour un polygone valide
    MIN_POLYGON_POINTS = 3


# ═══════════════════════════════════════════════════════════
# 2. PARSING JSON xView2
# ═══════════════════════════════════════════════════════════

def load_json_buildings(json_path: str) -> List[Dict]:
    """
    Parse un fichier JSON post-disaster xView2.
    Supporte les formats WKT et GeoJSON.

    Retourne :
        [{"polygon": shapely_geom, "damage": "major-damage"}, ...]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    buildings = []
    for feature in data.get("features", {}).get("xy", []):
        props  = feature.get("properties", {})
        damage = props.get("subtype", "un-classified")
        geom   = None

        # Format WKT (plus courant dans xView2)
        wkt_str = feature.get("wkt", "")
        if wkt_str:
            try:
                geom = shapely_wkt.loads(wkt_str)
            except Exception:
                pass

        # Fallback GeoJSON
        if geom is None:
            try:
                geom = shapely_shape(feature.get("geometry", {}))
            except Exception:
                continue

        if geom is None or geom.is_empty or not geom.is_valid:
            continue

        buildings.append({"polygon": geom, "damage": damage})

    return buildings


# ═══════════════════════════════════════════════════════════
# 3. CONVERSION POLYGONE → FORMAT YOLO SEGMENTATION
# ═══════════════════════════════════════════════════════════

def polygon_to_yolo_seg(
    polygon,
    image_w: int,
    image_h: int,
    class_id: int = PrepConfig.SEG_CLASS_ID,
) -> Optional[str]:
    """
    Convertit un polygone Shapely en ligne YOLO segmentation.

    Format YOLO seg :
        class_id  x1 y1 x2 y2 ... xn yn
        (coordonnées normalisées par la taille de l'image, valeurs dans [0,1])

    Exemple :
        0  0.412 0.318 0.445 0.318 0.445 0.356 0.412 0.356

    Args:
        polygon  : shapely Polygon
        image_w  : largeur de l'image en pixels
        image_h  : hauteur de l'image en pixels
        class_id : identifiant de classe YOLO (0 = building)

    Retourne :
        Chaîne formatée prête à écrire dans le fichier .txt,
        ou None si le polygone est invalide / trop petit.
    """
    try:
        # Extraire les coordonnées du contour extérieur
        if polygon.geom_type == "MultiPolygon":
            # Prendre le plus grand polygone
            polygon = max(polygon.geoms, key=lambda p: p.area)

        coords = list(polygon.exterior.coords)

        if len(coords) < PrepConfig.MIN_POLYGON_POINTS:
            return None

        # Normaliser chaque point dans [0, 1]
        normalized = []
        for x, y in coords[:-1]:    # coords[-1] == coords[0] (polygone fermé)
            nx = max(0.0, min(1.0, x / image_w))
            ny = max(0.0, min(1.0, y / image_h))
            normalized.extend([nx, ny])

        # Vérifier que la surface normalisée est non nulle
        if len(normalized) < 6:     # minimum 3 points = 6 valeurs
            return None

        coords_str = " ".join(f"{v:.6f}" for v in normalized)
        return f"{class_id} {coords_str}"

    except Exception:
        return None


def image_to_yolo_seg_label(
    json_path:  str,
    image_w:    int,
    image_h:    int,
) -> List[str]:
    """
    Génère toutes les lignes YOLO seg pour une image.

    Retourne une liste de lignes (une par bâtiment valide),
    prêtes à écrire dans le fichier .txt correspondant.

    Note : les bâtiments "un-classified" sont INCLUS dans le fichier
    de segmentation (on veut détecter tous les bâtiments, quelle que
    soit leur classe de dommage). Le label de dommage n'est pas utilisé
    ici — c'est le rôle du modèle de classification.
    """
    buildings = load_json_buildings(json_path)
    lines     = []

    for b in buildings:
        line = polygon_to_yolo_seg(b["polygon"], image_w, image_h)
        if line is not None:
            lines.append(line)

    return lines


# ═══════════════════════════════════════════════════════════
# 4. EXTRACTION DES CROPS POUR LA CLASSIFICATION
# ═══════════════════════════════════════════════════════════

def polygon_to_crop(
    image:   np.ndarray,
    polygon,
    padding: int = PrepConfig.CROP_PADDING,
) -> Optional[np.ndarray]:
    """
    Extrait le crop d'un bâtiment depuis une image post-disaster.

    Args:
        image   : np.array (H, W, 3) uint8 — image post-disaster
        polygon : shapely Polygon — contour du bâtiment
        padding : marge en pixels

    Retourne :
        np.array (h, w, 3) uint8 — crop brut (sans resize),
        ou None si le crop est trop petit.
    """
    h, w = image.shape[:2]
    minx, miny, maxx, maxy = polygon.bounds

    x0 = max(0, int(minx) - padding)
    y0 = max(0, int(miny) - padding)
    x1 = min(w, int(maxx) + padding)
    y1 = min(h, int(maxy) + padding)

    if (x1 - x0) < PrepConfig.CROP_MIN_SIZE or (y1 - y0) < PrepConfig.CROP_MIN_SIZE:
        return None

    return image[y0:y1, x0:x1]


# ═══════════════════════════════════════════════════════════
# 5. SCAN DES IMAGES POST-DISASTER xView2
# ═══════════════════════════════════════════════════════════

def find_post_disaster_images(
    xview2_root:   str       = PrepConfig.XVIEW2_ROOT,
    source_splits: List[str] = PrepConfig.SOURCE_SPLITS,
) -> List[Dict[str, str]]:
    """
    Parcourt xView2 et retourne toutes les images POST-disaster avec
    leur fichier d'annotation JSON.

    Différence vs l'ancien builder (paires pré/post) :
        AVANT : {"pre_img": ..., "post_img": ..., "post_label": ...}
        APRÈS : {"post_img": ..., "post_label": ..., "event": ...}

    Retourne :
        [{
            "post_img":   "/path/xxx_post_disaster.png",
            "post_label": "/path/xxx_post_disaster.json",
            "event":      "hurricane-florence",
        }, ...]
    """
    records = []
    root    = Path(xview2_root)

    for split in source_splits:
        img_dir   = root / split / "images"
        label_dir = root / split / "labels"

        if not img_dir.exists():
            print(f"[WARN] Dossier introuvable : {img_dir}")
            continue

        for post_img_path in sorted(img_dir.glob("*_post_disaster.png")):
            stem            = post_img_path.stem
            post_label_path = label_dir / f"{stem}.json"

            if not post_label_path.exists():
                continue

            event = "_".join(stem.split("_")[:-2])
            records.append({
                "post_img":   str(post_img_path),
                "post_label": str(post_label_path),
                "event":      event,
            })

    print(f"[INFO] {len(records)} images post-disaster trouvées.")
    return records


# ═══════════════════════════════════════════════════════════
# 6. SPLIT PAR ÉVÉNEMENT (sans data leakage)
# ═══════════════════════════════════════════════════════════

def stratified_event_split(
    records:     List[Dict],
    train_ratio: float = PrepConfig.TRAIN_RATIO,
    val_ratio:   float = PrepConfig.VAL_RATIO,
    seed:        int   = PrepConfig.RANDOM_SEED,
) -> Tuple[List, List, List]:
    """
    Split au niveau des événements pour éviter le data leakage.
    Toutes les images d'un même événement (même catastrophe) restent
    dans le même split.
    """
    random.seed(seed)
    events = defaultdict(list)
    for r in records:
        events[r["event"]].append(r)

    names   = sorted(events.keys())
    random.shuffle(names)
    n       = len(names)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ev = names[:n_train]
    val_ev   = names[n_train:n_train + n_val]
    test_ev  = names[n_train + n_val:]

    train_r = [r for e in train_ev for r in events[e]]
    val_r   = [r for e in val_ev   for r in events[e]]
    test_r  = [r for e in test_ev  for r in events[e]]

    print(f"\n[INFO] Split par événements (seed={seed}) :")
    print(f"  Train : {len(train_ev):>3} événements → {len(train_r):>5} images")
    print(f"  Val   : {len(val_ev):>3} événements → {len(val_r):>5} images")
    print(f"  Test  : {len(test_ev):>3} événements → {len(test_r):>5} images")

    return train_r, val_r, test_r


# ═══════════════════════════════════════════════════════════
# 7. CONSTRUCTION DU DATASET YOLO26-SEG
# ═══════════════════════════════════════════════════════════

def build_segmentation_dataset(
    records:    List[Dict],
    split_name: str,
    output_dir: str = PrepConfig.SEG_DIR,
    verbose:    bool = True,
) -> int:
    """
    Génère le dataset YOLO26-seg pour un split donné.

    Pour chaque image post-disaster :
        1. Copie l'image dans output_dir/images/{split}/
        2. Génère le fichier .txt de labels YOLO seg dans output_dir/labels/{split}/

    Toutes les instances de bâtiments (class_id=0) sont incluses,
    quelle que soit leur classe de dommage.

    Args:
        records    : liste de dicts de find_post_disaster_images()
        split_name : "train", "val" ou "test"
        output_dir : répertoire racine du dataset YOLO seg
        verbose    : affiche la progression si True

    Retourne :
        Nombre total de bâtiments annotés dans ce split.
    """
    img_out_dir   = Path(output_dir) / "images" / split_name
    label_out_dir = Path(output_dir) / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    total_buildings = 0
    errors          = 0

    for i, record in enumerate(records):
        if verbose and i % 100 == 0:
            print(f"  [SEG/{split_name}] {i+1}/{len(records)} — {record['event']}")

        try:
            post_img_path   = record["post_img"]
            post_label_path = record["post_label"]
            stem            = Path(post_img_path).stem

            # ── Dimensions de l'image (sans charger le pixel data)
            with Image.open(post_img_path) as img:
                image_w, image_h = img.size

            # ── Générer les lignes YOLO seg
            lines = image_to_yolo_seg_label(post_label_path, image_w, image_h)

            if not lines:
                continue    # Image sans bâtiment annoté → ignorer

            # ── Copier l'image
            dst_img = img_out_dir / f"{stem}.png"
            shutil.copy2(post_img_path, dst_img)

            # ── Écrire le fichier label YOLO
            dst_lbl = label_out_dir / f"{stem}.txt"
            dst_lbl.write_text("\n".join(lines) + "\n")

            total_buildings += len(lines)

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  [ERROR] {Path(record['post_img']).name} → {e}")

    if verbose:
        print(f"  [SEG/{split_name}] {len(records)} images | "
              f"{total_buildings} bâtiments | {errors} erreurs")

    return total_buildings


def write_segmentation_yaml(output_dir: str = PrepConfig.SEG_DIR):
    """
    Écrit le fichier dataset.yaml nécessaire pour l'entraînement YOLO26-seg.

    Ce fichier indique à YOLO :
        - Les chemins vers les splits train/val/test
        - Le nombre de classes
        - Les noms des classes

    Utilisation :
        yolo train model=yolo26n-seg.pt data=yolo26_dataset/segmentation/dataset.yaml
    """
    yaml_content = f"""# YOLO26-seg — Segmentation de bâtiments xView2 (post-disaster)
# Généré automatiquement par yolo26_data_preparation.py

path:  {Path(output_dir).resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: building
"""
    yaml_path = Path(output_dir) / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[INFO] dataset.yaml segmentation → {yaml_path}")


# ═══════════════════════════════════════════════════════════
# 8. CONSTRUCTION DU DATASET YOLO26-CLS
# ═══════════════════════════════════════════════════════════

def build_classification_dataset(
    records:    List[Dict],
    split_name: str,
    output_dir: str  = PrepConfig.CLS_DIR,
    verbose:    bool = True,
) -> Counter:
    """
    Génère le dataset YOLO26-cls pour un split donné.

    Pour chaque bâtiment annoté dans les JSON post-disaster :
        1. Extrait le crop post-disaster du bâtiment
        2. Sauvegarde le crop dans output_dir/{split}/{class_name}/

    La structure de dossiers est le format natif attendu par YOLO26-cls.

    Args:
        records    : liste de dicts de find_post_disaster_images()
        split_name : "train", "val" ou "test"
        output_dir : répertoire racine du dataset YOLO cls
        verbose    : affiche la progression si True

    Retourne :
        Counter des crops sauvegardés par classe.
    """
    # Créer les sous-dossiers par classe
    for cls_name in PrepConfig.CLASS_NAMES:
        (Path(output_dir) / split_name / cls_name).mkdir(
            parents=True, exist_ok=True
        )

    stats  = Counter()
    errors = 0

    for i, record in enumerate(records):
        if verbose and i % 100 == 0:
            print(f"  [CLS/{split_name}] {i+1}/{len(records)} — {record['event']}")

        try:
            post_img_path   = record["post_img"]
            post_label_path = record["post_label"]
            stem            = Path(post_img_path).stem

            # ── Charger l'image post-disaster
            post_img = np.array(Image.open(post_img_path).convert("RGB"))

            # ── Charger les annotations
            buildings = load_json_buildings(post_label_path)

            for j, b in enumerate(buildings):
                damage = b["damage"]
                label  = PrepConfig.DAMAGE_TO_LABEL.get(damage, None)

                if label is None:
                    continue    # "un-classified" → ignorer

                # ── Extraire le crop
                crop = polygon_to_crop(post_img, b["polygon"])
                if crop is None:
                    continue

                # ── Sauvegarder dans le dossier de la classe
                cls_name  = PrepConfig.CLASS_NAMES[label]
                crop_name = f"{stem}_bld{j:04d}.png"
                crop_path = Path(output_dir) / split_name / cls_name / crop_name

                Image.fromarray(crop).save(crop_path)
                stats[cls_name] += 1

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  [ERROR] {Path(record['post_img']).name} → {e}")

    if verbose:
        total = sum(stats.values())
        print(f"\n  [CLS/{split_name}] {total} crops  | {errors} erreurs")
        for cls_name in PrepConfig.CLASS_NAMES:
            c   = stats[cls_name]
            bar = "█" * int(25 * c / max(total, 1))
            print(f"    {cls_name:<18} : {c:>6} ({100*c/max(total,1):5.1f}%)  {bar}")

    return stats


def write_classification_yaml(output_dir: str = PrepConfig.CLS_DIR):
    """
    Écrit le fichier dataset.yaml pour YOLO26-cls.

    Utilisation :
        yolo train model=yolo26n-cls.pt data=yolo26_dataset/classification
    """
    yaml_content = f"""# YOLO26-cls — Classification de dommages xView2
# Généré automatiquement par yolo26_data_preparation.py

path:  {Path(output_dir).resolve()}
train: train
val:   val
test:  test

nc: 4
names:
  0: no-damage
  1: minor-damage
  2: major-damage
  3: destroyed
"""
    yaml_path = Path(output_dir) / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[INFO] dataset.yaml classification → {yaml_path}")


# ═══════════════════════════════════════════════════════════
# 9. PIPELINE COMPLET DE PRÉPARATION
# ═══════════════════════════════════════════════════════════

def prepare_all_datasets(
    xview2_root: str = PrepConfig.XVIEW2_ROOT,
    output_root: str = PrepConfig.OUTPUT_ROOT,
    verbose:     bool = True,
) -> Dict:
    """
    Pipeline complet de préparation des deux datasets YOLO26.

    Étapes :
        1. Scan des images post-disaster xView2
        2. Split par événement (sans data leakage)
        3. Construction du dataset YOLO26-seg  (images + labels .txt)
        4. Construction du dataset YOLO26-cls  (crops par dossier classe)
        5. Écriture des fichiers dataset.yaml

    Args:
        xview2_root : chemin racine du dataset xView2
        output_root : chemin racine de sortie
        verbose     : affiche la progression

    Retourne :
        dict avec les statistiques de chaque split et dataset
    """
    seg_dir = str(Path(output_root) / "segmentation")
    cls_dir = str(Path(output_root) / "classification")

    print("═" * 62)
    print("  Préparation datasets YOLO26 — xView2 POST-DISASTER")
    print("═" * 62)

    # ── 1. Scan
    print("\n[1/4] Scan des images post-disaster...")
    records = find_post_disaster_images(xview2_root)
    if not records:
        raise FileNotFoundError(
            f"Aucune image post-disaster trouvée dans '{xview2_root}'.\n"
            "Vérifiez : xview2/train/images/*_post_disaster.png"
        )

    # ── 2. Split par événement
    print("\n[2/4] Split train / val / test...")
    train_r, val_r, test_r = stratified_event_split(records)

    splits = {"train": train_r, "val": val_r, "test": test_r}
    stats  = {"segmentation": {}, "classification": {}}

    # ── 3. Dataset YOLO26-seg
    print("\n[3/4] Construction dataset YOLO26-seg...")
    for split_name, split_records in splits.items():
        n = build_segmentation_dataset(
            split_records, split_name, seg_dir, verbose=verbose
        )
        stats["segmentation"][split_name] = n
    write_segmentation_yaml(seg_dir)

    # ── 4. Dataset YOLO26-cls
    print("\n[4/4] Construction dataset YOLO26-cls...")
    for split_name, split_records in splits.items():
        c = build_classification_dataset(
            split_records, split_name, cls_dir, verbose=verbose
        )
        stats["classification"][split_name] = dict(c)
    write_classification_yaml(cls_dir)

    # ── Résumé
    print(f"\n{'═' * 62}")
    print("  Datasets YOLO26 prêts !")
    print(f"\n  YOLO26-seg  → {seg_dir}")
    for s, n in stats["segmentation"].items():
        print(f"    {s:<8} : {n:>6} bâtiments annotés")
    print(f"\n  YOLO26-cls  → {cls_dir}")
    for s, cls_counts in stats["classification"].items():
        total = sum(cls_counts.values())
        print(f"    {s:<8} : {total:>6} crops")
    print(f"\n  Commandes d'entraînement :")
    print(f"    yolo train model=yolo26n-seg.pt "
          f"data={seg_dir}/dataset.yaml epochs=50 imgsz=640")
    print(f"    yolo train model=yolo26n-cls.pt "
          f"data={cls_dir} epochs=50 imgsz=128")
    print(f"{'═' * 62}")

    return stats


# ═══════════════════════════════════════════════════════════
# 10. POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    stats = prepare_all_datasets()

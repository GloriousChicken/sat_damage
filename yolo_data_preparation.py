"""
yolo26_data_preparation.py  (v2 — détection uniquement)
=========================================================
Prépare les données xView2 (images POST-disaster uniquement)
pour entraîner UN SEUL modèle YOLO26n de détection à 4 classes.

Différence clé vs version précédente (seg + cls) :
────────────────────────────────────────────────────────────────
  AVANT  : 2 modèles
    Modèle 1 YOLO26-seg  → 1 classe  "building"   (polygones)
    Modèle 2 YOLO26-cls  → 4 classes de dommages  (crops)

  APRÈS  : 1 modèle
    YOLO26n-det → 4 classes de dommages sur l'image complète
    Chaque bâtiment = 1 bbox + 1 label de dommage directement

Avantages :
    ✓ Un seul modèle à entraîner et à déployer
    ✓ Inférence plus rapide (une seule passe)
    ✓ Pas d'étape de crop intermédiaire
    ✓ Le modèle apprend localisation + classification conjointement

Inconvénient :
    ✗ Bboxes moins précises que des masques de segmentation
      (mais suffisant pour évaluer les dommages au niveau bâtiment)

Format YOLO détection (un fichier .txt par image) :
──────────────────────────────────────────────────────────────
  class_id  cx  cy  w  h   (toutes les valeurs normalisées [0, 1])

  class_id : 0 = no-damage
             1 = minor-damage
             2 = major-damage
             3 = destroyed
  cx, cy   : centre du bounding box (normalisé par W et H image)
  w, h     : largeur et hauteur du bbox (normalisées)

  Exemple (3 bâtiments dans une image) :
    0  0.412 0.318 0.065 0.072
    2  0.587 0.441 0.048 0.061
    3  0.721 0.290 0.055 0.068

Structure de sortie sur disque :
──────────────────────────────────────────────────────────────
yolo26_dataset/
└── detection/
    ├── images/
    │   ├── train/  hurricane-florence_00000001_post_disaster.png
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/  hurricane-florence_00000001_post_disaster.txt
    │   ├── val/
    │   └── test/
    └── dataset.yaml

Commande d'entraînement :
    yolo train model=yolo26n.pt \\
               data=yolo26_dataset/detection/dataset.yaml \\
               epochs=100 imgsz=640 batch=16

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

    # ── Sortie — un seul dossier detection
    OUTPUT_ROOT    = "yolo_dataset"
    DET_DIR        = "yolo_dataset/detection"

    # ── Split par événement
    TRAIN_RATIO    = 0.70
    VAL_RATIO      = 0.15      # Test = 1 - 0.70 - 0.15 = 0.15
    RANDOM_SEED    = 42

    # ── 4 classes de dommages (ordre = class_id YOLO)
    CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    DAMAGE_TO_LABEL = {
        "no-damage":     0,
        "minor-damage":  1,
        "major-damage":  2,
        "destroyed":     3,
        "un-classified": None,   # ignoré dans les labels
    }

    # ── Filtrage des bboxes
    BBOX_MIN_SIZE  = 10    # pixels — bboxes plus petites = bruit d'annotation
    BBOX_PADDING   = 2     # pixels de marge autour du bbox extrait du polygone


# ═══════════════════════════════════════════════════════════
# 2. PARSING JSON xView2
# ═══════════════════════════════════════════════════════════

def load_json_buildings(json_path: str) -> List[Dict]:
    """
    Parse un fichier JSON post-disaster xView2.
    Supporte les formats WKT (principal) et GeoJSON (fallback).

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

        wkt_str = feature.get("wkt", "")
        if wkt_str:
            try:
                geom = shapely_wkt.loads(wkt_str)
            except Exception:
                pass

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
# 3. CONVERSION POLYGONE → FORMAT YOLO DÉTECTION
# ═══════════════════════════════════════════════════════════

def polygon_to_yolo_det(
    polygon,
    image_w:  int,
    image_h:  int,
    class_id: int,
    padding:  int = PrepConfig.BBOX_PADDING,
) -> Optional[str]:
    """
    Convertit un polygone Shapely en ligne YOLO détection.

    Étapes de conversion :
        1. Extraire le bounding box du polygone  (bounds)
        2. Ajouter un padding en pixels
        3. Clipper aux dimensions de l'image
        4. Convertir en format YOLO (cx, cy, w, h normalisés)

    Format YOLO détection :
        class_id  cx  cy  w  h
        Toutes les valeurs normalisées par la taille de l'image ∈ [0, 1]

        cx = (x_min + x_max) / 2 / image_w
        cy = (y_min + y_max) / 2 / image_h
        w  = (x_max - x_min) / image_w
        h  = (y_max - y_min) / image_h

    Exemple :
        2  0.587 0.441 0.048 0.061
        → classe 2 (major-damage), centre (58.7%, 44.1%), taille (4.8% × 6.1%)

    Args:
        polygon  : shapely Polygon du bâtiment
        image_w  : largeur image en pixels
        image_h  : hauteur image en pixels
        class_id : 0=no-damage, 1=minor, 2=major, 3=destroyed
        padding  : marge en pixels autour du bbox

    Retourne :
        Chaîne "class_id cx cy w h" prête à écrire dans le .txt,
        ou None si le bbox est invalide / trop petit.
    """
    try:
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda p: p.area)

        minx, miny, maxx, maxy = polygon.bounds

        # Appliquer padding et clipper
        x0 = max(0,        minx - padding)
        y0 = max(0,        miny - padding)
        x1 = min(image_w,  maxx + padding)
        y1 = min(image_h,  maxy + padding)

        bbox_w = x1 - x0
        bbox_h = y1 - y0

        # Rejeter les bboxes trop petites
        if bbox_w < PrepConfig.BBOX_MIN_SIZE or bbox_h < PrepConfig.BBOX_MIN_SIZE:
            return None

        # Convertir en format YOLO normalisé
        cx = ((x0 + x1) / 2) / image_w
        cy = ((y0 + y1) / 2) / image_h
        w  = bbox_w / image_w
        h  = bbox_h / image_h

        # Clamp strict dans [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    except Exception:
        return None


def image_to_yolo_det_labels(
    json_path: str,
    image_w:   int,
    image_h:   int,
) -> Tuple[List[str], Counter]:
    """
    Génère toutes les lignes YOLO détection pour une image post-disaster.

    Différence vs version segmentation :
        SEG : class_id = 0 (building) pour tous les bâtiments
              + polygone complet en coordonnées normalisées
        DET : class_id = 0,1,2,3 selon le niveau de dommage
              + bbox (cx, cy, w, h) uniquement

    Les bâtiments "un-classified" sont ignorés (pas de class_id valide).

    Args:
        json_path : chemin JSON post-disaster
        image_w   : largeur image
        image_h   : hauteur image

    Retourne :
        lines  : liste de chaînes YOLO det (une par bâtiment valide)
        counts : Counter des labels générés (pour les stats)
    """
    buildings = load_json_buildings(json_path)
    lines     = []
    counts    = Counter()

    for b in buildings:
        label = PrepConfig.DAMAGE_TO_LABEL.get(b["damage"], None)
        if label is None:
            continue   # "un-classified" → ignoré

        line = polygon_to_yolo_det(b["polygon"], image_w, image_h, label)
        if line is not None:
            lines.append(line)
            counts[PrepConfig.CLASS_NAMES[label]] += 1

    return lines, counts


# ═══════════════════════════════════════════════════════════
# 4. SCAN DES IMAGES POST-DISASTER xView2
# ═══════════════════════════════════════════════════════════

def find_post_disaster_images(
    xview2_root:   str       = PrepConfig.XVIEW2_ROOT,
    source_splits: List[str] = PrepConfig.SOURCE_SPLITS,
) -> List[Dict[str, str]]:
    """
    Parcourt xView2 et retourne toutes les images POST-disaster
    avec leur JSON d'annotation.

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
# 5. SPLIT PAR ÉVÉNEMENT (sans data leakage)
# ═══════════════════════════════════════════════════════════

def stratified_event_split(
    records:     List[Dict],
    train_ratio: float = PrepConfig.TRAIN_RATIO,
    val_ratio:   float = PrepConfig.VAL_RATIO,
    seed:        int   = PrepConfig.RANDOM_SEED,
) -> Tuple[List, List, List]:
    """
    Split au niveau des événements.

    Toutes les images d'un même événement (même catastrophe,
    même zone géographique) restent dans le même split.
    Évite le data leakage entre train et val/test.
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
# 6. CONSTRUCTION DU DATASET YOLO26-DET
# ═══════════════════════════════════════════════════════════

def build_detection_dataset(
    records:    List[Dict],
    split_name: str,
    output_dir: str  = PrepConfig.DET_DIR,
    verbose:    bool = True,
) -> Dict:
    """
    Génère le dataset YOLO26n détection pour un split donné.

    Pour chaque image post-disaster :
        1. Copie l'image dans output_dir/images/{split}/
        2. Génère le fichier .txt de labels YOLO det
           (une ligne par bâtiment : class_id cx cy w h)

    Différence fondamentale vs l'ancienne version :
        Ancienne version seg  : class_id=0 pour tous (on détecte "building")
                                + polygone complet en coordonnées normalisées
        Cette version det     : class_id=0,1,2,3 selon le dommage
                                + bbox simplifié (cx, cy, w, h)
        → Le modèle apprend directement à distinguer les 4 niveaux de dommage
          par leur aspect visuel sans étape de classification séparée.

    Args:
        records    : liste de dicts de find_post_disaster_images()
        split_name : "train", "val" ou "test"
        output_dir : répertoire racine du dataset YOLO det
        verbose    : affiche la progression si True

    Retourne :
        {"total_buildings": int, "per_class": Counter, "errors": int}
    """
    img_out_dir   = Path(output_dir) / "images" / split_name
    label_out_dir = Path(output_dir) / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    total_counts = Counter()
    errors       = 0
    images_saved = 0

    for i, record in enumerate(records):
        if verbose and i % 100 == 0:
            print(f"  [DET/{split_name}] {i+1}/{len(records)} — {record['event']}")

        try:
            post_img_path   = record["post_img"]
            post_label_path = record["post_label"]
            stem            = Path(post_img_path).stem

            # Dimensions de l'image (sans charger le pixel data)
            with Image.open(post_img_path) as img:
                image_w, image_h = img.size

            # Générer les lignes YOLO det (4 classes)
            lines, counts = image_to_yolo_det_labels(
                post_label_path, image_w, image_h
            )

            # Ne copier l'image que si elle a au moins 1 bâtiment annoté
            if not lines:
                continue

            # Copier l'image
            shutil.copy2(post_img_path, img_out_dir / f"{stem}.png")

            # Écrire le fichier label
            (label_out_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n"
            )

            total_counts += counts
            images_saved += 1

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  [ERROR] {Path(record['post_img']).name} → {e}")

    total_buildings = sum(total_counts.values())

    if verbose:
        print(f"\n  [DET/{split_name}] {images_saved} images | "
              f"{total_buildings} bâtiments | {errors} erreurs")
        for cls_name in PrepConfig.CLASS_NAMES:
            c   = total_counts[cls_name]
            bar = "█" * int(25 * c / max(total_buildings, 1))
            print(f"    {cls_name:<18} : {c:>6} "
                  f"({100*c/max(total_buildings,1):5.1f}%)  {bar}")

    return {
        "images_saved":    images_saved,
        "total_buildings": total_buildings,
        "per_class":       dict(total_counts),
        "errors":          errors,
    }


# ═══════════════════════════════════════════════════════════
# 7. FICHIER dataset.yaml
# ═══════════════════════════════════════════════════════════

def write_detection_yaml(output_dir: str = PrepConfig.DET_DIR):
    """
    Écrit le fichier dataset.yaml pour YOLO26n détection.

    Ce fichier est la seule configuration nécessaire pour lancer
    l'entraînement YOLO26n avec 4 classes de dommages.

    Utilisation :
        yolo train model=yolo26n.pt \\
                   data=yolo26_dataset/detection/dataset.yaml \\
                   epochs=100 imgsz=640 batch=16
    """
    yaml_content = f"""# YOLO26n — Détection de bâtiments avec classification dommages
# Dataset : xView2 (post-disaster uniquement)
# Généré automatiquement par yolo26_data_preparation.py
#
# Chaque bâtiment annoté = 1 bbox + 1 label de dommage
# Format label : class_id  cx  cy  w  h  (valeurs normalisées [0,1])

path:  {Path(output_dir).resolve()}
train: images/train
val:   images/val
test:  images/test

# Nombre de classes
nc: 4

# Noms des classes (ordre = class_id)
names:
  0: no-damage
  1: minor-damage
  2: major-damage
  3: destroyed
"""
    yaml_path = Path(output_dir) / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[INFO] dataset.yaml → {yaml_path}")
    return str(yaml_path)


# ═══════════════════════════════════════════════════════════
# 8. PIPELINE COMPLET
# ═══════════════════════════════════════════════════════════

def prepare_detection_dataset(
    xview2_root: str  = PrepConfig.XVIEW2_ROOT,
    output_root: str  = PrepConfig.OUTPUT_ROOT,
    verbose:     bool = True,
) -> Dict:
    """
    Pipeline complet de préparation du dataset YOLO26n détection.

    Étapes :
        1. Scan des images post-disaster xView2
        2. Split par événement (sans data leakage)
        3. Construction du dataset (images + labels .txt) pour train/val/test
        4. Écriture du fichier dataset.yaml

    Args:
        xview2_root : chemin racine du dataset xView2
        output_root : chemin racine de sortie
        verbose     : affiche la progression

    Retourne :
        dict avec les statistiques et le chemin du dataset.yaml
    """
    det_dir = str(Path(output_root) / "detection")

    print("═" * 62)
    print("  Préparation YOLO26n Détection — xView2 POST-DISASTER")
    print("  4 classes : no-damage / minor / major / destroyed")
    print("═" * 62)

    # ── 1. Scan
    print("\n[1/3] Scan des images post-disaster...")
    records = find_post_disaster_images(xview2_root)
    if not records:
        raise FileNotFoundError(
            f"Aucune image trouvée dans '{xview2_root}'.\n"
            "Structure attendue : xview2/train/images/*_post_disaster.png"
        )

    # ── 2. Split par événement
    print("\n[2/3] Split train / val / test...")
    train_r, val_r, test_r = stratified_event_split(records)

    # ── 3. Construction du dataset
    print("\n[3/3] Construction du dataset YOLO26n détection...")
    splits = {"train": train_r, "val": val_r, "test": test_r}
    stats  = {}
    for split_name, split_records in splits.items():
        stats[split_name] = build_detection_dataset(
            split_records, split_name, det_dir, verbose=verbose
        )

    yaml_path = write_detection_yaml(det_dir)

    # ── Résumé final
    print(f"\n{'═' * 62}")
    print("  Dataset YOLO26n prêt !")
    print(f"  Dossier : {det_dir}")
    print(f"\n  {'Split':<8}  {'Images':>8}  {'Bâtiments':>10}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}")
    for split_name, s in stats.items():
        print(f"  {split_name:<8}  {s['images_saved']:>8}  "
              f"{s['total_buildings']:>10}")

    # Distribution globale des classes
    all_counts = Counter()
    for s in stats.values():
        all_counts += Counter(s["per_class"])
    total = sum(all_counts.values())

    print(f"\n  Distribution des classes (toutes splits) :")
    for cls_name in PrepConfig.CLASS_NAMES:
        c   = all_counts[cls_name]
        bar = "█" * int(25 * c / max(total, 1))
        print(f"    {cls_name:<18} : {c:>7} "
              f"({100*c/max(total,1):5.1f}%)  {bar}")

    print(f"\n  Commande d'entraînement :")
    print(f"    yolo train model=yolo26n.pt \\")
    print(f"               data={yaml_path} \\")
    print(f"               epochs=100 imgsz=640 batch=16")
    print(f"{'═' * 62}")

    stats["yaml_path"] = yaml_path
    return stats


# ═══════════════════════════════════════════════════════════
# 9. POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    prepare_detection_dataset()

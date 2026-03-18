"""
yolo26_pipeline.py
==================
Pipeline d'inférence en deux étapes — images POST-disaster uniquement.

  Étape 1 — YOLO26-seg  :  segmente les bâtiments sur l'image post-disaster
  Étape 2 — YOLO26-cls  :  classifie le niveau de dommage de chaque bâtiment

Flux complet :
──────────────────────────────────────────────────────────────────────
  Image post-disaster
        ↓
  [YOLO26-seg]
  detect_buildings()
        ↓
  Masques + bboxes des bâtiments
        ↓
  Extraction des crops post-disaster
        ↓
  [YOLO26-cls]  (en batch sur tous les crops)
  classify_damage()
        ↓
  Liste de résultats par bâtiment :
    { "bbox": (x0,y0,x1,y1), "mask": ...,
      "label": "major-damage", "confidence": 0.87,
      "probabilities": {...} }
        ↓
  [Visualisation optionnelle]
  draw_results()  →  image annotée sauvegardée

Utilisation dans FastAPI :
──────────────────────────
    from yolo26_pipeline import YOLO26Pipeline

    pipeline = YOLO26Pipeline(
        seg_weights = "runs/segment/train/weights/best.pt",
        cls_weights = "runs/classify/train/weights/best.pt",
    )
    results = pipeline.predict("post_disaster.png")

Pré-requis :
    pip install ultralytics pillow numpy
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional, Union


# ═══════════════════════════════════════════════════════════
# 1. CONFIGURATION DU PIPELINE
# ═══════════════════════════════════════════════════════════

class PipelineConfig:
    # ── Modèles fine-tunés sur xView2
    # Remplacer par les chemins réels après l'entraînement
    SEG_WEIGHTS  = "yolo26n-seg.pt" # "runs/segment/train/weights/best.pt"
    CLS_WEIGHTS  = "yolo26n-cls.pt" # "runs/classify/train/weights/best.pt"

    # ── Seuils YOLO26-seg
    SEG_CONF     = 0.25     # Confiance minimale pour valider une détection
    SEG_IOU      = 0.45     # NMS IoU
    SEG_IMGSZ    = 640      # Taille d'inférence (multiple de 32)

    # ── Crop pour YOLO26-cls
    CLS_IMGSZ    = 128      # Taille des crops envoyés au classifieur
    CLS_PADDING  = 10       # Marge en pixels autour du bbox

    # ── Classes de dommages (doivent correspondre à l'entraînement cls)
    CLASS_NAMES  = ["no-damage", "minor-damage", "major-damage", "destroyed"]

    # ── Couleurs de visualisation par classe
    CLASS_COLORS = {
        "no-damage":     (  0, 200,   0),   # vert
        "minor-damage":  (255, 200,   0),   # jaune
        "major-damage":  (255, 100,   0),   # orange
        "destroyed":     (220,   0,   0),   # rouge
    }

    # ── Device
    DEVICE       = "cpu"     # "cpu" ou "cuda:0"

    # ── Batch size pour la classification (traitement en lot des crops)
    CLS_BATCH    = 32


# ═══════════════════════════════════════════════════════════
# 2. CHARGEMENT DES MODÈLES
# ═══════════════════════════════════════════════════════════

class YOLO26Pipeline:
    """
    Pipeline YOLO26 à deux étapes pour l'évaluation des dommages de bâtiments.

    Attributs publics :
        seg_model : modèle YOLO26-seg chargé
        cls_model : modèle YOLO26-cls chargé
        config    : PipelineConfig

    Méthode principale :
        predict(image) → List[Dict]
    """

    def __init__(
        self,
        seg_weights: str = PipelineConfig.SEG_WEIGHTS,
        cls_weights: str = PipelineConfig.CLS_WEIGHTS,
        device:      str = PipelineConfig.DEVICE,
        verbose:     bool = False,
    ):
        """
        Charge les deux modèles YOLO26.

        Args:
            seg_weights : chemin vers best.pt du modèle YOLO26-seg
            cls_weights : chemin vers best.pt du modèle YOLO26-cls
            device      : "cpu", "cuda:0", "cuda:1", ...
            verbose     : affiche les résumés des modèles si True
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics non installé.\n"
                "Installer avec : pip install ultralytics"
            )

        print(f"[Pipeline] Chargement YOLO26-seg  : {seg_weights}")
        self.seg_model = YOLO(seg_weights)
        self.seg_model.to(device)

        print(f"[Pipeline] Chargement YOLO26-cls  : {cls_weights}")
        self.cls_model = YOLO(cls_weights)
        self.cls_model.to(device)

        self.device  = device
        self.verbose = verbose

        if verbose:
            self.seg_model.info()
            self.cls_model.info()

        print(f"[Pipeline] Deux modèles chargés sur {device} ✓")


    # ──────────────────────────────────────────────────────
    # ÉTAPE 1 : Segmentation des bâtiments
    # ──────────────────────────────────────────────────────

    def segment_buildings(
        self,
        image_path: str,
    ) -> List[Dict]:
        """
        Détecte et segmente les bâtiments sur une image post-disaster.

        Args:
            image_path : chemin vers l'image post-disaster

        Retourne :
            Liste de dicts par bâtiment détecté :
            [{
                "bbox":       (x0, y0, x1, y1),   ← pixels absolus
                "centroid":   (cx, cy),
                "confidence": float,               ← score YOLO26-seg
                "mask":       np.ndarray | None,   ← masque booléen (H×W)
                "crop":       np.ndarray,          ← crop post-disaster (H,W,3)
            }, ...]
        """
        results = self.seg_model.predict(
            source  = image_path,
            conf    = PipelineConfig.SEG_CONF,
            iou     = PipelineConfig.SEG_IOU,
            imgsz   = PipelineConfig.SEG_IMGSZ,
            verbose = False,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        result   = results[0]
        boxes    = result.boxes
        image    = np.array(Image.open(image_path).convert("RGB"))
        h, w     = image.shape[:2]

        for i in range(len(boxes)):
            x0, y0, x1, y1 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf_score      = float(boxes.conf[i].item())

            # Clipper le bbox aux limites de l'image
            x0 = max(0, x0 - PipelineConfig.CLS_PADDING)
            y0 = max(0, y0 - PipelineConfig.CLS_PADDING)
            x1 = min(w, x1 + PipelineConfig.CLS_PADDING)
            y1 = min(h, y1 + PipelineConfig.CLS_PADDING)

            if (x1 - x0) < 10 or (y1 - y0) < 10:
                continue

            # Extraire le crop
            crop = image[y0:y1, x0:x1]

            # Masque de segmentation (optionnel)
            mask = None
            if result.masks is not None:
                mask = result.masks.data[i].cpu().numpy().astype(bool)

            detections.append({
                "bbox":       (x0, y0, x1, y1),
                "centroid":   ((x0 + x1) // 2, (y0 + y1) // 2),
                "confidence": conf_score,
                "mask":       mask,
                "crop":       crop,
            })

        return detections


    # ──────────────────────────────────────────────────────
    # ÉTAPE 2 : Classification des dommages
    # ──────────────────────────────────────────────────────

    def classify_damage(
        self,
        detections: List[Dict],
    ) -> List[Dict]:
        """
        Classifie le niveau de dommage de chaque bâtiment détecté.

        Traite les crops en batch pour maximiser l'utilisation du GPU.

        Args:
            detections : sortie de segment_buildings()

        Retourne :
            La même liste enrichie avec les champs de classification :
            [{
                ...,                              ← champs de seg inchangés
                "label":         "major-damage",
                "label_id":      2,
                "cls_confidence": 0.87,
                "probabilities": {
                    "no-damage":    0.03,
                    "minor-damage": 0.08,
                    "major-damage": 0.87,
                    "destroyed":    0.02,
                }
            }, ...]
        """
        if not detections:
            return detections

        import tempfile, os

        # ── Sauvegarder les crops en fichiers temporaires pour YOLO-cls
        # YOLO26-cls attend des chemins ou des arrays PIL/numpy
        tmp_dir   = tempfile.mkdtemp(prefix="yolo_cls_")
        tmp_paths = []

        for idx, det in enumerate(detections):
            crop_pil  = Image.fromarray(det["crop"]).resize(
                (PipelineConfig.CLS_IMGSZ, PipelineConfig.CLS_IMGSZ),
                Image.BILINEAR
            )
            crop_path = os.path.join(tmp_dir, f"crop_{idx:04d}.png")
            crop_pil.save(crop_path)
            tmp_paths.append(crop_path)

        # ── Inférence YOLO26-cls en batch
        cls_results = self.cls_model.predict(
            source  = tmp_paths,
            imgsz   = PipelineConfig.CLS_IMGSZ,
            verbose = False,
        )

        # ── Enrichir les détections avec les résultats cls
        class_names = (self.cls_model.names
                       if hasattr(self.cls_model, "names")
                       else {i: n for i, n in enumerate(PipelineConfig.CLASS_NAMES)})

        for idx, (det, cls_result) in enumerate(zip(detections, cls_results)):
            # cls_result.probs : vecteur de probabilités par classe
            probs  = cls_result.probs.data.cpu().numpy()   # shape (4,)
            lid    = int(np.argmax(probs))
            label  = class_names.get(lid, PipelineConfig.CLASS_NAMES[lid])

            det["label"]          = label
            det["label_id"]       = lid
            det["cls_confidence"] = round(float(probs[lid]), 4)
            det["probabilities"]  = {
                class_names.get(i, f"class_{i}"): round(float(p), 4)
                for i, p in enumerate(probs)
            }

        # ── Nettoyer les fichiers temporaires
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return detections


    # ──────────────────────────────────────────────────────
    # PIPELINE COMPLET
    # ──────────────────────────────────────────────────────

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        tmp_path:    str = "/tmp/yolo_post_infer.png",
    ) -> List[Dict]:
        """
        Pipeline complet : segmentation → classification.

        Args:
            image_input : chemin vers l'image post-disaster,
                          ou np.array (H, W, 3) uint8

        Retourne :
            Liste de résultats par bâtiment détecté :
            [{
                "bbox":          (x0, y0, x1, y1),
                "centroid":      (cx, cy),
                "confidence":    float,      ← confiance segmentation
                "mask":          np.ndarray | None,
                "crop":          np.ndarray,
                "label":         "major-damage",
                "label_id":      2,
                "cls_confidence": 0.87,
                "probabilities": {"no-damage": 0.03, ...}
            }, ...]

            Liste vide si aucun bâtiment détecté.
        """
        # Gérer l'input numpy (inférence FastAPI)
        if isinstance(image_input, np.ndarray):
            Image.fromarray(image_input).save(tmp_path)
            image_path = tmp_path
        else:
            image_path = str(image_input)

        # ── Étape 1 : segmentation
        detections = self.segment_buildings(image_path)

        if not detections:
            if self.verbose:
                print(f"[Pipeline] Aucun bâtiment détecté dans {image_path}")
            return []

        if self.verbose:
            print(f"[Pipeline] {len(detections)} bâtiments détectés")

        # ── Étape 2 : classification
        detections = self.classify_damage(detections)

        if self.verbose:
            counts = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            print(f"[Pipeline] Résultats : {counts}")

        return detections


    # ──────────────────────────────────────────────────────
    # VISUALISATION
    # ──────────────────────────────────────────────────────

    def draw_results(
        self,
        image_input: Union[str, np.ndarray],
        detections:  List[Dict],
        save_path:   str = "pipeline_output.png",
        show_mask:   bool = False,
    ) -> np.ndarray:
        """
        Dessine les bboxes et labels de dommage sur l'image post-disaster.

        Code couleur :
            Vert    → no-damage
            Jaune   → minor-damage
            Orange  → major-damage
            Rouge   → destroyed

        Args:
            image_input : chemin ou np.array
            detections  : sortie de predict()
            save_path   : chemin de sauvegarde de l'image annotée
            show_mask   : superpose les masques de segmentation si True

        Retourne :
            np.array (H, W, 3) — image annotée
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = Image.fromarray(image_input)

        draw = ImageDraw.Draw(img, "RGBA")
        h, w = img.size[1], img.size[0]

        # Essayer de charger une police, fallback sur défaut
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            label  = det.get("label", "unknown")
            conf_s = det.get("confidence",    0.0)     # seg confidence
            conf_c = det.get("cls_confidence", 0.0)    # cls confidence
            x0, y0, x1, y1 = det["bbox"]
            color  = PipelineConfig.CLASS_COLORS.get(label, (128, 128, 128))

            # ── Masque de segmentation (fond semi-transparent)
            if show_mask and det.get("mask") is not None:
                mask = det["mask"]
                if mask.shape != (h, w):
                    mask_img = Image.fromarray(
                        (mask * 255).astype(np.uint8)
                    ).resize((w, h), Image.NEAREST)
                    mask = np.array(mask_img) > 127
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                r, g, b = color
                ys, xs  = np.where(mask)
                for px, py in zip(xs.tolist(), ys.tolist()):
                    overlay_draw.point((px, py), fill=(r, g, b, 80))
                img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                draw = ImageDraw.Draw(img)

            # ── Bounding box
            r, g, b = color
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=(r, g, b),
                width=2
            )

            # ── Étiquette
            text = f"{label} {conf_c:.2f}"
            bbox_text = draw.textbbox((x0, y0 - 14), text, font=font)
            draw.rectangle(bbox_text, fill=(r, g, b, 200))
            draw.text((x0, y0 - 14), text, fill=(255, 255, 255), font=font)

        img.save(save_path)
        print(f"[Pipeline] Résultat sauvegardé → {save_path}")
        return np.array(img)


    # ──────────────────────────────────────────────────────
    # STATISTIQUES
    # ──────────────────────────────────────────────────────

    def summarize(self, detections: List[Dict]) -> Dict:
        """
        Résume les résultats du pipeline pour une image.

        Retourne :
        {
            "total_buildings": 12,
            "counts": {
                "no-damage":    5,
                "minor-damage": 3,
                "major-damage": 3,
                "destroyed":    1,
            },
            "damage_score": 0.42,   ← 0 = pas de dommage, 1 = tout détruit
                                       (moyenne pondérée : minor=0.33, major=0.67, destroyed=1.0)
        }
        """
        counts = {name: 0 for name in PipelineConfig.CLASS_NAMES}
        weights = {"no-damage": 0.0, "minor-damage": 0.33,
                   "major-damage": 0.67, "destroyed": 1.0}

        for det in detections:
            label = det.get("label", "no-damage")
            if label in counts:
                counts[label] += 1

        total         = sum(counts.values())
        damage_score  = (
            sum(counts[l] * weights[l] for l in PipelineConfig.CLASS_NAMES)
            / max(total, 1)
        )

        return {
            "total_buildings": total,
            "counts":          counts,
            "damage_score":    round(damage_score, 4),
        }


# ═══════════════════════════════════════════════════════════
# 3. POINT D'ENTRÉE — EXEMPLE COMPLET
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── Initialiser le pipeline
    pipeline = YOLO26Pipeline()

    # ── Inférence sur une image post-disaster
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_post.png"

    detections = pipeline.predict(image_path)

    # ── Résumé
    summary = pipeline.summarize(detections)
    print(f"\n── Résumé ──")
    print(f"  Bâtiments détectés : {summary['total_buildings']}")
    for cls_name in PipelineConfig.CLASS_NAMES:
        c   = summary["counts"][cls_name]
        bar = "█" * c
        print(f"  {cls_name:<18} : {c:>3}  {bar}")
    print(f"  Score de dommage  : {summary['damage_score']:.3f}  "
          f"(0=intact → 1=tout détruit)")

    # ── Visualisation
    pipeline.draw_results(
        image_input = image_path,
        detections  = detections,
        save_path   = "pipeline_output.png",
        show_mask   = True,
    )

    # ── Détails par bâtiment
    print(f"\n── Détails ({len(detections)} bâtiments) ──")
    for i, det in enumerate(detections):
        print(
            f"  [{i+1:>3}] bbox={det['bbox']}  "
            f"label={det['label']:<18}  "
            f"conf_seg={det['confidence']:.2f}  "
            f"conf_cls={det['cls_confidence']:.2f}"
        )

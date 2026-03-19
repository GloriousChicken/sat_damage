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
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import shutil
import tempfile, os
import json
from collections import defaultdict
from shapely import wkt as shapely_wkt
from shapely.geometry import shape as shapely_shape
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import List, Dict, Optional, Union, Tuple
from yolo_data_preparation import find_post_disaster_images, stratified_event_split

# ═══════════════════════════════════════════════════════════
# 1. CONFIGURATION DU PIPELINE
# ═══════════════════════════════════════════════════════════

class PipelineConfig:
    # ── Modèles fine-tunés sur xView2
    # Remplacer par les chemins réels après l'entraînement
    SEG_WEIGHTS  = "../.pyenv/runs/segment/train/weights/best.pt" # "runs/segment/train/weights/best.pt"
    CLS_WEIGHTS  = "../.pyenv/runs/classify/train/weights/best.pt" # "runs/classify/train/weights/best.pt"

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
    DEVICE       = "cuda:0"     # "cpu" ou "cuda:0"

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
    # ÉVALUATION SUR LE TEST SET
    # ──────────────────────────────────────────────────────

    def evaluate(
        self,
        test_records:   List[Dict],
        output_dir:     str  = "evaluation_results",
        seg_data_yaml:  str  = "yolo26_dataset/segmentation/dataset.yaml",
        save_plots:     bool = True,
        verbose:        bool = True,
    ) -> Dict:
        """
        Évaluation complète du pipeline sur le test set.

        Deux évaluations distinctes sont réalisées :

        ┌──────────────────────────────────────────────────────────────┐
        │  ÉVALUATION 1 — YOLO26-seg  (métriques natives Ultralytics) │
        │    Calcul officiel YOLO via model.val()                      │
        │    → mAP50/95 bbox, mAP50/95 mask, Precision, Recall        │
        └──────────────────────────────────────────────────────────────┘
        ┌──────────────────────────────────────────────────────────────┐
        │  ÉVALUATION 2 — Pipeline complet  (métriques custom)        │
        │    Pour chaque bâtiment du test set :                        │
        │      1. YOLO26-seg détecte le bâtiment                      │
        │      2. YOLO26-cls prédit la classe de dommage               │
        │      3. On compare la prédiction au label JSON (ground truth)│
        │    → Rapport de classification, matrice de confusion, F1     │
        └──────────────────────────────────────────────────────────────┘

        Args:
            test_records   : liste de dicts issue de find_post_disaster_images()
                             [{"post_img": ..., "post_label": ..., "event": ...}]
            output_dir     : dossier de sauvegarde des résultats
            seg_data_yaml  : chemin vers dataset.yaml YOLO-seg (pour model.val())
            save_plots     : sauvegarde confusion matrix et courbes si True
            verbose        : affiche la progression si True

        Retourne :
            {
              "segmentation": {
                  "box_map50":    float,   ← mAP50 bounding boxes
                  "box_map50_95": float,   ← mAP50-95 bounding boxes
                  "mask_map50":   float,   ← mAP50 masques
                  "mask_map50_95":float,   ← mAP50-95 masques
                  "precision":    float,
                  "recall":       float,
              },
              "classification": {
                  "accuracy":     float,
                  "f1_macro":     float,
                  "f1_weighted":  float,
                  "f1_per_class": {"no-damage": float, ...},
                  "report":       str,     ← texte complet sklearn
                  "confusion_matrix": np.ndarray  (4×4)
              },
              "pipeline": {
                  "total_images":     int,
                  "total_gt_buildings": int,   ← bâtiments dans les JSON
                  "total_detected":   int,     ← bâtiments détectés par YOLO
                  "total_matched":    int,     ← détections avec GT valide
                  "detection_recall": float,   ← matched / gt_buildings
              }
            }
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ════════════════════════════════════════
        # ÉVALUATION 1 — YOLO26-seg (métriques natives)
        # ════════════════════════════════════════
        seg_metrics = self._evaluate_segmentation(seg_data_yaml, verbose)

        # ════════════════════════════════════════
        # ÉVALUATION 2 — Pipeline complet (classification par bâtiment)
        # ════════════════════════════════════════
        cls_metrics, pipeline_stats = self._evaluate_classification(
            test_records, output_dir, save_plots, verbose
        )

        # ── Résumé console
        self._print_evaluation_summary(seg_metrics, cls_metrics, pipeline_stats)

        # ── Sauvegarde JSON
        summary = {
            "segmentation":  seg_metrics,
            "classification": {
                k: v for k, v in cls_metrics.items()
                if k != "confusion_matrix"   # np.ndarray non sérialisable
            },
            "pipeline": pipeline_stats,
        }
        json_path = Path(output_dir) / "evaluation_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Evaluate] Résumé sauvegardé → {json_path}")

        # Réinjecter la matrice pour le retour Python
        summary["classification"]["confusion_matrix"] = cls_metrics["confusion_matrix"]
        return summary


    # ──────────────────────────────────────────────────────
    # ÉVALUATION 1 : métriques YOLO26-seg natives
    # ──────────────────────────────────────────────────────

    def _evaluate_segmentation(
        self,
        data_yaml: str,
        verbose:   bool,
    ) -> Dict:
        """
        Lance model.val() sur le split test du dataset YOLO-seg.

        Métriques retournées par Ultralytics :
        ─────────────────────────────────────────────────────────
        box_map50    : mAP@IoU=0.50  sur les bounding boxes
                       ↑ Principale métrique de localisation.
                       Un bâtiment est "trouvé" si IoU(pred,GT) ≥ 0.50.

        box_map50_95 : mAP moyen de IoU=0.50 à 0.95 (pas=0.05)
                       ↑ Métrique plus sévère, pénalise les bbox imprécis.

        mask_map50   : mAP@IoU=0.50 sur les masques de segmentation
                       ↑ Mesure la qualité du contour prédit.
                       Toujours ≤ box_map50 car la segmentation est plus fine.

        mask_map50_95: mAP masque de IoU=0.50 à 0.95
                       ↑ Métrique la plus sévère du pipeline.

        precision    : TP/(TP+FP) — peu de fausses détections
        recall       : TP/(TP+FN) — peu de bâtiments manqués
                       ↑ Pour xView2 : Recall > Precision est préférable
                         (mieux vaut sur-détecter que rater des bâtiments)
        """
        print("\n" + "─" * 55)
        print("  Évaluation YOLO26-seg (métriques natives)")
        print("─" * 55)

        if not Path(data_yaml).exists():
            print(f"  [WARN] dataset.yaml introuvable : {data_yaml}")
            print("  → Évaluation seg ignorée. Fournir seg_data_yaml=...")
            return {
                "box_map50": None, "box_map50_95": None,
                "mask_map50": None, "mask_map50_95": None,
                "precision": None, "recall": None,
                "note": f"dataset.yaml introuvable : {data_yaml}"
            }

        try:
            val_results = self.seg_model.val(
                data   = data_yaml,
                split  = "test",
                imgsz  = PipelineConfig.SEG_IMGSZ,
                verbose= verbose,
            )

            # Extraire les métriques depuis l'objet Ultralytics
            # L'API Ultralytics retourne un objet Metrics avec :
            #   .box.map50, .box.map, .seg.map50, .seg.map
            #   .box.p, .box.r  (precision, recall)
            metrics = {
                "box_map50":     round(float(val_results.box.map50),  4),
                "box_map50_95":  round(float(val_results.box.map),    4),
                "mask_map50":    round(float(val_results.seg.map50),  4),
                "mask_map50_95": round(float(val_results.seg.map),    4),
                "precision":     round(float(val_results.box.p.mean()), 4),
                "recall":        round(float(val_results.box.r.mean()), 4),
            }
            return metrics

        except Exception as e:
            print(f"  [ERROR] model.val() a échoué : {e}")
            return {"error": str(e)}


    # ──────────────────────────────────────────────────────
    # ÉVALUATION 2 : classification par bâtiment
    # ──────────────────────────────────────────────────────

    def _evaluate_classification(
        self,
        test_records: List[Dict],
        output_dir:   str,
        save_plots:   bool,
        verbose:      bool,
    ) -> Tuple[Dict, Dict]:
        """
        Évalue la classification en faisant tourner le pipeline complet
        sur toutes les images du test set.

        Ground truth : labels extraits des JSON post-disaster.
        Prédictions  : YOLO26-seg → crop → YOLO26-cls.

        Matching GT ↔ prédiction :
            Pour chaque bâtiment JSON (GT), on cherche la détection YOLO
            dont le centroïde est le plus proche. Si la distance est
            ≤ MAX_MATCH_DIST pixels, on considère que c'est le même bâtiment
            et on compare la prédiction cls au label GT.

        Retourne : (cls_metrics_dict, pipeline_stats_dict)
        """
        print("\n" + "─" * 55)
        print("  Évaluation pipeline complet (classification bâtiment)")
        print("─" * 55)

        y_true   = []
        y_pred   = []
        y_prob   = []    # probabilités softmax — pour courbes ROC futures

        total_gt_buildings = 0
        total_detected     = 0
        total_matched      = 0
        errors             = 0

        for i, record in enumerate(test_records):
            if verbose and i % 50 == 0:
                print(f"  [{i+1:>4}/{len(test_records)}] {record.get('event', '')}")

            try:
                post_img_path   = record["post_img"]
                post_label_path = record["post_label"]

                # ── Ground truth depuis JSON
                gt_buildings = self._load_gt_buildings(post_label_path)
                total_gt_buildings += len(gt_buildings)

                if not gt_buildings:
                    continue

                # ── Prédiction pipeline
                detections = self.predict(post_img_path)
                total_detected += len(detections)

                if not detections:
                    continue

                # ── Matching GT ↔ détections
                matched = self._match_gt_to_detections(gt_buildings, detections)
                total_matched += len(matched)

                for gt_label, det in matched:
                    y_true.append(gt_label)
                    y_pred.append(det["label_id"])
                    y_prob.append([
                        det["probabilities"].get(name, 0.0)
                        for name in PipelineConfig.CLASS_NAMES
                    ])

            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  [ERROR] {Path(record['post_img']).name} → {e}")

        # ── Vérification
        if len(y_true) == 0:
            print("  [WARN] Aucune détection matchée — vérifier les modèles et le dataset.")
            return (
                {"error": "Aucun sample évalué"},
                {"total_images": len(test_records), "errors": errors}
            )

        y_true  = np.array(y_true,  dtype=int)
        y_pred  = np.array(y_pred,  dtype=int)
        y_prob  = np.array(y_prob,  dtype=float)

        # ── Métriques sklearn
        cls_metrics = self._compute_cls_metrics(
            y_true, y_pred, y_prob, output_dir, save_plots
        )

        pipeline_stats = {
            "total_images":       len(test_records),
            "total_gt_buildings": total_gt_buildings,
            "total_detected":     total_detected,
            "total_matched":      total_matched,
            "detection_recall":   round(total_matched / max(total_gt_buildings, 1), 4),
            "errors":             errors,
        }

        return cls_metrics, pipeline_stats


    def _load_gt_buildings(self, json_path: str) -> List[Dict]:
        """
        Charge les bâtiments ground truth depuis un JSON post-disaster.
        Retourne uniquement ceux avec un label valide (pas "un-classified").
        """

        with open(json_path, "r") as f:
            data = json.load(f)

        gt_buildings = []
        for feat in data.get("features", {}).get("xy", []):
            props  = feat.get("properties", {})
            damage = props.get("subtype", "un-classified")
            label  = PipelineConfig.DAMAGE_TO_LABEL.get(damage, None)
            if label is None:
                continue

            geom = None
            wkt_str = feat.get("wkt", "")
            if wkt_str:
                try:
                    geom = shapely_wkt.loads(wkt_str)
                except Exception:
                    pass
            if geom is None:
                try:
                    geom = shapely_shape(feat.get("geometry", {}))
                except Exception:
                    continue
            if geom is None or geom.is_empty:
                continue

            minx, miny, maxx, maxy = geom.bounds
            gt_buildings.append({
                "label":    label,
                "centroid": ((minx + maxx) / 2, (miny + maxy) / 2),
            })

        return gt_buildings


    def _match_gt_to_detections(
        self,
        gt_buildings: List[Dict],
        detections:   List[Dict],
    ) -> List[Tuple[int, Dict]]:
        """
        Associe chaque bâtiment GT à la détection YOLO la plus proche.

        Stratégie : distance euclidienne entre centroïdes.
        Un match est valide si distance ≤ MAX_MATCH_DIST pixels.
        Un seul match par détection (greedy, premier arrivé).

        Retourne :
            [(gt_label_int, detection_dict), ...]
        """
        matched      = []
        used_det_idx = set()

        for gt in gt_buildings:
            gt_cx, gt_cy = gt["centroid"]
            best_dist    = PipelineConfig.MAX_MATCH_DIST
            best_idx     = -1

            for j, det in enumerate(detections):
                if j in used_det_idx:
                    continue
                det_cx, det_cy = det["centroid"]
                dist = ((gt_cx - det_cx) ** 2 + (gt_cy - det_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = j

            if best_idx >= 0:
                matched.append((gt["label"], detections[best_idx]))
                used_det_idx.add(best_idx)

        return matched


    def _compute_cls_metrics(
        self,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        y_prob:     np.ndarray,
        output_dir: str,
        save_plots: bool,
    ) -> Dict:
        """
        Calcule toutes les métriques de classification.

        ─────────────────────────────────────────────────────────────
        Rapport de classification (sklearn) :
            Pour chaque classe i :
              Precision  = TP_i / (TP_i + FP_i)
                           Parmi les bâtiments prédits "classe i",
                           combien le sont vraiment ?
              Recall     = TP_i / (TP_i + FN_i)
                           Parmi les vrais bâtiments "classe i",
                           combien ont été correctement identifiés ?
              F1-score   = 2 × (P × R) / (P + R)
                           Moyenne harmonique — utile sur données déséquilibrées.
              Support    = nombre de vrais samples de la classe i dans le test set.

        F1 macro     : moyenne simple des F1 par classe (toutes les classes
                       comptent autant → approprié pour xView2 car toutes
                       les classes de dommage sont critiques).

        F1 weighted  : moyenne des F1 pondérée par le support (nombre de
                       samples). Favorise les classes majoritaires
                       (no-damage dans xView2).

        Matrice de confusion 4×4 :
            Ligne   = classe réelle  (ground truth)
            Colonne = classe prédite
            Diagonale = prédictions correctes
            Hors-diagonale = confusions (ex: minor prédit comme major)

            Confusions typiques attendues sur xView2 :
              minor ↔ major  (aspects visuels proches)
              major ↔ destroyed  (frontière floue)
        ─────────────────────────────────────────────────────────────
        """
        CLASS_NAMES = PipelineConfig.CLASS_NAMES

        # ── Rapport textuel
        report = classification_report(
            y_true, y_pred,
            target_names = CLASS_NAMES,
            digits       = 4,
            zero_division= 0,
        )

        # ── Matrice de confusion
        cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))

        # ── F1 scores
        f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_per      = f1_score(y_true, y_pred, average=None,       zero_division=0)
        accuracy    = float(np.mean(y_true == y_pred))

        print(f"\n── Rapport de classification ──")
        print(report)
        self._print_confusion_matrix_console(cm, CLASS_NAMES)
        print(f"\nTop-1 Accuracy : {accuracy:.4f}")
        print(f"F1 macro       : {f1_macro:.4f}")
        print(f"F1 weighted    : {f1_weighted:.4f}")
        print(f"\nF1 par classe :")
        for name, f1 in zip(CLASS_NAMES, f1_per):
            bar = "█" * int(f1 * 25)
            print(f"  {name:<18} : {f1:.4f}  {bar}")

        # ── Sauvegarde des plots
        if save_plots:
            self._save_confusion_matrix_plot(
                cm, CLASS_NAMES, output_dir
            )
            self._save_f1_barplot(
                CLASS_NAMES, f1_per,
                f1_macro, f1_weighted,
                output_dir
            )
            self._save_report_txt(report, output_dir)

        return {
            "accuracy":        round(accuracy, 4),
            "f1_macro":        round(f1_macro, 4),
            "f1_weighted":     round(f1_weighted, 4),
            "f1_per_class":    {n: round(float(f), 4)
                                for n, f in zip(CLASS_NAMES, f1_per)},
            "report":          report,
            "confusion_matrix":cm,
            "n_samples":       int(len(y_true)),
        }


    # ──────────────────────────────────────────────────────
    # HELPERS AFFICHAGE ET PLOTS
    # ──────────────────────────────────────────────────────

    def _print_confusion_matrix_console(self, cm: np.ndarray, names: List[str]):
        """Affiche la matrice de confusion 4×4 dans la console."""
        print("\n── Matrice de confusion ──")
        print("       " + "  ".join(f"{n[:7]:>9}" for n in names))
        print("       " + "  ".join([f"{'(pred)':>9}"] * 4))
        for i, row in enumerate(cm):
            tag  = f"[GT{i}] {names[i][:7]:<7}"
            vals = "  ".join(f"{v:>9}" for v in row)
            print(f"  {tag}  {vals}")
        print()

    def _save_confusion_matrix_plot(
        self, cm: np.ndarray, names: List[str], output_dir: str
    ):
        """Sauvegarde la matrice de confusion en PNG."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Matrice des counts absolus
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
        disp.plot(ax=axes[0], colorbar=True, cmap="Blues", xticks_rotation=20)
        axes[0].set_title("Matrice de confusion — Valeurs absolues", pad=12)

        # Matrice normalisée (taux par classe GT)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        disp2   = ConfusionMatrixDisplay(
            confusion_matrix=np.round(cm_norm, 2), display_labels=names
        )
        disp2.plot(ax=axes[1], colorbar=True, cmap="Blues", xticks_rotation=20)
        axes[1].set_title("Matrice de confusion — Normalisée (recall par classe)", pad=12)

        plt.suptitle("Pipeline YOLO26 — Évaluation Classification xView2",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = Path(output_dir) / "confusion_matrix_pipeline.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluate] Confusion matrix → {path}")

    def _save_f1_barplot(
        self,
        names:       List[str],
        f1_per:      np.ndarray,
        f1_macro:    float,
        f1_weighted: float,
        output_dir:  str,
    ):
        """Sauvegarde un barplot des F1 par classe."""
        colors = [
            PipelineConfig.CLASS_COLORS.get(n, (128, 128, 128))
            for n in names
        ]
        # Normaliser en [0,1] pour matplotlib
        colors_mpl = [(r/255, g/255, b/255) for r, g, b in colors]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, f1_per, color=colors_mpl, edgecolor="white",
                      linewidth=1.5, zorder=3)

        for bar, val in zip(bars, f1_per):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        ax.axhline(f1_macro,    linestyle="--", color="#444", linewidth=1.2,
                   label=f"F1 macro = {f1_macro:.3f}")
        ax.axhline(f1_weighted, linestyle=":",  color="#888", linewidth=1.2,
                   label=f"F1 weighted = {f1_weighted:.3f}")

        ax.set_ylim(0, 1.08)
        ax.set_ylabel("F1-score", fontsize=11)
        ax.set_title("F1-score par classe — Pipeline YOLO26 (test set)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_xticklabels(names, rotation=15, ha="right")

        plt.tight_layout()
        path = Path(output_dir) / "f1_per_class_pipeline.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluate] F1 barplot → {path}")

    def _save_report_txt(self, report: str, output_dir: str):
        """Sauvegarde le rapport sklearn en fichier texte."""
        path = Path(output_dir) / "classification_report.txt"
        path.write_text(report)
        print(f"[Evaluate] Rapport textuel → {path}")

    def _print_evaluation_summary(
        self,
        seg_metrics:    Dict,
        cls_metrics:    Dict,
        pipeline_stats: Dict,
    ):
        """Affiche le résumé final de l'évaluation."""
        print("\n" + "═" * 60)
        print("  RÉSUMÉ ÉVALUATION PIPELINE YOLO26 — xView2")
        print("═" * 60)

        print("\n── YOLO26-seg ──────────────────────────────────────")
        if "error" not in seg_metrics and "note" not in seg_metrics:
            print(f"  mAP50   (box)  : {seg_metrics.get('box_map50', 'N/A')}")
            print(f"  mAP50-95(box)  : {seg_metrics.get('box_map50_95', 'N/A')}")
            print(f"  mAP50   (mask) : {seg_metrics.get('mask_map50', 'N/A')}")
            print(f"  mAP50-95(mask) : {seg_metrics.get('mask_map50_95', 'N/A')}")
            print(f"  Precision      : {seg_metrics.get('precision', 'N/A')}")
            print(f"  Recall         : {seg_metrics.get('recall', 'N/A')}")
        else:
            print(f"  {seg_metrics.get('note', seg_metrics.get('error', 'N/A'))}")

        print("\n── YOLO26-cls (via pipeline) ───────────────────────")
        if "error" not in cls_metrics:
            print(f"  Top-1 Accuracy : {cls_metrics.get('accuracy', 'N/A')}")
            print(f"  F1 macro       : {cls_metrics.get('f1_macro', 'N/A')}")
            print(f"  F1 weighted    : {cls_metrics.get('f1_weighted', 'N/A')}")
            print(f"  Samples évalués: {cls_metrics.get('n_samples', 'N/A')}")
        else:
            print(f"  {cls_metrics.get('error')}")

        print("\n── Statistiques pipeline ───────────────────────────")
        print(f"  Images test          : {pipeline_stats.get('total_images', 'N/A')}")
        print(f"  Bâtiments GT (JSON)  : {pipeline_stats.get('total_gt_buildings', 'N/A')}")
        print(f"  Bâtiments détectés   : {pipeline_stats.get('total_detected', 'N/A')}")
        print(f"  Matches GT↔YOLO      : {pipeline_stats.get('total_matched', 'N/A')}")
        print(f"  Recall détection     : {pipeline_stats.get('detection_recall', 'N/A')}")
        print("═" * 60)


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

    # ── Initialiser le pipeline
    pipeline = YOLO26Pipeline()

    # ── Récupérer le test set
    all_records = find_post_disaster_images()
    _, _, test_records = stratified_event_split(all_records)

    # ── Évaluation complète
    results = pipeline.evaluate(
        test_records   = test_records,
        output_dir     = "evaluation_results/",
        seg_data_yaml  = "yolo_dataset/segmentation/dataset.yaml",
        save_plots     = True,
        verbose        = True,
    )












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

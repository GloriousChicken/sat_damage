"""
yolo26_pipeline.py  (v3 — détection seule, un seul modèle)
===========================================================
Pipeline YOLO26n — un seul modèle de détection à 4 classes.

Différence fondamentale vs version précédente (v2) :
─────────────────────────────────────────────────────────────────────
  v2 (seg + cls) :  2 modèles en cascade
    YOLO26-seg  → détecte les bâtiments (1 classe "building")
    YOLO26-cls  → classifie chaque crop (4 classes de dommages)
    Avantage    : segmentation précise au pixel près
    Inconvénient: 2 modèles, 2 inférences, pipeline complexe

  v3 (detect) :   1 seul modèle
    YOLO26n-det → détecte ET classe en une seule passe
                  (4 classes : no-damage / minor / major / destroyed)
    Avantage    : pipeline simple, inférence rapide, 1 modèle
    Inconvénient: pas de masque de segmentation pixel-précis
─────────────────────────────────────────────────────────────────────

Méthodes principales :
    predict(image)            → inférence sur une image
    evaluate(test_records)    → évaluation complète test set
    draw_results(image, dets) → visualisation annotée
    summarize(detections)     → score de dommage global

═══════════════════════════════════════════════════════════════════
MÉTRIQUES YOLO26n DÉTECTION — RÉFÉRENCE COMPLÈTE
═══════════════════════════════════════════════════════════════════

╔══════════════════════════════════════════════════════════════════╗
║  MÉTRIQUES NATIVES YOLO26n (calculées par model.val())          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  mAP50      (mean Average Precision @ IoU=0.50)                 ║
║    Métrique principale de YOLO26n.                              ║
║    Un bâtiment est "bien détecté" si IoU(pred_bbox, GT) ≥ 0.50 ║
║    ET la classe prédite = classe GT.                            ║
║    → Mesure à la fois localisation ET classification.           ║
║    Référence xView2 : >0.40 = acceptable                        ║
║                                                                  ║
║  mAP50-95   (mAP moyenné de IoU=0.50 à 0.95, pas=0.05)         ║
║    Version plus sévère : pénalise les bboxes imprécis.          ║
║    Référence xView2 : >0.25 = acceptable                        ║
║                                                                  ║
║  Precision  = TP / (TP + FP)                                    ║
║    Parmi toutes les détections YOLO, quelle fraction est        ║
║    correctement localisée ET correctement classifiée ?          ║
║                                                                  ║
║  Recall     = TP / (TP + FN)                                    ║
║    Parmi tous les vrais bâtiments annotés, quelle fraction      ║
║    a été détectée avec la bonne classe ?                        ║
║    ► Pour xView2 : Recall > Precision est préférable            ║
║      (mieux vaut sur-détecter que manquer des dommages)         ║
║                                                                  ║
║  Ces 4 métriques sont calculées globalement (toutes classes     ║
║  confondues) ET par classe (no-damage, minor, major, destroyed).║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  MÉTRIQUES COMPLÉMENTAIRES (calculées par evaluate())           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Ces métriques ne sont PAS calculées par YOLO nativement.       ║
║  Elles sont obtenues en faisant tourner model.predict() sur     ║
║  chaque image du test set et en matchant les prédictions aux    ║
║  bâtiments ground truth du JSON.                                ║
║                                                                  ║
║  Rapport de classification  (sklearn)                           ║
║    Pour chaque classe de dommage :                              ║
║      Precision  = TP_i / (TP_i + FP_i)                         ║
║      Recall     = TP_i / (TP_i + FN_i)                         ║
║      F1-score   = 2 × P × R / (P + R)                          ║
║      Support    = nombre de vrais samples de la classe          ║
║                                                                  ║
║  Matrice de confusion 4×4                                       ║
║    Ligne  = classe réelle  (ground truth JSON)                  ║
║    Colonne= classe prédite (YOLO26n)                            ║
║    Confusions typiques : minor ↔ major, major ↔ destroyed       ║
║                                                                  ║
║  F1 macro     : moyenne des F1 par classe (toutes comptent      ║
║                 autant — recommandé car xView2 est déséquilibré)║
║  F1 weighted  : F1 pondéré par le support (favorise no-damage)  ║
║  Top-1 Accuracy : % de prédictions exactement correctes         ║
║                                                                  ║
║  Matching GT ↔ prédiction :                                     ║
║    Chaque prédiction YOLO est associée au bâtiment JSON         ║
║    le plus proche (distance centroïde ≤ MAX_MATCH_DIST px).     ║
║    Seules les prédictions avec un match valide contribuent      ║
║    aux métriques de classification.                             ║
╚══════════════════════════════════════════════════════════════════╝

Pré-requis :
    pip install ultralytics scikit-learn pillow numpy matplotlib
"""

import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from yolo_data_preparation import find_post_disaster_images, stratified_event_split

# ═══════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════

class PipelineConfig:
    # ── Modèle YOLO26n fine-tuné sur xView2
    DET_WEIGHTS  = "../.pyenv/runs/detect/train/weights/best.pt"

    # ── Seuils d'inférence
    CONF         = 0.25      # confiance minimale pour valider une détection
    IOU          = 0.45      # seuil NMS (suppression des doublons)
    IMGSZ        = 640       # taille d'image (multiple de 32)

    # ── 4 classes de dommages (ordre = class_id YOLO)
    CLASS_NAMES  = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    DAMAGE_TO_LABEL = {
        "no-damage":     0,
        "minor-damage":  1,
        "major-damage":  2,
        "destroyed":     3,
        "un-classified": None,
    }

    # ── Couleurs de visualisation par classe
    CLASS_COLORS = {
        "no-damage":    (  0, 200,   0),   # vert
        "minor-damage": (255, 200,   0),   # jaune
        "major-damage": (255, 100,   0),   # orange
        "destroyed":    (220,   0,   0),   # rouge
    }

    # ── Device
    DEVICE       = "cuda:0"

    # ── Distance max (pixels) pour matcher une prédiction à un GT JSON
    MAX_MATCH_DIST = 60


# ═══════════════════════════════════════════════════════════
# 2. PIPELINE — UN SEUL MODÈLE YOLO26n
# ═══════════════════════════════════════════════════════════

class YOLO26Pipeline:
    """
    Pipeline YOLO26n mono-modèle pour la détection et classification
    des dommages de bâtiments sur images post-disaster.

    Différence clé vs v2 (seg+cls) :
        v2 : self.seg_model + self.cls_model  → 2 modèles
        v3 : self.det_model uniquement        → 1 modèle

    Méthode principale :
        predict(image)  →  List[Dict] par bâtiment détecté
    """

    def __init__(
        self,
        det_weights: str  = PipelineConfig.DET_WEIGHTS,
        device:      str  = PipelineConfig.DEVICE,
        verbose:     bool = True,
    ):
        """
        Charge le modèle YOLO26n de détection.

        Args:
            det_weights : chemin vers best.pt après entraînement
                          (généré par : yolo train model=yolo26n.pt ...)
            device      : "cpu", "cuda:0", "cuda:1", ...
            verbose     : affiche le résumé du modèle si True
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics non installé.\n"
                "Installer avec : pip install ultralytics"
            )

        print(f"[Pipeline] Chargement YOLO26n : {det_weights}")
        self.det_model = YOLO(det_weights)
        self.det_model.to(device)
        self.device  = device
        self.verbose = verbose

        if verbose:
            self.det_model.info()

        print(f"[Pipeline] Modèle chargé sur {device} ✓")


    # ──────────────────────────────────────────────────────
    # INFÉRENCE
    # ──────────────────────────────────────────────────────

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        tmp_path:    str = "/tmp/yolo_det_infer.png",
    ) -> List[Dict]:
        """
        Détecte et classifie les bâtiments en une seule passe YOLO26n.

        Différence vs v2 :
            v2 : segment_buildings() → classify_damage()  (2 passes)
            v3 : une seule passe YOLO26n → bbox + classe directement

        Args:
            image_input : chemin str ou np.array (H, W, 3) uint8

        Retourne :
            [{
                "bbox":       (x0, y0, x1, y1),  ← pixels absolus
                "centroid":   (cx, cy),
                "label":      "major-damage",     ← classe YOLO26n
                "label_id":   2,
                "confidence": 0.73,               ← score de confiance
                "probabilities": {                ← scores par classe
                    "no-damage":    0.05,
                    "minor-damage": 0.12,
                    "major-damage": 0.73,
                    "destroyed":    0.10,
                }
            }, ...]
        """
        # Gérer l'input numpy (inférence FastAPI)
        if isinstance(image_input, np.ndarray):
            Image.fromarray(image_input).save(tmp_path)
            image_path = tmp_path
        else:
            image_path = str(image_input)

        results = self.det_model.predict(
            source  = image_path,
            conf    = PipelineConfig.CONF,
            iou     = PipelineConfig.IOU,
            imgsz   = PipelineConfig.IMGSZ,
            verbose = False,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        result      = results[0]
        boxes       = result.boxes
        class_names = (self.det_model.names
                       if hasattr(self.det_model, "names")
                       else {i: n for i, n in
                             enumerate(PipelineConfig.CLASS_NAMES)})

        for i in range(len(boxes)):
            x0, y0, x1, y1 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf_score      = float(boxes.conf[i].item())
            cls_id          = int(boxes.cls[i].item())
            label           = class_names.get(cls_id,
                                              PipelineConfig.CLASS_NAMES[cls_id])

            # scores par classe via probs si disponibles, sinon approxiamtion
            if hasattr(boxes, "probs") and boxes.probs is not None:
                probs_vec = boxes.probs[i].cpu().numpy()
            else:
                # YOLO26n detect ne retourne qu'un score de confiance
                # On construit un vecteur fictif concentré sur la classe prédite
                probs_vec = np.zeros(len(PipelineConfig.CLASS_NAMES))
                probs_vec[cls_id] = conf_score

            detections.append({
                "bbox":         (x0, y0, x1, y1),
                "centroid":     ((x0 + x1) // 2, (y0 + y1) // 2),
                "label":        label,
                "label_id":     cls_id,
                "confidence":   round(conf_score, 4),
                "probabilities": {
                    class_names.get(j, f"class_{j}"): round(float(p), 4)
                    for j, p in enumerate(probs_vec)
                },
            })

        if self.verbose:
            counts = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            print(f"[Pipeline] {len(detections)} bâtiments → {counts}")

        return detections


    # ──────────────────────────────────────────────────────
    # ÉVALUATION COMPLÈTE SUR LE TEST SET
    # ──────────────────────────────────────────────────────

    def evaluate(
        self,
        test_records:  List[Dict],
        data_yaml:     str  = "yolo_dataset/detection/dataset.yaml",
        output_dir:    str  = "evaluation_results",
        save_plots:    bool = True,
        verbose:       bool = True,
    ) -> Dict:
        """
        Évaluation complète du pipeline sur le test set.

        Deux évaluations complémentaires sont réalisées :

        ┌─────────────────────────────────────────────────────────┐
        │  ÉVALUATION 1 — Métriques natives YOLO26n               │
        │  (via model.val() sur le split test du dataset.yaml)    │
        │  → mAP50, mAP50-95, Precision, Recall                  │
        │    calculés par classe ET en moyenne globale            │
        └─────────────────────────────────────────────────────────┘
        ┌─────────────────────────────────────────────────────────┐
        │  ÉVALUATION 2 — Métriques custom par bâtiment          │
        │  (via predict() sur chaque image + matching GT JSON)    │
        │  → Rapport de classification, matrice de confusion,     │
        │    F1 macro/weighted, Top-1 Accuracy                   │
        └─────────────────────────────────────────────────────────┘

        Args:
            test_records : liste de dicts {"post_img", "post_label", "event"}
                           → obtenu avec find_post_disaster_images()
                             puis stratified_event_split()[2]
            data_yaml    : chemin dataset.yaml YOLO pour model.val()
            output_dir   : dossier de sauvegarde des plots et rapports
            save_plots   : sauvegarde confusion matrix + F1 barplot si True
            verbose      : affiche la progression si True

        Retourne :
            {
              "yolo_native": {           ← métriques model.val()
                  "map50":       float,  ← mAP@0.50 global (toutes classes)
                  "map50_95":    float,  ← mAP@0.50:0.95 global
                  "precision":   float,
                  "recall":      float,
                  "per_class":   {       ← métriques par classe
                      "no-damage":    {"map50": .., "precision": .., "recall": ..},
                      "minor-damage": {...},
                      "major-damage": {...},
                      "destroyed":    {...},
                  }
              },
              "classification": {        ← métriques custom sklearn
                  "accuracy":    float,
                  "f1_macro":    float,
                  "f1_weighted": float,
                  "f1_per_class":{"no-damage": float, ...},
                  "report":      str,
                  "confusion_matrix": np.ndarray,
                  "n_samples":   int,
              },
              "pipeline_stats": {        ← statistiques du matching
                  "total_images":       int,
                  "total_gt_buildings": int,
                  "total_detected":     int,
                  "total_matched":      int,
                  "detection_recall":   float,
              }
            }
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── Évaluation 1 : métriques natives YOLO
        yolo_metrics = self._evaluate_yolo_native(data_yaml, verbose)

        # ── Évaluation 2 : métriques custom
        cls_metrics, pipeline_stats = self._evaluate_custom(
            test_records, output_dir, save_plots, verbose
        )

        self._print_summary(yolo_metrics, cls_metrics, pipeline_stats)

        # ── Sauvegarde JSON
        import json as json_mod
        result = {
            "yolo_native":    yolo_metrics,
            "classification": {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in cls_metrics.items()
                if k != "confusion_matrix"
            },
            "pipeline_stats": pipeline_stats,
        }
        out_path = Path(output_dir) / "evaluation_summary.json"
        with open(out_path, "w") as f:
            json_mod.dump(result, f, indent=2)
        print(f"\n[Evaluate] Résumé sauvegardé → {out_path}")

        result["classification"]["confusion_matrix"] = cls_metrics.get(
            "confusion_matrix"
        )
        return result


    # ──────────────────────────────────────────────────────
    # ÉVALUATION 1 : métriques natives YOLO26n
    # ──────────────────────────────────────────────────────

    def _evaluate_yolo_native(self, data_yaml: str, verbose: bool) -> Dict:
        """
        Lance model.val() sur le split test.

        Métriques YOLO26n retournées :
        ──────────────────────────────────────────────────────────────
        mAP50     : mAP global à IoU=0.50 (toutes classes confondues)
                    Interprétation : un bâtiment compte comme détecté
                    si bbox IoU ≥ 0.50 ET bonne classe.

        mAP50-95  : mAP global moyenné de IoU=0.50 à 0.95.
                    Pénalise les bboxes approximatifs.

        Precision : TP / (TP + FP) global
        Recall    : TP / (TP + FN) global

        per_class : les mêmes métriques calculées séparément
                    pour chaque classe (no-damage, minor, major, destroyed)
                    → permet de voir quelle classe est la plus difficile
                      (en général : minor et major sont les plus confondues)
        """
        print("\n" + "─" * 58)
        print("  Évaluation 1 — Métriques natives YOLO26n (model.val())")
        print("─" * 58)

        if not Path(data_yaml).exists():
            msg = f"dataset.yaml introuvable : {data_yaml}"
            print(f"  [WARN] {msg}")
            print("  → Fournir data_yaml= pour activer cette évaluation.")
            return {"error": msg}

        try:
            val_res = self.det_model.val(
                data   = data_yaml,
                split  = "test",
                imgsz  = PipelineConfig.IMGSZ,
                verbose= verbose,
            )

            # ── Métriques globales
            metrics = {
                "map50":     round(float(val_res.box.map50), 4),
                "map50_95":  round(float(val_res.box.map),   4),
                "precision": round(float(val_res.box.p.mean()), 4),
                "recall":    round(float(val_res.box.r.mean()), 4),
            }

            # ── Métriques par classe
            # val_res.box.ap_class_index : indices des classes
            # val_res.box.ap50  : AP@0.50 par classe
            # val_res.box.p, .r : precision/recall par classe
            per_class = {}
            names = (self.det_model.names
                     if hasattr(self.det_model, "names") else {})

            if hasattr(val_res.box, "ap_class_index"):
                for idx, cls_idx in enumerate(val_res.box.ap_class_index):
                    cls_name = names.get(int(cls_idx),
                                        PipelineConfig.CLASS_NAMES[int(cls_idx)])
                    per_class[cls_name] = {
                        "map50":     round(float(val_res.box.ap50[idx]),  4),
                        "map50_95":  round(float(val_res.box.ap[idx]),    4),
                        "precision": round(float(val_res.box.p[idx]),     4),
                        "recall":    round(float(val_res.box.r[idx]),     4),
                    }

            metrics["per_class"] = per_class

            # ── Affichage tabulaire
            print(f"\n  {'Classe':<20} {'mAP50':>8} {'mAP50-95':>10}"
                  f" {'Precision':>10} {'Recall':>8}")
            print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
            print(f"  {'GLOBAL':<20} {metrics['map50']:>8.4f}"
                  f" {metrics['map50_95']:>10.4f}"
                  f" {metrics['precision']:>10.4f}"
                  f" {metrics['recall']:>8.4f}")
            for cls_name, m in per_class.items():
                print(f"  {cls_name:<20} {m['map50']:>8.4f}"
                      f" {m['map50_95']:>10.4f}"
                      f" {m['precision']:>10.4f}"
                      f" {m['recall']:>8.4f}")

            return metrics

        except Exception as e:
            print(f"  [ERROR] model.val() a échoué : {e}")
            return {"error": str(e)}


    # ──────────────────────────────────────────────────────
    # ÉVALUATION 2 : métriques custom par bâtiment
    # ──────────────────────────────────────────────────────

    def _evaluate_custom(
        self,
        test_records: List[Dict],
        output_dir:   str,
        save_plots:   bool,
        verbose:      bool,
    ) -> Tuple[Dict, Dict]:
        """
        Fait tourner predict() sur toutes les images du test set
        et compare les prédictions aux labels JSON ground truth.

        Retourne (cls_metrics, pipeline_stats).
        """
        print("\n" + "─" * 58)
        print("  Évaluation 2 — Métriques custom (sklearn)")
        print("─" * 58)

        y_true = []
        y_pred = []

        total_gt   = 0
        total_det  = 0
        total_match= 0
        errors     = 0

        for i, record in enumerate(test_records):
            if verbose and i % 50 == 0:
                print(f"  [{i+1:>4}/{len(test_records)}] "
                      f"{record.get('event', '')}")
            try:
                gt_buildings = self._load_gt(record["post_label"])
                total_gt    += len(gt_buildings)
                if not gt_buildings:
                    continue

                detections   = self.predict(record["post_img"])
                total_det   += len(detections)
                if not detections:
                    continue

                matched = self._match_gt_to_preds(gt_buildings, detections)
                total_match += len(matched)

                for gt_lbl, pred_lbl in matched:
                    y_true.append(gt_lbl)
                    y_pred.append(pred_lbl)

            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  [ERROR] "
                          f"{Path(record['post_img']).name} → {e}")

        pipeline_stats = {
            "total_images":       len(test_records),
            "total_gt_buildings": total_gt,
            "total_detected":     total_det,
            "total_matched":      total_match,
            "detection_recall":   round(total_match / max(total_gt, 1), 4),
            "errors":             errors,
        }

        if not y_true:
            print("  [WARN] Aucun sample évalué.")
            return {"error": "Aucun sample évalué"}, pipeline_stats

        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

        cls_metrics = self._compute_metrics(
            y_true, y_pred, output_dir, save_plots
        )
        return cls_metrics, pipeline_stats


    def _load_gt(self, json_path: str) -> List[Dict]:
        """Charge les bâtiments ground truth depuis un JSON post-disaster."""
        from shapely import wkt as shapely_wkt
        from shapely.geometry import shape as shapely_shape

        with open(json_path, "r") as f:
            data = json.load(f)

        gt = []
        for feat in data.get("features", {}).get("xy", []):
            damage = feat.get("properties", {}).get("subtype", "un-classified")
            label  = PipelineConfig.DAMAGE_TO_LABEL.get(damage)
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
            gt.append({
                "label":    label,
                "centroid": ((minx + maxx) / 2, (miny + maxy) / 2),
            })

        return gt


    def _match_gt_to_preds(
        self,
        gt_buildings: List[Dict],
        detections:   List[Dict],
    ) -> List[Tuple[int, int]]:
        """
        Associe chaque bâtiment GT à la détection YOLO la plus proche.

        Stratégie : distance euclidienne centroïde GT ↔ centroïde YOLO.
        Match valide si distance ≤ MAX_MATCH_DIST pixels.
        Un seul match par détection (greedy).

        Retourne :
            [(gt_label_int, pred_label_int), ...]
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
                matched.append((gt["label"], detections[best_idx]["label_id"]))
                used_det_idx.add(best_idx)

        return matched


    def _compute_metrics(
        self,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        output_dir: str,
        save_plots: bool,
    ) -> Dict:
        """
        Calcule et affiche toutes les métriques sklearn.

        ──────────────────────────────────────────────────────────────
        Rapport de classification :
            Precision  par classe : TP_i / (TP_i + FP_i)
            Recall     par classe : TP_i / (TP_i + FN_i)
            F1-score   par classe : 2PR/(P+R) — robuste au déséquilibre
            Support    par classe : nb de vrais samples dans le test set

        F1 macro    : moyenne des F1 par classe (toutes comptent autant)
                      → À privilégier pour xView2 où toutes les classes
                        de dommage sont critiques pour la réponse d'urgence

        F1 weighted : F1 pondéré par le support
                      → Favorise les classes majoritaires (no-damage)

        Matrice de confusion 4×4 :
            Ligne i  = vrais bâtiments de classe i (GT)
            Col j    = bâtiments prédits comme classe j
            CM[i][i] = bons : vrais positifs de la classe i
            CM[i][j] = erreurs : vrais i prédits comme j

            Lecture pour xView2 :
              CM[1][2] élevé = minor prédit comme major (sur-estimation)
              CM[2][1] élevé = major prédit comme minor (sous-estimation)
        ──────────────────────────────────────────────────────────────
        """
        CLASS_NAMES = PipelineConfig.CLASS_NAMES

        report      = classification_report(
            y_true, y_pred,
            target_names = CLASS_NAMES,
            digits       = 4,
            zero_division= 0,
        )
        cm          = confusion_matrix(y_true, y_pred, labels=list(range(4)))
        f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_per      = f1_score(y_true, y_pred, average=None,       zero_division=0)
        accuracy    = float(np.mean(y_true == y_pred))

        # ── Affichage console
        print(f"\n── Rapport de classification ──")
        print(report)
        self._print_cm_console(cm, CLASS_NAMES)
        print(f"\n  Top-1 Accuracy : {accuracy:.4f}")
        print(f"  F1 macro       : {f1_macro:.4f}")
        print(f"  F1 weighted    : {f1_weighted:.4f}")
        print(f"\n  F1 par classe :")
        for name, f1 in zip(CLASS_NAMES, f1_per):
            bar = "█" * int(f1 * 25)
            print(f"    {name:<18} : {f1:.4f}  {bar}")

        # ── Plots
        if save_plots:
            self._plot_confusion_matrix(cm, CLASS_NAMES, output_dir)
            self._plot_f1_bars(CLASS_NAMES, f1_per, f1_macro, f1_weighted,
                               output_dir)
            (Path(output_dir) / "classification_report.txt").write_text(report)
            print(f"  [Evaluate] Rapport → "
                  f"{Path(output_dir)/'classification_report.txt'}")

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
    # HELPERS AFFICHAGE
    # ──────────────────────────────────────────────────────

    def _print_cm_console(self, cm: np.ndarray, names: List[str]):
        print("\n── Matrice de confusion ──")
        header = "                    " + "".join(f"{n[:9]:>11}" for n in names)
        print(header)
        print("                    " + "".join(["   (prédit)"] * 4))
        for i, row in enumerate(cm):
            tag  = f"  (réel) [{i}] {names[i][:9]:<9}"
            vals = "".join(f"{v:>11}" for v in row)
            print(tag + vals)
        print()

    def _print_summary(
        self,
        yolo_m:    Dict,
        cls_m:     Dict,
        pip_stats: Dict,
    ):
        print("\n" + "═" * 60)
        print("  RÉSUMÉ ÉVALUATION — YOLO26n xView2")
        print("═" * 60)

        print("\n── Métriques YOLO26n natives ───────────────────────────")
        if "error" not in yolo_m:
            print(f"  mAP50       : {yolo_m.get('map50', 'N/A')}")
            print(f"  mAP50-95    : {yolo_m.get('map50_95', 'N/A')}")
            print(f"  Precision   : {yolo_m.get('precision', 'N/A')}")
            print(f"  Recall      : {yolo_m.get('recall', 'N/A')}")
            if yolo_m.get("per_class"):
                print("  Par classe  :")
                for cls_name, m in yolo_m["per_class"].items():
                    print(f"    {cls_name:<18} mAP50={m['map50']:.4f}  "
                          f"P={m['precision']:.4f}  R={m['recall']:.4f}")
        else:
            print(f"  {yolo_m.get('error')}")

        print("\n── Métriques custom (sklearn) ──────────────────────────")
        if "error" not in cls_m:
            print(f"  Top-1 Accuracy : {cls_m.get('accuracy', 'N/A')}")
            print(f"  F1 macro       : {cls_m.get('f1_macro', 'N/A')}")
            print(f"  F1 weighted    : {cls_m.get('f1_weighted', 'N/A')}")
            print(f"  Samples évalués: {cls_m.get('n_samples', 'N/A')}")
        else:
            print(f"  {cls_m.get('error')}")

        print("\n── Statistiques pipeline ───────────────────────────────")
        print(f"  Images test          : {pip_stats.get('total_images')}")
        print(f"  Bâtiments GT (JSON)  : {pip_stats.get('total_gt_buildings')}")
        print(f"  Bâtiments détectés   : {pip_stats.get('total_detected')}")
        print(f"  Matches GT↔YOLO      : {pip_stats.get('total_matched')}")
        print(f"  Recall détection     : {pip_stats.get('detection_recall')}")
        print("═" * 60)


    # ──────────────────────────────────────────────────────
    # PLOTS
    # ──────────────────────────────────────────────────────

    def _plot_confusion_matrix(
        self, cm: np.ndarray, names: List[str], output_dir: str
    ):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names).plot(
            ax=axes[0], colorbar=True, cmap="Blues", xticks_rotation=20
        )
        axes[0].set_title("Matrice de confusion — Valeurs absolues", pad=12)

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        ConfusionMatrixDisplay(
            confusion_matrix=np.round(cm_norm, 2), display_labels=names
        ).plot(ax=axes[1], colorbar=True, cmap="Blues", xticks_rotation=20)
        axes[1].set_title(
            "Matrice de confusion — Normalisée (recall par classe)", pad=12
        )

        plt.suptitle("YOLO26n — Évaluation Classification xView2",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = Path(output_dir) / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Evaluate] Confusion matrix → {path}")

    def _plot_f1_bars(
        self,
        names:    List[str],
        f1_per:   np.ndarray,
        f1_macro: float,
        f1_w:     float,
        output_dir: str,
    ):
        colors_mpl = [
            tuple(c / 255 for c in PipelineConfig.CLASS_COLORS.get(n, (128,128,128)))
            for n in names
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, f1_per, color=colors_mpl,
                      edgecolor="white", linewidth=1.5, zorder=3)
        for bar, val in zip(bars, f1_per):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.axhline(f1_macro, linestyle="--", color="#333", linewidth=1.2,
                   label=f"F1 macro = {f1_macro:.3f}")
        ax.axhline(f1_w,     linestyle=":",  color="#888", linewidth=1.2,
                   label=f"F1 weighted = {f1_w:.3f}")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("F1-score", fontsize=11)
        ax.set_title("F1-score par classe — YOLO26n (test set)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_xticklabels(names, rotation=15, ha="right")
        plt.tight_layout()
        path = Path(output_dir) / "f1_per_class.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Evaluate] F1 barplot → {path}")


    # ──────────────────────────────────────────────────────
    # VISUALISATION ET RÉSUMÉ
    # ──────────────────────────────────────────────────────

    def draw_results(
        self,
        image_input: Union[str, np.ndarray],
        detections:  List[Dict],
        save_path:   str  = "pipeline_output.png",
    ) -> np.ndarray:
        """Dessine les bboxes et labels sur l'image post-disaster."""
        img  = (Image.open(image_input) if isinstance(image_input, str)
                else Image.fromarray(image_input)).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12
            )
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            label   = det.get("label", "unknown")
            conf    = det.get("confidence", 0.0)
            x0,y0,x1,y1 = det["bbox"]
            r,g,b   = PipelineConfig.CLASS_COLORS.get(label, (128,128,128))

            draw.rectangle([x0,y0,x1,y1], outline=(r,g,b), width=2)
            text     = f"{label} {conf:.2f}"
            bbox_txt = draw.textbbox((x0, y0 - 14), text, font=font)
            draw.rectangle(bbox_txt, fill=(r,g,b,200))
            draw.text((x0, y0 - 14), text, fill=(255,255,255), font=font)

        img.save(save_path)
        print(f"[Pipeline] Résultat → {save_path}")
        return np.array(img)

    def summarize(self, detections: List[Dict]) -> Dict:
        """Score de dommage global (0=intact → 1=tout détruit)."""
        counts  = {n: 0 for n in PipelineConfig.CLASS_NAMES}
        weights = {"no-damage": 0.0, "minor-damage": 0.33,
                   "major-damage": 0.67, "destroyed": 1.0}
        for det in detections:
            lbl = det.get("label", "no-damage")
            if lbl in counts:
                counts[lbl] += 1
        total = sum(counts.values())
        return {
            "total_buildings": total,
            "counts":          counts,
            "damage_score":    round(
                sum(counts[l] * weights[l] for l in PipelineConfig.CLASS_NAMES)
                / max(total, 1), 4
            ),
        }


# ═══════════════════════════════════════════════════════════
# 3. POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Charger le pipeline (un seul modèle)
    pipeline = YOLO26Pipeline()

    # ── Récupérer le test set
    all_records          = find_post_disaster_images()
    _, _, test_records   = stratified_event_split(all_records)

    # ── Évaluation complète
    results = pipeline.evaluate(test_records)

    # ── Inférence et visualisation sur une image exemple
    img_path   = "test_post.png" # test_records[0]["post_img"]
    detections = pipeline.predict(img_path)
    pipeline.draw_results(img_path, detections, save_path="pipeline_output.png")
    print(pipeline.summarize(detections))

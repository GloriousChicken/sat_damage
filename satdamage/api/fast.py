import pandas as pd
import numpy as np
import json
import time
import rasterio
import gc
import tensorflow as tf
from PIL import Image
from io import BytesIO
from typing import List, Dict
from shapely import wkt as shapely_wkt
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from satdamage.ml_logic import registry
from satdamage.ml_logic.preprocessor import _polygon_to_bbox, crop_building
from satdamage.params import MODEL_NAMES, DAMAGE_TO_BINARY, DAMAGE_TO_CLASS, MODEL_MODE
from contextlib import asynccontextmanager

# ── Model registry
models: Dict[str, tf.keras.Model] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan function to load models at startup and clean up at shutdown.
    """
    # Chargement au démarrage
    deb = time.time()
    for name in MODEL_NAMES:
        model = registry.load_model(model_name=name)
        if model is not None:
            models[name] = model
            print(f"  ✅ {name} loaded  (input: {model.input_shape}  output: {model.output_shape})")
        else:
            print(f"  ⚠️  {name} not found — skipped")
    fin = time.time()
    print(f"\n✅ All models ready in {fin - deb:.2f} seconds.")
    yield
    models.clear()
    gc.collect()

app = FastAPI(lifespan=lifespan)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# ── Helpers
def load_json_buildings(data: dict) -> List[Dict]:
    """
    Load building polygons and properties from JSON data.
     - data: dict loaded from JSON file, expected to have "features" -> "xy" list
    Returns: list of dicts with "polygon" (shapely geometry) and "properties" (dict)
    """
    return [
        {"polygon": shapely_wkt.loads(f["wkt"]), "properties": f.get("properties", {})}
        for f in data["features"]["xy"]
    ]


def load_image(content: bytes, filename: str) -> np.ndarray:
    """
    Load image from bytes content, supporting TIFF and PNG formats.
     - content: bytes of the image file
     - filename: used to determine the format
    Returns: HxWxC image array
    """
    if filename.endswith((".tif", ".tiff")):
        with rasterio.open(BytesIO(content)) as src:
            return src.read().transpose(1, 2, 0)
    elif filename.endswith(".png"):
        return np.array(Image.open(BytesIO(content)).convert("RGB"))
    raise HTTPException(status_code=400, detail=f"Unsupported format: {filename}")


def build_crops(img: np.ndarray, buildings: List[Dict]) -> np.ndarray:
    """
    Crop building images based on their polygons.
     - img: HxWxC image array
     - buildings: list of dicts with "polygon" and "properties"
    Returns: array of cropped building images
    """
    h, w = img.shape[:2]
    return np.array([
        crop_building(img, _polygon_to_bbox(b["polygon"], w, h))
        for b in buildings
    ])


def get_input(model_name: str, pre_crops: np.ndarray, post_crops: np.ndarray) -> np.ndarray:
    """
    Route input based on what the model actually expects — read from input_shape.
     - If model expects 6 channels, concatenate pre/post crops
     - If model expects 3 channels, use post crops only
     - Otherwise, raise an error
    """
    expected_channels = models[model_name].input_shape[-1]
    if expected_channels == 6:
        return np.concatenate([pre_crops, post_crops], axis=-1)
    elif expected_channels == 3:
        return post_crops
    raise HTTPException(
        status_code=500,
        detail=f"Unexpected input channels for {model_name}: {expected_channels}"
    )


def get_ground_truth(buildings_post: List[Dict], mode: str) -> Dict[int, list]:
    mapping = DAMAGE_TO_CLASS if mode == "multiclass" else DAMAGE_TO_BINARY
    return {
        i: [mapping.get(b["properties"].get("subtype", "no-damage"), 0)]
        for i, b in enumerate(buildings_post)
    }


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    model: str = Query(default="efficientnet", description="Model to use for inference"),
):
    """
    Expects 4 files via multipart/form-data:
      - pre_disaster image  (.png / .tif)
      - post_disaster image (.png / .tif)
      - pre_disaster label  (.json, xBD format)
      - post_disaster label (.json, xBD format)

    Optional query param: ?model=cnn_concat | cnn_dual | efficientnet

    Returns:
      {
        "model": "cnn_concat",
        "mode":  "multiclass",
        "buildings": [
          {"index": 0, "ground_truth": 0, "prediction": 1, "confidence": 0.87},
          ...
        ]
      }
    """
    if len(files) != 4:
        raise HTTPException(status_code=400, detail='Exactly 4 files are required.')

    if model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not loaded. Available: {list(models.keys())}"
        )

    # ── Parse uploads
    pre_img = post_img = pre_label = post_label = None

    for file in files:
        content = await file.read()
        fname = file.filename

        if fname.endswith(".json"):
            data = json.loads(content)
            if "pre_disaster" in fname:
                pre_label = data
            elif "post_disaster" in fname:
                post_label = data
            else:
                raise HTTPException(status_code=400, detail=f"JSON filename must contain 'pre_disaster' or 'post_disaster': {fname}")
        else:
            img = load_image(content, fname)
            if "pre_disaster" in fname:
                pre_img = img
            elif "post_disaster" in fname:
                post_img = img
            else:
                raise HTTPException(status_code=400, detail=f"Image filename must contain 'pre_disaster' or 'post_disaster': {fname}")

    for name, val in [("pre_img", pre_img), ("post_img", post_img),
                      ("pre_label", pre_label), ("post_label", post_label)]:
        if val is None:
            raise HTTPException(status_code=400, detail=f"Missing: {name}")

    # ── Build crops
    buildings_pre  = load_json_buildings(pre_label)
    buildings_post = load_json_buildings(post_label)
    pre_crops  = build_crops(pre_img,  buildings_pre)
    post_crops = build_crops(post_img, buildings_post)

    # ── Select input tensor based on model's actual input shape
    x = get_input(model, pre_crops, post_crops)

    # ── Inference
    deb = time.time()
    raw_preds = models[model].predict(x, batch_size=32, verbose=0)
    fin = time.time()
    print(f"Inference ({model}, input:{x.shape}) on {len(x)} crops: {fin-deb:.2f}s")

    # ── Decode predictions
    mode = MODEL_MODE
    if mode == "binary":
        class_preds = (raw_preds > 0.5).astype(int).flatten()
        confidences = np.where(class_preds == 1, raw_preds.flatten(), 1 - raw_preds.flatten())
    else:
        class_preds = np.argmax(raw_preds, axis=-1)
        confidences = raw_preds[np.arange(len(raw_preds)), class_preds]

    # ── Ground truth
    gt = get_ground_truth(buildings_post, mode)

    # ── Build response
    buildings_out = [
        {
            "index":        i,
            "ground_truth": gt[i][0],
            "prediction":   int(class_preds[i]),
            "confidence":   round(float(confidences[i]), 4),
        }
        for i in range(len(buildings_post))
    ]

    return {
        "model":     model,
        "mode":      mode,
        "buildings": buildings_out,
    }


@app.get("/models")
def list_models():
    """
    Returns loaded models and their input/output shapes.
    """
    return {
        name: {
            "input_shape":  str(m.input_shape),
            "output_shape": str(m.output_shape),
        }
        for name, m in models.items()
    }


@app.get("/")
def root():
    return {"status": "ok", "models_loaded": list(models.keys())}

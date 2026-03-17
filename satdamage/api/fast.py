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

app = FastAPI()
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...), model_name: str = Query(...)):
    """
    Endpoint pour uploader une paire d'images pré/post et leurs labels associés.
    Attendu : 4 fichiers - pré/post image (TIFF) + pré/post label (JSON)
    """
    def api_load_json_buildings(data: dict) -> List[Dict]:
        """
        Loads building annotations from a JSON file and returns a list of building dicts with 'polygon' key.
        """
        features = data["features"]["xy"]
        return [{"polygon": shapely_wkt.loads(feature["wkt"])} for feature in features]

    if len(files) != 4:
        raise HTTPException(status_code=400, detail='Exactly 4 files are required: pre-disaster image, post-disaster image, and label file.')

    if model_name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f'Model name must be one of {MODEL_NAMES}.')

    pre_img    = None
    post_img   = None
    pre_label  = None
    post_label = None

    for file in files:
        content = await file.read()
        if file.filename.endswith(".json"):
            if "pre_disaster" in file.filename:
                pre_label = json.loads(content)
            elif "post_disaster" in file.filename:
                post_label = json.loads(content)
        else:
            if file.filename.endswith((".tif", ".tiff")):
                with rasterio.open(BytesIO(content)) as src:
                    img = src.read()
                if "pre_disaster" in file.filename:
                    pre_img = img.transpose(1, 2, 0)
                elif "post_disaster" in file.filename:
                    post_img = img.transpose(1, 2, 0)
                else:
                    raise HTTPException(status_code=400, detail='Image files must contain "pre_disaster" or "post_disaster" in their filename.')
            elif file.filename.endswith(".png"):
                img = Image.open(BytesIO(content)).convert("RGB")
                if "pre_disaster" in file.filename:
                    pre_img = np.array(img)
                elif "post_disaster" in file.filename:
                    post_img = np.array(img)
                else:
                    raise HTTPException(status_code=400, detail='Image files must contain "pre_disaster" or "post_disaster" in their filename.')
            else:
                raise HTTPException(status_code=400, detail='Unsupported file type. Only TIFF and PNG are allowed.')

    h, w = pre_img.shape[:2]
    buildings_pre = api_load_json_buildings(pre_label)
    buildings_post = api_load_json_buildings(post_label)
    pre_crops  = np.array([crop_building(pre_img,  _polygon_to_bbox(b["polygon"], w, h)) for b in buildings_pre])
    post_crops = np.array([crop_building(post_img, _polygon_to_bbox(b["polygon"], w, h)) for b in buildings_post])
    y = np.concatenate([pre_crops, post_crops], axis=-1)

    deb = time.time()

    # Libérer explicitement avant de charger le nouveau
    if hasattr(app.state, 'model'):
        del app.state.model
    gc.collect()

    # Pour GPU (Keras/TF)
    tf.keras.backend.clear_session()

    # Charger le modèle depuis le registry
    app.state.model = registry.load_model(model_name=model_name)
    fin = time.time()

    if app.state.model is None:
        raise HTTPException(status_code=500, detail=f'Model {model_name} could not be loaded.')

    print(f"Model {model_name} loaded in {fin - deb:.2f} seconds.")

    # Prediction from y data (pre/post crops concatenated)
    deb = time.time()
    prediction = app.state.model.predict(y)
    fin = time.time()
    print(f"Prediction run in {fin - deb:.2f} seconds.")

    if MODEL_MODE == "multiclass":
        result = {
            i: [DAMAGE_TO_CLASS.get(building["properties"]["subtype"]), float(prediction[i])]
            for i, building in enumerate(post_label["features"]["xy"])
            }
    else:
        result = {
            i: [DAMAGE_TO_BINARY.get(building["properties"]["subtype"]), float(prediction[i])]
            for i, building in enumerate(post_label["features"]["xy"])
            }

    return result


@app.get("/")
def root():
    response = {
        'greeting': 'Hello'
    }
    return response

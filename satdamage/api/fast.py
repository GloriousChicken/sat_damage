import pandas as pd
import numpy as np
import json
import rasterio
from PIL import Image
from io import BytesIO
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from satdamage.ml_logic import registry
from satdamage.ml_logic.preprocessor import *
from satdamage.params import MODEL_TARGET

app = FastAPI()
app.state.model = registry.load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    Endpoint pour uploader une paire d'images pré/post et leurs labels associés.
    Attendu : 4 fichiers - pré/post image (TIFF) + pré/post label (JSON)
    """
    def api_load_json_buildings(data: dict) -> List[Dict]:
        """
        Loads building annotations from a JSON file and returns a list of building dicts with 'polygon' key.
        """
        buildings = []
        features  = data.get("features", {}).get("xy", [])

        for feature in features:
            props  = feature.get("properties", {})
            damage = props.get("subtype", "un-classified")

            geom = None

            # ── Format WKT (le plus courant dans xView2)
            wkt_str = feature.get("wkt", "")
            if wkt_str:
                try:
                    geom = shapely_wkt.loads(wkt_str)
                except Exception:
                    pass

            # ── Format GeoJSON (fallback)
            if geom is None:
                try:
                    geom = shape(feature.get("geometry", {}))
                except Exception:
                    continue

            if geom is None or geom.is_empty or not geom.is_valid:
                continue

            buildings.append({"polygon": geom, "damage": damage})

        return buildings

    if len(files) != 4:
        raise HTTPException(status_code=400, detail='Exactly 4 files are required: pre-disaster image, post-disaster image, and label file.')

    pre_img    = None
    post_img   = None
    pre_label  = None
    post_label = None

    for file in files:
        content = await file.read()
        if file.filename.endswith(".json"):
            if "pre_disaster" in file.filename:
                pre_label = json.loads(content)
                # pre_label_name = file.filename
            elif "post_disaster" in file.filename:
                post_label = json.loads(content)
                # post_label_name = file.filename
        else:
            if file.filename.endswith((".tif", ".tiff")):
                with rasterio.open(BytesIO(content)) as src:
                    img = src.read()
                if "pre_disaster" in file.filename:
                    pre_img = img.transpose(1, 2, 0)
                elif "post_disaster" in file.filename:
                    post_img = img.transpose(1, 2, 0)
            elif file.filename.endswith(".png"):
                img = Image.open(BytesIO(content)).convert("RGB")
                if "pre_disaster" in file.filename:
                    pre_img = np.array(img)
                elif "post_disaster" in file.filename:
                    post_img = np.array(img)

    h, w = pre_img.shape[:2]

    buildings_pre = api_load_json_buildings(pre_label)
    buildings_post = api_load_json_buildings(post_label)

    image_pairs = []
    for building_pre, building_post in zip(buildings_pre, buildings_post):
        bbox_pre = polygon_to_pixel_bbox(building_pre["polygon"], w, h)
        bbox_post = polygon_to_pixel_bbox(building_post["polygon"], w, h)
        pre_crop = crop_building(pre_img, bbox_pre)
        post_crop = crop_building(post_img, bbox_post)
        image_pairs.append((pre_crop, post_crop))

    pre_images = np.array([p[0] for p in image_pairs])
    post_images = np.array([p[1] for p in image_pairs])

    y = np.concatenate([pre_images, post_images], axis=-1)
    prediction = app.state.model.predict(y)

    result = {i: float(prediction[i, 0]) for i in range(len(prediction))}

    return {
        "message": f"Successfuly uploaded {[file.filename for file in files]}",
        "status": "completed",
        "prediction": result
        }

@app.get("/")
def root():
    response = {
        'greeting': 'Hello'
    }
    return response

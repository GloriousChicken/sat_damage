import pandas as pd
import json
from PIL import Image as PILimage
from io import BytesIO
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from satdamage.ml_logic import preprocessor, registry
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

    if len(files) != 3:
        raise HTTPException(status_code=400, detail='Exactly 3 files are required: pre-disaster image, post-disaster image, and label file.')

    pre_img    = None
    post_img   = None
    post_label = None

    for file in files:
        content = await file.read()

        if "pre_disaster" in file.filename:
            pre_img = content
        elif "post_disaster" in file.filename and file.filename.endswith(".json"):
            post_label = json.loads(content)
        elif "post_disaster" in file.filename:
            post_img = content

    if not all([pre_img, post_img, post_label]):
        raise HTTPException(status_code=400, detail=f"Fichiers manquants - reçus {[f.filename for f in files]}")

    # ... traitement
    # MODEL PREDICTION
    # prediction = api.state.model.predict(img)

    return {
        "message": f"Successfuly uploaded {[file.filename for file in files]}",
        "status": "completed",
        "prediction": "model prediction"
        }

@app.get("/")
def root():
    response = {
        'greeting': 'Hello'
    }
    return response

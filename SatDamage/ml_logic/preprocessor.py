import os
import json
import numpy as np
import rasterio
from shapely import wkt
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def crop_buildings(img_path, json_path, padding=10, target_size=128):
    """
    Handles both .tif (via rasterio) and .png (via PIL).
    """
    # --- 1. Load Image Based on Extension ---
    if img_path.endswith('.png'):
        # PNG (8-bit) - standard loading
        with Image.open(img_path) as pil_img:
            img_np = np.array(pil_img.convert("RGB")) # (H, W, 3)
        H, W, _ = img_np.shape
        # Re-order to (3, H, W) to stay compatible with the rest of the script
        img = img_np.transpose(2, 0, 1)
        is_tif = False
    else:
        # TIFF (16-bit) - professional satellite loading
        with rasterio.open(img_path) as src:
            img = src.read()  # (bands, H, W)
            H, W = src.height, src.width
        is_tif = True

    with open(json_path) as f:
        data = json.load(f)

    features = [f for f in data['features']['xy']]
    output = {}

    for f in features:
        geom = wkt.loads(f['wkt'])
        minx, miny, maxx, maxy = geom.bounds

        x1, y1 = max(0, int(minx) - padding), max(0, int(miny) - padding)
        x2, y2 = min(W, int(maxx) + padding), min(H, int(maxy) + padding)

        crop = img[:, y1:y2, x1:x2]

        if is_tif:
            # Special scaling for 16-bit TIFF
            def scale_band(band):
                p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
                if p98 == p2: return np.zeros_like(band, dtype=np.float32)
                return np.clip((band.astype(np.float32) - p2) / (p98 - p2), 0, 1)
            rgb = np.stack([scale_band(crop[0]), scale_band(crop[1]), scale_band(crop[2])], axis=-1)
        else:
            # Simple scaling for 8-bit PNG
            rgb = crop.transpose(1, 2, 0).astype(np.float32) / 255.0

        # Resize
        pil = Image.fromarray((rgb * 255).astype(np.uint8))
        pil = pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
        output[f['properties']['uid']] = np.array(pil)

    return output


def preprocess_sample(sample, data_path):
    # Detect extension
    extension = ".png" if os.path.exists(os.path.join(data_path, "images", f"{sample}_pre_disaster.png")) else ".tif"
    
    img_path_pre = os.path.join(data_path, "images", f"{sample}_pre_disaster{extension}")
    img_path_post = os.path.join(data_path, "images", f"{sample}_post_disaster{extension}")
    json_path_pre = os.path.join(data_path, "labels", f"{sample}_pre_disaster.json")
    json_path_post = os.path.join(data_path, "labels", f"{sample}_post_disaster.json")

    with open(json_path_post) as f:
        data = json.load(f)
    
    label_map = {feat['properties']['uid']: feat['properties']['subtype'] for feat in data['features']['xy']}
    
    pre_cropped = crop_buildings(img_path_pre, json_path_pre)
    post_cropped = crop_buildings(img_path_post, json_path_post)

    X_list, y_list, Z_list = [], [], []

    for uid in pre_cropped.keys():
        if uid in post_cropped and uid in label_map:
            combined = np.concatenate([pre_cropped[uid], post_cropped[uid]], axis=2)
            X_list.append(combined)
            y_list.append(label_map[uid])
            Z_list.append([uid, sample])

    if not X_list:
        return np.empty((0, 128, 128, 6)), np.empty(0), np.empty((0, 2))

    return np.stack(X_list, axis=0), np.array(y_list), np.array(Z_list)


def preprocess(data_dir, max_workers=None):
    image_dir = os.path.join(data_dir, "images")
    image_list = os.listdir(image_dir)
    label_list = os.listdir(os.path.join(data_dir, "labels"))

    samples = []
    for image in image_list:
        if image.endswith("_post_disaster.png") or image.endswith("_post_disaster.tif"):
            ext = ".png" if image.endswith(".png") else ".tif"
            image_pfx = image.replace(f"_post_disaster{ext}", "")
            if os.path.exists(os.path.join(data_dir, "labels", f"{image_pfx}_post_disaster.json")):
                samples.append(image_pfx)

    # --- FIXED INDENTATION: The processing starts AFTER the loop collects all IDs ---
    print(f"Found {len(samples)} valid image pairs. Starting extraction...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(partial(preprocess_sample, data_path=data_dir), samples))

    if results:
        X_list, y_list, Z_list = zip(*results)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        Z = np.concatenate(Z_list, axis=0)
    else:
        X, y, Z = np.empty((0, 128, 128, 6)), np.empty(0), np.empty((0, 2))

    return X, y, Z
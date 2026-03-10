import os
import json
import numpy as np
import rasterio
from shapely import wkt
from PIL import Image


def crop_buildings(tif_path, json_path, padding=10, target_size=128):
    """Crop buildings from a TIFF using its polygon geometry."""
    with rasterio.open(tif_path) as src:
        img = src.read()  # (bands, H, W)
        H, W = src.height, src.width

    with open(json_path) as f:
        data = json.load(f)

    features = [f for f in data['features']['xy'] ]
    output = {}

    for i,f in enumerate(features):
        geometry = f['wkt']
        geom = wkt.loads(geometry)
        minx, miny, maxx, maxy = geom.bounds

        # Add padding and clip to image bounds
        x1 = max(0, int(minx) - padding)
        y1 = max(0, int(miny) - padding)
        x2 = min(W, int(maxx) + padding)
        y2 = min(H, int(maxy) + padding)

        crop = img[:, y1:y2, x1:x2]

        # Scale each band
        def scale_band(band):
            p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
            if p98 == p2:
                return np.zeros_like(band, dtype=np.float32)
            return np.clip((band.astype(np.float32) - p2) / (p98 - p2), 0, 1)

        rgb = np.dstack([scale_band(crop[0]), scale_band(crop[1]), scale_band(crop[2])])

        # Resize to target_size x target_size
        pil = Image.fromarray((rgb * 255).astype(np.uint8))
        pil = pil.resize((target_size, target_size), Image.BILINEAR)

        # Add to output dict
        output[f['properties']['uid']] = np.array(pil)

    return output


def preprocess_sample(sample, data_path):
    tif_path_pre = data_path + "images/" + sample + "_pre_disaster.tif"
    tif_path_post = data_path + "images/" + sample + "_post_disaster.tif"
    json_path_pre = data_path + "labels/" + sample + "_pre_disaster.json"
    json_path_post = data_path + "labels/" + sample + "_post_disaster.json"

    with open(json_path_post) as f:
        data = json.load(f)
    labels = np.array([feat['properties']['subtype'] for feat in data['features']['xy']])
    ids = [feat['properties']['uid'] for feat in data['features']['xy']]
    annot = np.stack([ids, [sample] * len(ids)], axis=1)

    pre_cropped = crop_buildings(tif_path_pre, json_path_pre)
    post_cropped = crop_buildings(tif_path_post, json_path_post)
    all_cropped = [ np.concatenate( [pre_cropped[i],post_cropped[i]], axis= 2) for i in pre_cropped ]
    X = np.stack(all_cropped, axis=0)
    return X, labels, annot

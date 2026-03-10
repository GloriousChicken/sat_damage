import os
import json
import numpy as np
import rasterio
from shapely import wkt
from PIL import Image

"""
This module provides preprocessing utilities for satellite imagery data used in damage assessment.
It includes functions to crop buildings from TIFF images based on polygon geometries and to
preprocess samples by combining pre- and post-disaster images with labels.
"""


def crop_buildings(tif_path, json_path, padding=10, target_size=128):
    """
    Crop buildings from a TIFF image using polygon geometries from a JSON file.

    This function reads a TIFF image and a corresponding JSON file containing building
    geometries in WKT format. It crops each building with optional padding, scales the
    pixel values, and resizes the crops to a target size.

    Parameters
    ----------
    tif_path : str
        Path to the TIFF image file.
    json_path : str
        Path to the JSON file containing building geometries.
    padding : int, optional
        Number of pixels to add as padding around each building's bounding box.
        Default is 10.
    target_size : int, optional
        Size to resize each cropped image to (target_size x target_size).
        Default is 128.

    Returns
    -------
    dict
        A dictionary where keys are building UIDs and values are numpy arrays
        representing the cropped and processed images (RGB, float32, 0-1 range).
    """
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
    """
    Preprocess a sample by cropping buildings from pre- and post-disaster images and preparing labels and annotations.

    This function processes a given sample by loading pre- and post-disaster TIFF images
    and their corresponding JSON label files. It crops buildings from both images,
    concatenates them, and extracts labels and annotations for damage assessment.

    Parameters
    ----------
    sample : str
        The identifier for the sample (e.g., 'hurricane-florence_00000027').
    data_path : str
        Path to the directory containing 'images/' and 'labels/' subdirectories.

    Returns
    -------
    X : numpy.ndarray
        A 4D array of shape (n_buildings, target_size, target_size, 6) where the last
        dimension concatenates RGB channels from pre- and post-disaster images.
    labels : numpy.ndarray
        A 1D array of damage subtypes for each building.
    annot : numpy.ndarray
        A 2D array of shape (n_buildings, 2) containing building UIDs and sample names.
    """
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

def preprocess(data_dir):
    """
    Preprocesses satellite damage data from the given directory.

    This function loads pre-disaster and post-disaster satellite images along with their
    corresponding labels from the specified data directory. It processes each sample
    by checking for the presence of required files and concatenating the processed data
    into arrays.

    Parameters:
    data_dir (str): Path to the data directory containing 'images' and 'labels' subdirectories.
                   The 'images' directory should contain .tif files, and 'labels' should contain .json files.

    Returns:
    tuple: A tuple containing three numpy arrays:
        - X (numpy.ndarray): Array of shape (n_samples, 128, 128, 6) containing the processed image data.
        - y (numpy.ndarray): Array of shape (n_samples,) containing the labels.
        - Z (numpy.ndarray): Array of shape (n_samples, 2) containing additional metadata : file name and building's id.
    """
    image_dir = data_dir + "/images"
    label_dir = data_dir + "/labels"

    image_pfx = ""

    # Listing directory content
    image_list = os.listdir(image_dir)
    label_list = os.listdir(label_dir)

    X = np.empty((0, 128, 128, 6))
    # print("X", X.shape)
    y = np.empty(0)
    # print("y", y.shape)
    Z = np.empty((0, 2))
    # print("Z", Z.shape)

    # Looping over files
    for image in image_list:
        # Checking presence of 4 files with same prefix
        if image.endswith("_post_disaster.tif"):
            image_pfx = image.replace("_post_disaster.tif", "")
            image_pre = image_pfx + "_pre_disaster.tif"
            label_post = image_pfx + "_post_disaster.json"
            label_pre = image_pfx + "_pre_disaster.json"
            if image_pre not in image_list:
                print("Fichier image pre-disaster manquant")
                continue
            if label_post not in label_list:
                print("Fichier label post-disaster manquant")
                continue
            if label_pre not in label_list:
                print("Fichier label post-disaster manquant")
                continue
            # Calling preprocessing of sample
            new_X, new_y, new_Z = preprocess_sample(image_pfx)
            X = np.concatenate((X, new_X), axis=0)
            y = np.concatenate((y, new_y))
            Z = np.concatenate((Z, new_Z), axis=0)

    return X, y, Z

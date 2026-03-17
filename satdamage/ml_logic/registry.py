import glob
import os
import time
import json
import tempfile

from io import BytesIO
from colorama import Fore, Style
from tensorflow import keras

from satdamage.params import *


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
    metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")

    if MODEL_TARGET == "local":
        if params is not None:
            with open(params_path, "wb") as file:
                json.dump(params, file)

        if metrics is not None:
            with open(metrics_path, "wb") as file:
                json.dump(metrics, file)

        print("✅ Results saved locally")

    return None


def load_model(model_name: str) -> keras.Model:
    """
    Return a saved model locally (latest one in alphabetical order).
    Return None (but do not Raise) if no model is found
    """
    if model_name not in MODEL_NAMES:
        print(f"❌ Model name {model_name} not recognized. Available models are: {MODEL_NAMES}")
        return None

    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    local_model_directory = LOCAL_REGISTRY_PATH
    local_model_paths = glob.glob(f"{local_model_directory}/*{model_name}*.h5")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return latest_model


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """

    if model is None:
        print("❌ No model to save")
        return None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if model.name:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "checkpoints", f"{timestamp}_{model.name}.h5")
    else:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "checkpoints", f"{timestamp}.h5")

    model.save(model_path)
    print("✅ Model saved locally")

    return None

import glob
import os
import time
import json
import tempfile
import zipfile

from io import BytesIO
from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

from satdamage.params import *
from satdamage.ml_logic.model import BinaryF1Score

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
    metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")

    if MODEL_TARGET == "local":
        # Save params locally
        if params is not None:
            with open(params_path, "wb") as file:
                json.dump(params, file)

        # Save metrics locally
        if metrics is not None:
            with open(metrics_path, "wb") as file:
                json.dump(metrics, file)

        print("✅ Results saved locally")

        return None

    elif MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Save params in the cloud
        if params is not None:
            params_filename = params_path.split("/")[-1] # e.g. "20230208-161047.pickle" for instance
            blob = bucket.blob(f"params/{params_filename}")
            blob.upload_from_string(
                data=json.dumps(params),
                content_type="application/json"
            )

        # Save metrics in the cloud
        if metrics is not None:
            metrics_filename = metrics_path.split("/")[-1] # e.g. "20230208-161047.pickle" for instance
            blob = bucket.blob(f"metrics/{metrics_filename}")
            blob.upload_from_string(
                data=json.dumps(metrics),
                content_type="application/json"
            )

        print("✅ Results saved to GCS")

    return None


def load_model(model_name: str) -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'
    Return None (but do not Raise) if no model is found
    """
    if model_name not in MODEL_NAMES:
        print(f"❌ Model name {model_name} not recognized. Available models are: {MODEL_NAMES}")
        return None

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = LOCAL_REGISTRY_PATH
        local_model_paths = glob.glob(f"{local_model_directory}/*{model_name}*.h5")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        deb = time.time()
        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
        latest_model = keras.models.load_model(most_recent_model_path_on_disk)
        fin = time.time()
        print(f"✅ Model loaded from local disk in {fin - deb:.2f} seconds.")

        return latest_model

    elif MODEL_TARGET == "gcs":
        deb = time.time()
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = [
            blob for blob in client.get_bucket(BUCKET_NAME).list_blobs(prefix="models")
            if model_name in blob.name
        ]
        if not blobs:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME} with name {model_name}")
            return None

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            print(f"Latest model found in GCS bucket {BUCKET_NAME} with name {model_name} is: {latest_blob.name} (updated on {latest_blob.updated})")
            # Télécharger en mémoire sans toucher le disque
            buffer = BytesIO()
            latest_blob.download_to_file(buffer)
            buffer.seek(0)

            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                tmp.write(buffer.read())
                tmp_path = tmp.name

            try:
                with zipfile.ZipFile(tmp_path, 'r') as z:
                    with z.open('config.json') as f:
                        config = json.load(f)

                config_str = json.dumps(config)
                needs_binary_f1 = "BinaryF1Score" in config_str
                custom_objects = {"BinaryF1Score": BinaryF1Score()} if needs_binary_f1 else {}
                latest_model = keras.models.load_model(
                    tmp_path,
                    custom_objects=custom_objects
                    )
                fin = time.time()
                print(f"Model {latest_model.name} loaded from GCS bucket {BUCKET_NAME} in {fin - deb:.2f} seconds.")

            except Exception as e:
                print(f"Error while loading model : {type(e).__name__}: {e}")
                return None
            finally:
                os.remove(tmp_path)

            print("\n✅ Latest model downloaded from cloud storage")
            return latest_model

        except:
            print(f"\n❌ {model_name} model not found in GCS bucket {BUCKET_NAME}")
            return None

    return None

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    """

    if model is None:
        print("❌ No model to save")

        return None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if model.name:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "checkpoints", f"{timestamp}_{model.name}.h5")
    else:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "checkpoints", f"{timestamp}.h5")

    # Save model locally
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # Save model in the cloud
        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

    return None

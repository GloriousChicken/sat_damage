import os
import time
import tensorflow as tf
from SatDamage.ml_logic.model import Config

def save_model(model: tf.keras.Model, params: dict = None) -> None:
    """
    Saves the trained model with a timestamp.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save model locally
    model_path = os.path.join("training_outputs", "models", f"model_{timestamp}.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    
    print(f"✅ Model saved locally at {model_path}")
    
    # Optional: Save params to a JSON for tracking
    if params:
        import json
        params_path = os.path.join("training_outputs", "models", f"params_{timestamp}.json")
        with open(params_path, "w") as f:
            json.dump(params, f)
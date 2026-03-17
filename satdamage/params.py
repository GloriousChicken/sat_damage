import os
import tensorflow as tf
from pathlib import Path

##################  VARIABLES  ##################

##################  MAIN VARIABLES  #############
CROP_SIZE        = (128, 128)
CROP_PADDING     = 10
DAMAGE_TO_BINARY = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 1,
    "destroyed": 1
}
DAMAGE_TO_CLASS = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}
CLASS_NAMES = list(DAMAGE_TO_CLASS.keys())
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
RANDOM_SEED   = 42
BATCH_SIZE    = 32
MAX_WORKERS = 8
BALANCE_MAJORITY_RATIO = float(os.environ.get("BALANCE_MAJORITY_RATIO", "2.0"))

# EfficientNet parameters
EPOCHS_WARMUP   = 10
EPOCHS_FINETUNE = 30
LR_WARMUP    = 1e-3    # Phase 1 : backbone gelé
LR_FINETUNE  = 5e-5    # Phase 2 : fine-tuning couches profondes
UNFREEZE_LAYERS = 40

# CNN parameters
EPOCHS        = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
DROPOUT_RATE  = 0.5
FOCAL_GAMMA   = 1.0   # was 0.0; re-enable focal — class_weight experiment failed

# Model selection
MODEL_NAMES = ["cnn_concat", "cnn_dual", "efficientnet"]
MODEL_ARCHITECTURE = "cnn_concat"  # Options: "cnn_concat", "cnn_dual", "efficientnet"
MODEL_MODE = "multiclass" # multiclass or binary
NUM_CLASSES = len(DAMAGE_TO_CLASS)

# Sauvegarde
CHECKPOINT_PATH = f"checkpoints/satdamage_{MODEL_ARCHITECTURE}_best.keras"
LOG_DIR         = f"logs/satdamage_{MODEL_ARCHITECTURE}"

####################  LOCAL PATH  ############
# PROJECT_ROOT is the satdamage package directory
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = os.environ.get("DATA_DIR", str(PROJECT_ROOT / "challenge_dataset"))
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
CROPS_DIR   = os.environ.get("CROPS_DIR", str(PROJECT_ROOT.parent / "data" / "crops"))

# Always local (no GCP)
MODEL_TARGET = "local"

##################  GPU/M5 OPTIMIZATION  ############
# Optimize for Apple Silicon M5
USE_MIXED_PRECISION = True  # Use bfloat16 for faster training
NUM_PARALLEL_CALLS = os.cpu_count() or 8
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE  # Auto-optimize prefetch

##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "GloriousChicken", "sat_damage", "checkpoints")

##################  VALIDATIONS  ################

env_valid_options = dict(
    MODEL_TARGET=["local"]
)

def validate_env_value(env, valid_options):
    env_value = os.environ.get(env, "local")
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)

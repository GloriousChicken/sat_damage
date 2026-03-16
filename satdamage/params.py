import os
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
DATA_DIR = os.environ.get("DATA_DIR")
CROPS_DIR   = os.environ.get("CROPS_DIR", "data/crops")

##################  CLOUD VARIABLES  ############
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")

GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "GloriousChicken", "sat_damage", "checkpoints")

##################  VALIDATIONS  ################

env_valid_options = dict(
    MODEL_TARGET=["local", "gcs"]
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)

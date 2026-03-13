import os
##################  VARIABLES  ##################

##################  MAIN VARIABLES  #############
CROP_SIZE        = (128, 128)
CROP_PADDING     = 10
DAMAGE_TO_BINARY = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 1,
    "destroyed": 1,
    "un-classified": None
}
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
RANDOM_SEED   = 42
BATCH_SIZE    = 32
EPOCHS        = 50
SOURCE_SPLITS = ["train", "tier3"]

# Chemins xView2
TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"
TEST_DIR  = "data/test"

# Entraînement
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
DROPOUT_RATE  = 0.5

# Model selection
MODEL_ARCHITECTURE = "dual"  # Options: "concat", "dual"

# Sauvegarde
CHECKPOINT_PATH = f"checkpoints/cnn_damage_{MODEL_ARCHITECTURE}_best.keras"
LOG_DIR         = f"logs/cnn_damage_{MODEL_ARCHITECTURE}"

####################  LOCAL PATH  ############
DATA_DIR = os.environ.get("DATA_DIR")

##################  CLOUD VARIABLES  ############
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")

PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")

GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

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

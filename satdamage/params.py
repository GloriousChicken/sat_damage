import os

##################  VARIABLES  ##################

##################  MAIN VARIABLES  #############
CROP_SIZE        = os.environ.get("CROP_SIZE")
CROP_PADDING     = os.environ.get("CROP_PADDING")
DAMAGE_TO_BINARY = os.environ.get("DAMAGE_TO_BINARY")
TRAIN_RATIO      = os.environ.get("TRAIN_RATIO")
VAL_RATIO        = os.environ.get("VAL_RATIO")
TEST_RATIO       = os.environ.get("TEST_RATIO")
RANDOM_SEED      = os.environ.get("RANDOM_SEED")
BATCH_SIZE       = os.environ.get("BATCH_SIZE")
EPOCHS           = os.environ.get("EPOCHS")
SOURCE_SPLITS    = os.environ.get("SOURCE_SPLITS")

# Chemins xView2
TRAIN_DIR = os.environ.get("TRAIN_DIR")
VAL_DIR   = os.environ.get("VAL_DIR")
TEST_DIR  = os.environ.get("TEST_DIR")

# Entraînement
LEARNING_RATE  = os.environ.get("LEARNING_RATE")
WEIGHT_DECAY   = os.environ.get("WEIGHT_DECAY")
DROPOUT_RATE   = os.environ.get("DROPOUT_RATE")

# Sauvegarde
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")
LOG_DIR         = os.environ.get("LOG_DIR")

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

##################  VALIDATIONS  ################

env_valid_options = dict(
    MODEL_TARGET=["local", "gcs", "mlflow"]
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)

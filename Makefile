.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y satdamage || :
	@pip install -e .

run_api:
	uvicorn taxifare.api.fast:app --reload


################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
HTTPS_DIR=https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/
GS_DIR=gs://datascience-mlops/taxi-fare-ny

show_sources_all:
	-gsutil ls gs://${BUCKET_NAME}

reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_gcs_files


##################### CLEANING #####################

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints

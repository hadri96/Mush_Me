# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* Mush_Me/*.py

black:
	@black scripts/* Mush_Me/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr Mush_Me-*.dist-info
	@rm -fr Mush_Me.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#            GCP SETUP
# ----------------------------------

BUCKET_NAME=mush_me_hector2gt
BUCKET_NAME_HADRIEN=mush_me

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = trainings

##### Machine configuration - - - - - - - - - - - - - - - -

REGION=europe-west4

PYTHON_VERSION=3.7
RUNTIME_VERSION=2.4

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=Mush_Me
FILENAME=trainer
FILENAME_EFNET=trainerv2
FILENAME_RESNET=trainer_resnet
FILENAME_EFNETV3=trainerv3
FILENAME_EFNETV3_TRAIN=trainerv4
FILENAME_HADRIEN=trainer_hadrien
FILENAME_HADRIEN_V2=trainer_hadrien_v2
FILENAME_HADRIEN_V3=trainer_hadrien_v3

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=mush_me_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')


GCR_MULTI_REGION=eu.gcr.io
GCR_REGION=europe-west1
DOCKER_IMAGE_NAME=mush-me-api
GCP_PROJECT_ID=le-wagon-bootcamp-316213

docker_build:
	sudo docker build -t ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} ./back_end/
	sudo docker push ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}
	sudo gcloud run deploy --image ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} \
		--platform managed \
		--region ${GCR_REGION}

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs

gcp_submit_training_GPU:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--master-accelerator count=1,type=nvidia-tesla-t4 \
		--stream-logs
		
gcp_submit_training_GPU_resnet:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_RESNET} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--master-accelerator count=1,type=nvidia-tesla-t4 \
		--stream-logs


gcp_submit_training_GPU_efnet:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_EFNET} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--master-accelerator count=1,type=nvidia-tesla-t4 \
		--stream-logs

gcp_submit_training_GPU_efnetv3:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_EFNETV3} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--master-accelerator count=1,type=nvidia-tesla-t4 \
		--stream-logs

gcp_submit_training_GPU_efnetv3_train:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_EFNETV3_TRAIN} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--master-accelerator count=1,type=nvidia-tesla-t4 \
		--stream-logs

gcp_submit_training_GPU_hadrien:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME_HADRIEN}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_HADRIEN} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--master-accelerator count=1,type=nvidia-tesla-t4 \
		--stream-logs

gcp_submit_training_GPU_hadrien_v2:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME_HADRIEN}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_HADRIEN_V2} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--master-accelerator count=1,type=nvidia-tesla-t4 \
		--stream-logs

gcp_submit_training_GPU_hadrien_v3:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME_HADRIEN}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_HADRIEN_V3} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--master-accelerator count=1,type=nvidia-tesla-t4 \
		--stream-logs

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ __pycache__
	@rm -fr build dist *.dist-info *.egg-info
	@rm -fr */*.pyc

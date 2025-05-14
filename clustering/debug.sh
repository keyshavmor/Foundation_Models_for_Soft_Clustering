#!/bin/bash

#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu=36GB  
#SBATCH --cpus-per-task=1
#SBATCH --output=tumor.out
#SBATCH --error=tumor.err
#SBATCH --open-mode=truncate
#SBATCH --mail-user=maurdu@student.ethz.ch
#SBATCH --mail-type=START,END,FAIL

echo "--- Starting Slurm Job ---"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on host: $(hostname)"
echo "Initial CWD: $(pwd)"
echo "User: $(whoami)"
echo "Home: ${HOME}"
echo "-----------------------------"

echo "Loading eth_proxy module..."
module load eth_proxy
echo "eth_proxy module loaded. HTTP_PROXY=${HTTP_PROXY}" # Check if proxy var is set
echo "-----------------------------"

echo "Setting up Conda..."
eval "$(conda shell.bash hook)"
if [ $? -ne 0 ]; then echo "ERROR: conda shell.bash hook FAILED"; exit 1; fi
echo "Activating Conda environment: cancer_found"
conda activate cancer_found
if [ $? -ne 0 ]; then echo "ERROR: conda activate cancer_found FAILED"; exit 1; fi
echo "Conda environment activated: ${CONDA_DEFAULT_ENV}"
echo "Python interpreter: $(which python)"
echo "Python version: $(python --version)"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "-----------------------------"

TARGET_DIR="${HOME}/Foundation_models_NB/clustering"
echo "Attempting to cd to: ${TARGET_DIR}"
cd "${TARGET_DIR}"
if [ $? -ne 0 ]; then
    echo "ERROR: cd to ${TARGET_DIR} FAILED. Current CWD: $(pwd)"
    exit 1
fi
echo "Successfully changed CWD. New CWD: $(pwd)"
echo "-----------------------------"

PYTHON_SCRIPT="cancerfnd.py"
CONFIG_FILE="config_sn_tumor.yaml"

echo "Checking if Python script exists: ${TARGET_DIR}/${PYTHON_SCRIPT}"
ls -lh "${TARGET_DIR}/${PYTHON_SCRIPT}"
if [ ! -f "${TARGET_DIR}/${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script ${TARGET_DIR}/${PYTHON_SCRIPT} NOT FOUND."
    exit 1
fi

echo "Checking if Config file exists: ${TARGET_DIR}/${CONFIG_FILE}"
ls -lh "${TARGET_DIR}/${CONFIG_FILE}"
if [ ! -f "${TARGET_DIR}/${CONFIG_FILE}" ]; then
    echo "WARNING: Config file ${TARGET_DIR}/${CONFIG_FILE} NOT FOUND in CWD. Python script might fail to load it unless an absolute path is used or default logic handles it."
    # The Python script has its own check, so this is just a warning here.
fi
echo "-----------------------------"

echo "Executing command: python -u -X faulthandler cancerfnd.py --config=${CONFIG_FILE}"
python -u -X faulthandler "${PYTHON_SCRIPT}" --config="${CONFIG_FILE}"
PYTHON_EXIT_CODE=$?

echo "-----------------------------"
echo "Python script finished with exit code: ${PYTHON_EXIT_CODE}"
echo "Listing files in CWD (${TARGET_DIR}) after script execution:"
ls -lah "${TARGET_DIR}"
echo "--- Slurm Job Ended ---"
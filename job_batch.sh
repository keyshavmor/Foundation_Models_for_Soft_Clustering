#!/bin/bash

#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu=64GB
#SBATCH --output=cancer.out
#SBATCH --error=cancer.err
#SBATCH --open-mode=truncate # truncate overwrites output and error files, append just appends
#SBATCH --mail-user=maurdu@student.ethz.ch
#SBATCH --mail-type=END,FAIL
module load eth_proxy
eval "$(conda shell.bash hook)"
conda activate cancer_found
cd "${HOME}/Foundation_models_NB/notebooks" # path to your project folder
# your command
python run_clustering.py

#!/bin/bash

#SBATCH -n 1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=45GB
#SBATCH --output=cancerfnd.out
#SBATCH --error=cancerfnd.err
#SBATCH --open-mode=truncate # truncate overwrites output and error files, append just appends
#SBATCH --mail-user=maurdu@student.ethz.ch
#SBATCH --mail-type=START,END,FAIL
module load eth_proxy
eval "$(conda shell.bash hook)"
conda activate cancer_found
cd "${HOME}/Foundation_models_NB/clustering" # path to your project folder
# your command
python cancerfnd.py

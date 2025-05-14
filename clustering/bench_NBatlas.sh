#!/bin/bash

#SBATCH -n 1
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu=42GB
#SBATCH --cpus-per-task=1
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
srun python cancerfnd.py --config=config_NBatlas.yaml

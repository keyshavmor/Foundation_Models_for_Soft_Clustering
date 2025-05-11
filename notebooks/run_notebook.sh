#!/bin/bash
#SBATCH --job-name=run_notebook
#SBATCH --partition=jobs
#SBATCH --time=2:00:00        # adjust as needed
#SBATCH --account=pmlr_jobs
#SBATCH -o logs/notebook_%j.out
#SBATCH -e logs/notebook_%j.err

nvidia-smi

source ~/.bashrc
conda activate scimilarity_home

# execute the notebook in-place (or to a new file)
jupyter nbconvert \
    --to notebook \
    --execute scfoundation_evaluation_dataset_2.ipynb \
    --output scfoundation_evaluation_dataset_2_ran.ipynb \
    --ExecutePreprocessor.timeout=1200

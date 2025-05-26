#!/usr/bin/env bash
#SBATCH --job-name=emb_chunk           # shown in squeue
#SBATCH --partition=jobs               # GPU partition on your cluster
#SBATCH --time=24:00:00                # same wall-clock as before
#SBATCH --account=pmlr_jobs
#SBATCH -o logs/emb_chunk_%j.out
#SBATCH -e logs/emb_chunk_%j.err

set -euo pipefail

################################################################################
# 1 · positional arguments
################################################################################
CHUNK_PATH=${1:? "Need the chunk file as first argument"}

# If the driver exported TEMP_EMBS it arrives here; otherwise fall back to
# the hard-coded path you used in the driver.
SAVE_DIR=${TEMP_EMBS:-/work/scratch/ndickenmann/tmp_embeddings_smaller_variable_genes_only}

################################################################################
# 2 · environment
################################################################################
source ~/.bashrc
conda activate scimilarity_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

################################################################################
# 3 · derive a short index for nice, unique filenames
################################################################################
# chunk_000  →  000
# chunk_000  →  000
raw=$(basename "$CHUNK_PATH" | grep -o '[0-9]\+')
# Force base-10 interpretation, then zero-pad to three digits:
IDX=$(printf "%03d" "$((10#$raw))")

################################################################################
# 4 · run the embedding
################################################################################
echo "[$(date +'%F %T')]  Running get_embedding.py on ${CHUNK_PATH} (idx=${IDX})"

python get_embedding.py \
  --task_name Embedding_part${IDX} \
  --input_type singlecell \
  --output_type cell \
  --pool_type all \
  --tgthighres t4 \
  --data_path "$CHUNK_PATH" \
  --save_path "$SAVE_DIR" \
  --pre_normalized F \

echo "[$(date +'%F %T')]  Done – output in ${SAVE_DIR}"

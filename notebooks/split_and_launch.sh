#!/usr/bin/env bash

# ─── USER SETTINGS ──────────────────────────────────────────────────────────────
INPUT=/work/scratch/ndickenmann/dataset_2_3_combined_preprocessed_highly_variable_genes_only.csv
TEMP_CHUNKS=/work/scratch/ndickenmann/tmp_chunks_smaller_variable_genes_only
TEMP_EMBS=/work/scratch/ndickenmann/tmp_embeddings_smaller_variable_genes_only
LINES_PER_CHUNK=1000
CHUNK_JOB_SCRIPT=/home/ndickenmann/scFoundation/model/embedding_array.sh   # GPU script
GPU_PARTITION=jobs   
SLEEP_SEC=60     
# ────────────────────────────────────────────────────────────────────────────────

set -euo pipefail
source ~/.bashrc
conda activate scimilarity_home

#rm -rf "$TEMP_CHUNKS" "$TEMP_EMBS"                # ← clear old stuff
mkdir -p "$TEMP_CHUNKS" "$TEMP_EMBS"

#echo "Splitting CSV into ${LINES_PER_CHUNK}-line chunks…"
#header=$(head -n 1 "$INPUT")
#tail -n +2 "$INPUT" | split -l "$LINES_PER_CHUNK" \
#        --numeric-suffixes=0 -d -a 3 - "$TEMP_CHUNKS/chunk_"
#for part in "$TEMP_CHUNKS"/chunk_*; do          # restore header
#  { printf '%s\n' "$header"; cat "$part"; } > "${part}.csv" && mv "${part}.csv" "$part"
#done
#echo "Created $(ls "$TEMP_CHUNKS"/chunk_* | wc -l) chunks."

########################################
# 3 · submit each chunk, one‐by‐one
########################################
START_AT=064                        # first chunk index to process

for part in $(ls "$TEMP_CHUNKS"/chunk_* | sort \
             | awk -F'chunk_' -v s="$START_AT" '{n=$2+0} n>=s'); do

    # ––––– guard: wait until ≤ 1 pending GPU job –––––
    while [[ $(squeue -u "$USER" -p "$GPU_PARTITION" --states=PD --noheader | wc -l) -ge 1 ]]; do
        sleep "$SLEEP_SEC"
    done

    # ––––– submit this chunk –––––
    sbatch --partition="$GPU_PARTITION" \
           --time=20:00:00 \
           --export=ALL,TEMP_EMBS="$TEMP_EMBS" \
           "$CHUNK_JOB_SCRIPT" "$part"
    echo "Submitted $(basename "$part")  |  $(date +'%F %T')"

done

echo "All chunks submitted in serial-queue mode. Driver exiting."
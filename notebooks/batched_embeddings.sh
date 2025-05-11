# file should be in the model folder of scFoundation
#!/usr/bin/env bash
# ──── USER SETTINGS ────────────────────────────────────────────────
INPUT=/work/scratch/ndickenmann/dataset_3_preprocessed.csv
TEMP_CHUNKS=/work/scratch/ndickenmann/tmp_chunks            # will be created
TEMP_EMBS=/work/scratch/ndickenmann/tmp_embeddings          # will be created
FINAL_OUT=/work/scratch/ndickenmann/Embeddings_3_2.npy
LINES_PER_CHUNK=30000
# ───────────────────────────────────────────────────────────────────

set -euo pipefail

### FIX 0 – start clean
rm -rf "$TEMP_CHUNKS" "$TEMP_EMBS"                # ← clear old stuff
mkdir -p "$TEMP_CHUNKS" "$TEMP_EMBS"
mkdir -p "$(dirname "$FINAL_OUT")"                # ← make final dir

echo "Step 1 / 3  ➜  Splitting CSV into blocks of $LINES_PER_CHUNK rows …"

# Grab header once
header=$(head -n 1 "$INPUT")

# Skip header, split the remainder
tail -n +2 "$INPUT" | split -l "$LINES_PER_CHUNK" - "$TEMP_CHUNKS/chunk_"

# Put the header back on every chunk
for part in "$TEMP_CHUNKS"/chunk_*; do
  printf '%s\n' "$header" | cat - "$part" > "${part}.csv" && mv "${part}.csv" "$part"
done

echo "Step 2 / 3  ➜  Running get_embedding.py on each chunk …"

set +e
i=0
for part in "$TEMP_CHUNKS"/chunk_*; do
  printf '  • chunk %s → GPU\n' "$part"
  idx=$(printf "%03d" "$i")
  python get_embedding.py \
    --task_name Embedding_part${idx} \
    --input_type singlecell \
    --output_type cell \
    --pool_type all \
    --tgthighres a5 \
    --data_path "$part" \
    --save_path "$TEMP_EMBS" \
    --pre_normalized F
     \
    --version rde
  status=$?
  echo "exit‑status for $part ⇒ $status"
  ((i++))
done
set -e 

echo "Step 3 / 3  ➜  Concatenating per‑chunk .npy files …"

python - "$TEMP_EMBS" "$FINAL_OUT" <<'PY'
import glob, os, numpy as np, sys
temp_embs = sys.argv[1]
final_out = sys.argv[2]

# Matches names produced by the script (task_name + default ckpt suffix)
parts = sorted(glob.glob(os.path.join(
    temp_embs, 'Embedding_part*_*_singlecell_cell_embedding_a5_resolution.npy')))

print("   files found:", len(parts))
arrays = [np.load(p) for p in parts]
full = np.concatenate(arrays, axis=0)
np.save(final_out, full)
print("   saved ➜", final_out)
PY

echo "All done!  Concatenated embedding array is at:"
echo "  $FINAL_OUT"

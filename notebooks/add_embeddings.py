import anndata as ad
import h5py, numpy as np, os, re, sys, glob

src  = "/work/scratch/ndickenmann/snlong2025_nbatlas2024_malignant_concat_with_embed.h5ad"
parts_dir = "/work/scratch/ndickenmann/tmp_embeddings_smaller_variable_genes_only"

### 1  Inspect AnnData (read-only)
adata = ad.read_h5ad(src, backed="r")   # no chance of writing yet
print(adata)
n_cells = adata.n_obs
adata.file.close() 
del adata   

### 2  Collect and sort part files
pat = re.compile(r"Embedding_part(\d+)_.*\.npy$")
files = sorted(
    [os.path.join(parts_dir, f) for f in os.listdir(parts_dir) if pat.match(f)],
    key=lambda x: int(pat.search(x).group(1))
)
if not files:
    sys.exit("No part files found!")

### 3  Peek at first part to know shape/dtype
probe = np.load(files[0], mmap_mode="r")
n_feat, dtype = probe.shape[1], probe.dtype
del probe

### 4  Open the COPY in r+ and create dataset
with h5py.File(src, "r+") as f:
    if "obsm/X_scFoundation" in f:
        sys.exit("X_scFoundation already exists in target file – aborting to stay safe")

    dset = f.create_dataset(
        "obsm/X_scFoundation",
        shape=(n_cells, n_feat),
        dtype=dtype,
        chunks=(32_000, n_feat),
        compression="gzip", compression_opts=4,
    )

    cursor = 0
    for i, fn in enumerate(files, 1):
        arr = np.load(fn, mmap_mode="r")
        rows = arr.shape[0]
        dset[cursor:cursor+rows] = arr
        cursor += rows
        print(f"[{i:3}/{len(files)}] wrote rows {cursor:>6}/{n_cells}", end="\r")

    print()  # newline after progress
    if cursor != n_cells:
        raise RuntimeError(f"Wrote {cursor} rows, expected {n_cells}")

print("✓ Finished streaming embeddings into copy")

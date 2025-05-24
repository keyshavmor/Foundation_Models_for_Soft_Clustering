import scanpy as sc
import numpy as np
import pandas as pd
import scanpy.external as sce

SEED        = 42      # reproducibility
KWITHIN     = 3       # neighbours per batch
N_PCS       = 50      # PCs to keep
TRIM_EDGES  = 15      # sparsify the graph

adata_path = "/cluster/home/maurdu/Foundation_models_NB/clustering/resources/embeddings/embedding.h5ad"
adata_save_path = "/cluster/home/maurdu/Foundation_models_NB/clustering/resources/embeddings/embedding_processed.h5ad"
embed_adata = sc.read_h5ad(adata_path)
print("Cleaning Data...")
# ── quick QC filters: keep good cells and reasonably common genes ─────
sc.pp.filter_cells(embed_adata, min_genes=200)
sc.pp.filter_genes(embed_adata, min_cells=3)


print("Starting Z-score normalization")
# ------------------------------------------------------------------
# 0)  Z‑score each dimension of the Cancerfnd embedding (obsm)
# ------------------------------------------------------------------
X = embed_adata.obsm["CancerGPT"]
mean = X.mean(axis=0, keepdims=True)
std  = X.std(axis=0, keepdims=True)
Xz   = (X - mean) / std
Xz   = np.clip(Xz, -10, 10)              # mimic scanpy.pp.scale clipping
embed_adata.obsm["cancerfnd_z"] = Xz

print("Starting PCA on Z-score normalization")
# ------------------------------------------------------------------
# 1)  PCA on that z‑scored matrix      (Scanpy will use scikit‑learn under the hood)
#     We do this in a *temporary* AnnData that has the right shape
# ------------------------------------------------------------------
tmp = sc.AnnData(
    Xz,
    obs=embed_adata.obs.copy(),                          # keep metadata
    var=pd.DataFrame(index=[f"dim_{i}" for i in range(Xz.shape[1])]),
)
sc.tl.pca(tmp, n_comps=N_PCS, svd_solver="arpack", random_state=SEED)

# Copy the PCs back to the main object
embed_adata.obsm["cancerfnd_pca"] = tmp.obsm["X_pca"]

# ------------------------------------------------------------------
# 2)  BBKNN graph on the PCA space
# ------------------------------------------------------------------
### keep only batches with at least KWITHIN neighbours ###
# Mask: True for cells in batches with at least KWITHIN members
print("Keep only batches with atleast KWITHIN members")
batches_sufficient_size_mask = embed_adata.obs["SAMPLES_JOINT"].groupby(embed_adata.obs["SAMPLES_JOINT"]).transform('size') >= KWITHIN
embed_adata = embed_adata[batches_sufficient_size_mask].copy()
print("Applying BBKNN...")
sce.pp.bbknn(
    embed_adata,
    batch_key="SAMPLES_JOINT",
    use_rep="cancerfnd_pca",
    neighbors_within_batch=KWITHIN,
    metric="euclidean", # you can also use the an
    trim=TRIM_EDGES,
)
print("Calculating UMAP...")
# ------------------------------------------------------------------
# 3)  UMAP from that graph
# ------------------------------------------------------------------
sc.tl.umap(
    embed_adata,
    key_added="X_umap_cfnd_bbknn",
)
print("plotting")
# ------------------------------------------------------------------
# 4)  Plot
# ------------------------------------------------------------------
sc.pl.embedding(
    embed_adata,
    basis="X_umap_cfnd_bbknn",
    color=["cell_state", "cell_type_joint", "SAMPLES_JOINT", "Stage_Code"],
    palette=sc.pl.palettes.default_20,
    legend_loc="right margin",
    frameon=False,
    title=f"cancerfnd → z‑score → {N_PCS} PCs → BBKNN(k={KWITHIN})",
)

print("repeating plotting with harmony")
# ------------------------------------------------------------------
# 5) Now repeat with the same PCAs and plot with harmony
# ------------------------------------------------------------------

sc.external.pp.harmony_integrate(
    embed_adata,
    key="SAMPLES_JOINT",
    basis="cancerfnd_pca",
    adjusted_basis="cancerfnd_pca_harmony",
)
print("calculating neighbours")
# neighbours & UMAP
sc.pp.neighbors(embed_adata, use_rep="cancerfnd_pca_harmony", n_neighbors=15, metric="cosine")
sc.tl.umap(embed_adata, random_state=SEED, key_added="X_umap_cfnd_harmony")

sc.pl.embedding(
    embed_adata,
    basis="X_umap_cfnd_harmony",
    color=["cell_state", "SAMPLES_JOINT"],
    legend_loc="right margin",
    frameon=False,
    title="cancerfnd → 50 PCs → Harmony",
)
print("saving embed file:")
print(embed_adata)
embed_adata.write(adata_save_path)
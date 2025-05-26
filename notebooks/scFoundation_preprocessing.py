import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import gc, sys, os

import get_embedding
from get_embedding import main_gene_selection


# --------------------------------------------------
# Load
# --------------------------------------------------
adata = sc.read(
    "/work/scratch/ndickenmann/snlong2025_nbatlas2024_malignant_concat.h5ad",
    #backed="r",
)
print(adata)

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2_000,
    flavor="seurat_v3"
)

adata= adata[:, adata.var['highly_variable']]
print(adata)

# --------------------------------------------------
# Prepare gene list
# --------------------------------------------------
gene_list_df = pd.read_csv("OS_scRNA_gene_index.19264.tsv",header=0, sep="\t")
gene_list = gene_list_df["gene_name"].values

# Boolean mask for the columns we need
gene_mask   = adata.var_names.isin(gene_list)
#sel_genes   = adata.var_names[gene_mask]

print(f"Genes in list: {len(gene_list)}  – overlap with 19264: {gene_mask.sum()}  ")

# --------------------------------------------------
# Streaming write
# --------------------------------------------------
batch_size  = 2_000
out_file    = "/work/scratch/ndickenmann/dataset_2_3_combined_preprocessed_highly_variable_genes_only.csv"


for start in range(194000, adata.n_obs, batch_size):
    end   = min(start + batch_size, adata.n_obs)
    print(f"processing {start}-{end-1}")

    # slice rows **and** selected columns – this keeps the slice on disk until read
    batch = adata[start:end]#, gene_mask]
    X = batch.X.toarray() if sp.issparse(batch.X) else batch.X
    df = pd.DataFrame(X, index=batch.obs_names,
                  columns=batch.var_names)#, columns=sel_genes)
    
    # Process the batch
    X_df_batch, _, _ = main_gene_selection(df, gene_list)

    mode, header = ("w", True) if start == 0 else ("a", False)
    X_df_batch.to_csv(out_file, mode=mode, header=header)

    del df, batch, X, X_df_batch
    gc.collect()


print("Processing complete!")
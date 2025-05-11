# file should be in the model folder of scFoundation

import scanpy as sc
import pandas as pd
import numpy as np

import sys
import os
import gc

import scipy.sparse as sp


# Add the current directory to Python's path
sys.path.append(os.getcwd())

import get_embedding
import load

from get_embedding import main_gene_selection

#adata=sc.read("/work/scratch/ndickenmann/NB.bone.Met_preprocessed.h5ad")
adata=sc.read("/work/scratch/ndickenmann/sn_tumor_cells_NB_hvg.h5ad")
#adata=sc.read("/work/scratch/ndickenmann/NBatlas_cnv_endothelial_ref_percluster2000hvg.h5ad")
# Assuming you have an AnnData object named 'adata'
# Extract the gene expression matrix as a DataFrame
print(adata)


# make sure that it doesn't get reducred too much by the alignment
#sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=4000, layer="RNA")  # CAUTION change to log1p_norm when not using dataset 2
adata_filterd = adata#[:, adata.var['highly_variable']]
#adata_filterd = adata

print(adata_filterd)

rna = adata_filterd.layers["RNA"]
if sp.issparse(rna):
    print(rna)
    rna = rna.toarray() 

print(f"Adata_filtered.obs_names{adata_filterd.obs_names}")
print(f"Adata_filtered.var_names{adata_filterd.var_names}")
print(f"adata_filterd.var['gene_name'].values {adata_filterd.var['gene_name'].values}")

gene_expression_df = pd.DataFrame(
    data=rna,
    index=adata_filterd.obs_names,
    columns=adata_filterd.var['gene_name'].values
)

print("done 2")

# Match genes with scFoundation's gene list
gene_list_df = pd.read_csv('OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])

print("Checks: ")

print(gene_list_df.head())
print("unique genes:", gene_list_df['gene_name'].nunique())

expr_genes = set(adata_filterd.var_names)
master_genes = set(gene_list)
expr_genes_new = set(gene_expression_df.columns)
print("overlap:", len(expr_genes & master_genes))
print("current overlap:", len(expr_genes_new & master_genes))


nonzero_rna   = (adata_filterd.layers["RNA"]   != 0).sum()
nonzero_log1p = (adata_filterd.layers["log1p_norm"] != 0).sum()
print("RNA non-zeros:  ", nonzero_rna)
print("log1p_norm non-zeros:", nonzero_log1p)

print("Checks done")

# Process in smaller batches
batch_size = 2000
total_cells = gene_expression_df.shape[0]
batches = (total_cells + batch_size - 1) // batch_size

# Create output file and write header first
output_file = '/work/scratch/ndickenmann/dataset_2_preprocessed.csv'
first_batch = True

for batch_idx in range(batches):
    print(f"Processing batch {batch_idx+1}/{batches}")
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, total_cells)
    
    # Extract the current batch
    batch_df = gene_expression_df.iloc[start_idx:end_idx]
    
    # Process the batch
    X_df_batch, to_fill_columns, var = main_gene_selection(batch_df, gene_list)

    print(f"X size: {X_df_batch.shape}")
    
    # Write results to file incrementally with mode='a' (append) after first batch
    if first_batch:
        X_df_batch.to_csv(output_file, index=True)
        first_batch = False
    else:
        X_df_batch.to_csv(output_file, mode='a', header=False, index=True)
    
    # Free memory
    del X_df_batch
    del batch_df
    gc.collect()

print("Processing complete!")
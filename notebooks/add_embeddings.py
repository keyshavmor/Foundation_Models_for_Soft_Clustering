import numpy as np
import anndata as ad

# Load the .h5ad file
adata = ad.read_h5ad("/work/scratch/ndickenmann/sn_tumor_cells_NB_hvg.h5ad")


#adata_temp = ad.read_h5ad("/home/nicolas/Documents/ETH-projects/Foundational_models/sn_tumor_cells_NB_hvg.h5ad")

print(adata)
#print(adata_temp)

# Load the .npy file
new_layer = np.load("/work/scratch/ndickenmann/Embeddings_2_2.npy")

print(f"New layer shape: {new_layer.shape}")
print(f"adata.X shape: {adata.X.shape}")

# Embeddings have a different shape than the original data numb_er of cells x number of features
#if new_layer.shape != adata.X.shape:
 #   raise ValueError("Shape of the new layer does not match adata.X shape.")

# Add the new layer
adata.obsm["X_scFoundation"] = new_layer

# Save the updated .h5ad file
adata.write("/work/scratch/ndickenmann/scfoundation_dataset2.h5ad")

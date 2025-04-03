import scanpy as sc
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scimilarity.utils import lognorm_counts, align_dataset
from scimilarity import CellEmbedding
import os

# Define the model directory
model_dir = "/work/scratch/ndickenmann"
model_path = f"{model_dir}/model_v1.1"

# Load data
data_path = "/work/scratch/ndickenmann/NB_bone_Met_preprocessed.h5ad"
data = sc.read(data_path)

print(f"Data{data}")
print(f"Data Shape{data.X.shape}")

# Initialize embedding mode
ce = CellEmbedding(
    model_path,
    use_gpu=True,
)

# Process in batches
batch_size = 1000  # Adjust based on your memory constraints
n_cells = data.X.shape[0]
n_batches = int(np.ceil(n_cells / batch_size))

print(f"Min value in data: {data.X.min()}")

print(f"Processing {n_cells} cells in {n_batches} batches of size {batch_size}")

# Initialize an array to store all embeddings
all_embeddings = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, n_cells)
    
    print(f"Processing batch {i+1}/{n_batches} (cells {start_idx} to {end_idx})")
    
    # Extract batch data
    batch_indices = list(range(start_idx, end_idx))
    batch_data = data[batch_indices].copy()
    
    # Convert sparse matrix to dense format directly
    # This handles the sparse matrix issue without using scipy directly
    batch_data.X = data.layers['log1p_norm'][batch_indices].toarray()
    
    # Now we can safely use np.maximum
    batch_data.X = np.maximum(batch_data.X, 0)
    
    # Align the batch
    batch_data = align_dataset(batch_data, ce.gene_order)
    
    # Skip normalization since data is already normalized
    # batch_data = lognorm_counts(batch_data)
    
    # Get embeddings for this batch
    batch_embeddings = ce.get_embeddings(batch_data.X)
    
    # Store the batch embeddings
    all_embeddings.append(batch_embeddings)


# After your batch processing loop, add this line to concatenate all batches:
all_embeddings = np.concatenate(all_embeddings, axis=0)

print(f"All embeddings computed. Shape: {all_embeddings.shape}")

# Optional: Save embeddings to file
np.save(f"/home/ndickenmann/Foundation_models_NB/embeddings/embeddings.npy", all_embeddings)
print("Embeddings saved to file")


embeddings = ce.get_embeddings(data.X)
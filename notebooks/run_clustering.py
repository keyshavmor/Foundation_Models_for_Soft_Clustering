# %%
import sys
sys.path.insert(0, "../")
import scanpy as sc
from model.embedding import embed
import numpy as np


# # --- Configuration ---
# # Choose the ground truth label column from embed_adata.obs
# # 'cell1' (coarser) or 'cell2' (finer) are good candidates based on the data exploration cell
# ground_truth_key = 'cell2'

# # %%
# model_dir = "../assets/"
# adata_path = "../data/NB.bone.Met_preprocessed.h5ad" # INSERT the path to your anndata object here

# # 1. Load the full AnnData object
# print(f"Loading full AnnData object from: {adata_path}")
# adata = sc.read_h5ad(adata_path)
# print(f"Successfully loaded. Original shape: {adata.shape}")
# print(adata)
# # --- Subsampling Step ---
# fraction_to_keep = 1.0
# # For reproducibility of the random sampling
# random_seed = 42
# np.random.seed(random_seed)


# # %%
# ### Check the NOR and ADR signatures in the tumor cells ###

# # 1) Subset the AnnData to only include Tumor cells
# tumor_adata = adata[adata.obs["cell2"] == "Tumor"].copy()

# # 2) Plot the expression of your genes of interest (e.g. "GeneA", "GeneB")
# sc.pl.umap(tumor_adata, color=["CD44", "PHOX2B"])

# # 3) Plot the expression of your genes of interest (e.g. "GeneA", "GeneB")

# ## PHOX2B ADR population
# ## CD44 MES
# ## TNFRSF1A AND EGFR Bridge to MES

# sc.pl.umap(tumor_adata, color=["CD44", "PHOX2B", "TNFRSF1A", "EGFR"])

# ## NOR genes from https://www.nature.com/articles/s41467-023-38239-5/figures/4
# sc.pl.umap(tumor_adata, color=["TFAP2B", "PHOX2B", "PHOX2A", "GATA2", "GATA3"])


# ## MES genes from https://www.nature.com/articles/s41467-023-38239-5/figures/4

# sc.pl.umap(tumor_adata, color=["EGR3", "SMAD3", "RUNX2", "CD44", "MEOX2"])

# %%
import os
import scanpy as sc

# --- Configuration ---
output_filename = "NB.bone.Met_embedded.h5ad"
output_subdir = "data"
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
output_embedded_path = os.path.join(script_dir, "..", output_subdir, output_filename)
batch_key = "sample"
batch_size = 64

# --- Check for Existing File or Perform Embedding ---
if os.path.exists(output_embedded_path):
    print(f"Loading existing embedded AnnData: {output_embedded_path}")
    embed_adata = sc.read_h5ad(output_embedded_path)
    print("Load complete.")
else:
    print(f"Embedded file not found: {output_embedded_path}")
    print("Performing embedding...")
    if 'adata' not in locals():
         raise NameError("Variable 'adata' is not defined.")

    embed_adata = embed(
        adata_or_file=adata,
        model_dir=model_dir,
        batch_key=batch_key,
        batch_size=batch_size,
    )
    print("Embedding complete.")

    # Save the newly embedded object
    print(f"\nSaving embedded AnnData object to: {output_embedded_path}")
    output_dir = os.path.dirname(output_embedded_path)
    if output_dir: # Ensure output_dir is not empty (e.g., saving in current dir)
        os.makedirs(output_dir, exist_ok=True)

    embed_adata.write(output_embedded_path, compression='gzip')
    print("Save complete.")

# --- Final Output ---
if embed_adata is not None:
    print("\n--- Final Embedded AnnData Object Info ---")
    print(embed_adata)
else:
    # This path is less likely without try-except but could occur if 'adata' was missing initially
    print("\nError: embed_adata object could not be loaded or created.")

# %%
sc.pp.neighbors(embed_adata, use_rep="CancerGPT")
sc.tl.umap(embed_adata)
sc.tl.leiden(adata, flavor='igraph', n_iterations=2)
for feature in ["cell1", "cell2", "fraction", "sample"]:
    sc.pl.umap(adata, color=feature, show=False, save=f"_harmony_{feature}.png")
    sc.pl.umap(embed_adata, color=feature, show=False, save=f"_embed_{feature}.png")

# %%
# --- GMM Soft Clustering and Entropy Calculation ---
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score 
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc



# Define necessary variables (adjust as needed)
random_seed = 42
embedding_key = 'X_umap' # The embedding key in embed_adata.obsm for visualization
ground_truth_key = 'ground_truth' # Key in embed_adata.obs for ground truth labels, if available
output_fig_dpi = 150 # DPI for saving figures if uncommented

# Extract embeddings
embeddings = embed_adata.obsm['CancerGPT'] # Use your actual embedding key
print(f"Using embeddings from embed_adata.obsm['CancerGPT'] with shape: {embeddings.shape}")

# --- BIC/AIC Calculation for Component Selection ---
print("\n--- Calculating BIC/AIC for GMM component selection ---")
# Define the range of components to test

n_components_range = range(2, 21) # Test from i to j components
bic_scores = []
aic_scores = []
silhouette_scores = []
# Ensure covariance_type is defined before the loop
covariance_type = 'tied' # 'full', 'tied', 'diag', 'spherical'

for n_components in n_components_range:
    print(f"Fitting GMM with {n_components} components...")
    gmm_temp = GaussianMixture(n_components=n_components,
                               n_init = 5,
                               covariance_type=covariance_type,
                               random_state=random_seed)
    gmm_temp.fit(embeddings)
    bic_scores.append(gmm_temp.bic(embeddings))
    aic_scores.append(gmm_temp.aic(embeddings))
    silhouette_scores.append(silhouette_score(embeddings, gmm_temp.predict(embeddings), metric='euclidean'))
print("Finished calculating BIC/AIC scores.")

# --- Visualize BIC/AIC Scores ---
print("Plotting BIC/AIC scores...")
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, label='BIC', color='blue', marker='o')
plt.plot(n_components_range, aic_scores, label='AIC', color='red', marker='x')
plt.plot(n_components_range, silhouette_scores, label='Silhoutte Scores', color='purple', marker='v')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion Value')
plt.title(f'GMM BIC and AIC ({covariance_type} covariance)')
plt.xticks(n_components_range) # Ensure all component numbers are shown as ticks if range is small
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
# Highlight the minimum BIC
min_bic_index = np.argmin(bic_scores)
min_bic_components = n_components_range[min_bic_index]
plt.axvline(min_bic_components, color='green', linestyle=':',
            label=f'Min BIC at {min_bic_components} components')
plt.legend() # Show legend again to include the vertical line label
# plt.savefig(f'gmm_bic_aic_selection_{covariance_type}.png', dpi=output_fig_dpi, bbox_inches='tight')
plt.show()

# --- GMM Fitting with Chosen Number of Components ---
# Choose n_clusters based on the BIC/AIC plot above !!!
# minimum BIC good default value
n_clusters = min_bic_components # set manually if no num cluster component detection is desired, e.g. n_clusters = 10

print(f"\n--- Proceeding with GMM fitting using {n_clusters} components ---")
print(f"Fitting final GMM with {n_clusters} components...")
gmm = GaussianMixture(n_components=n_clusters,
                      random_state=random_seed,
                      covariance_type=covariance_type)
gmm.fit(embeddings)


gmm_probabilities = gmm.predict_proba(embeddings)
print("Calculating entropy...")
# Entropy H(p) = - sum(p_i * log2(p_i))
# small epsilon to avoid log(0)
epsilon = 1e-9
cell_entropy = entropy(gmm_probabilities.T + epsilon, base=2)

# 6. Store results in AnnData object
embed_adata.obsm['GMM_probabilities'] = gmm_probabilities
print("Added 'GMM_probabilities' to embed_adata.obsm")
embed_adata.obs['GMM_entropy'] = cell_entropy
print("Added 'GMM_entropy' to embed_adata.obs")

# --- Add Dominant Cluster Assignment ---
# Find the cluster index with the highest probability for each cell
dominant_cluster = np.argmax(gmm_probabilities, axis=1)
# Store as categorical strings for better plotting with scanpy
embed_adata.obs['GMM_cluster'] = [f'GMM_{c}' for c in dominant_cluster]
embed_adata.obs['GMM_cluster'] = embed_adata.obs['GMM_cluster'].astype('category')
print("Added 'GMM_cluster' (dominant cluster) to embed_adata.obs")

print("\nDone with GMM and entropy calculation.")

# --- Visualization ---
print(f"\n--- Starting Visualization (using embedding: '{embedding_key}') ---")

# Configure plotting settings
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)
sns.set_style("whitegrid") # Or "white", "ticks", etc.

# --- Titles ---
title_cluster = f'GMM Dominant Cluster ({n_clusters} components)'
title_entropy = 'GMM Assignment Entropy'
title_y_position = 1.05 # Adjust this value to move title further up (e.g., 1.05) or down (e.g., 1.01)

# 1. Visualize Dominant GMM Clusters
print(f"Plotting dominant GMM clusters on {embedding_key}...")
fig1, ax1 = plt.subplots()
sc.pl.embedding(embed_adata,
                basis=embedding_key,
                color='GMM_cluster',
                legend_loc='on data', # Changed for potentially better placement with many clusters
                legend_fontsize=8,   # Adjusted font size
                ax=ax1,
                show=False,
                title="") # Clear scanpy title
# Set title manually using matplotlib with adjusted vertical position
ax1.set_title(title_cluster, y=title_y_position)
# plt.savefig(f'gmm_clusters_{embedding_key}.png', dpi=output_fig_dpi, bbox_inches='tight')
plt.show()

# 2. Visualize GMM Assignment Entropy
print(f"Plotting GMM entropy on {embedding_key}...")
fig2, ax2 = plt.subplots()
sc.pl.embedding(embed_adata,
                basis=embedding_key,
                color='GMM_entropy',
                cmap='viridis',
                colorbar_loc='right',
                ax=ax2,
                show=False,
                title="") # Clear scanpy title
# Set title manually using matplotlib with adjusted vertical position
ax2.set_title(title_entropy, y=title_y_position)
# plt.savefig(f'gmm_entropy_{embedding_key}.png', dpi=output_fig_dpi, bbox_inches='tight')
plt.show()

# 3. Visualize Both Side-by-Side
print(f"Plotting clusters and entropy side-by-side on {embedding_key}...")
fig3, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot Clusters on the left
sc.pl.embedding(embed_adata,
                basis=embedding_key,
                color='GMM_cluster',
                title="", # Remove title from scanpy call
                legend_loc='on data',
                legend_fontsize=8,   # Adjusted font size
                ax=axes[0],
                show=False)
axes[0].set_title(title_cluster, y=title_y_position) # Set title manually
axes[0].set_xlabel(f"{embedding_key.split('_')[-1].upper()} 1")
axes[0].set_ylabel(f"{embedding_key.split('_')[-1].upper()} 2")

# Plot Entropy on the right
sc.pl.embedding(embed_adata,
                basis=embedding_key,
                color='GMM_entropy',
                title="", # Remove title from scanpy call
                cmap='viridis',
                colorbar_loc='right',
                ax=axes[1],
                show=False)
axes[1].set_title(title_entropy, y=title_y_position) # Set title manually
axes[1].set_xlabel(f"{embedding_key.split('_')[-1].upper()} 1")
axes[1].set_ylabel(f"{embedding_key.split('_')[-1].upper()} 2")

# Use tight_layout *after* setting titles
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjusted rect top value slightly
fig3.suptitle("GMM Analysis", fontsize=14, y=0.99)

# plt.savefig(f'gmm_clusters_entropy_{embedding_key}_combined.png', dpi=output_fig_dpi, bbox_inches='tight')
plt.show()

# 4. Visualize Ground Truth (if available) on a new figure
if ground_truth_key in embed_adata.obs:
    print(f"Plotting ground truth ('{ground_truth_key}') on {embedding_key}...")
    fig4, ax4 = plt.subplots() # Create a new figure and axes
    sc.pl.embedding(embed_adata,
                    basis=embedding_key,
                    color=ground_truth_key,
                    legend_loc='on data', # Changed for potentially better placement
                    legend_fontsize=8,   # Adjusted font size
                    ax=ax4,
                    show=False,
                    title=f'Ground Truth ({ground_truth_key})') # Add title via scanpy
    # plt.savefig(f'ground_truth_{embedding_key}.png', dpi=output_fig_dpi, bbox_inches='tight')
    plt.show()
else:
    print(f"Ground truth key '{ground_truth_key}' not found in embed_adata.obs. Skipping ground truth plot.")

print("\n--- Analysis Complete ---")

# %%
# --- 3a: K-Means Clustering ---
# Decide on the number of clusters for k-means.
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN

n_clusters_kmeans = n_clusters 
print(f"Running K-Means with {n_clusters_kmeans} clusters on embeddings...")
kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=random_seed, n_init=10)
kmeans_labels = kmeans.fit_predict(embeddings)
embed_adata.obs['kmeans'] = pd.Categorical([f'KMeans_{c}' for c in kmeans_labels])
print(f"Stored K-Means results in embed_adata.obs['kmeans']")


# --- 3b: Louvain Clustering ---
# Louvain uses the neighbors graph computed earlier
# Resolution parameter influences the number of clusters found. Tune as needed.
print("Running Louvain clustering...")
sc.tl.louvain(embed_adata, random_state=random_seed, key_added='louvain') # Default resolution=1.0
print(f"Stored Louvain results in embed_adata.obs['louvain']")
# Visualize number of clusters found
print(f"Found {len(embed_adata.obs['louvain'].cat.categories)} Louvain clusters.")

# --- 3c: Leiden Clustering ---
# Leiden also uses the neighbors graph and is generally preferred over Louvain
print("Running Leiden clustering...")
sc.tl.leiden(embed_adata, random_state=random_seed, key_added='leiden') # Default resolution=1.0
print(f"Stored Leiden results in embed_adata.obs['leiden']")
# Visualize number of clusters found
print(f"Found {len(embed_adata.obs['leiden'].cat.categories)} Leiden clusters.")


# --- 3d: HDBSCAN Clustering ---
print("Running HDBSCAN...")
min_cluster_size_hdbscan = len(embed_adata) // 1000
print(f"Using min_cluster_size={min_cluster_size_hdbscan} for HDBSCAN.")
Hdbscan_cluster = HDBSCAN(min_cluster_size=min_cluster_size_hdbscan)
hdbscan_labels = Hdbscan_cluster.fit_predict(embeddings)
# Noise points are labelled -1 by HDBSCAN.
embed_adata.obs['HDBSCAN'] = pd.Categorical([f'HDBSCAN_{c}' for c in hdbscan_labels])
unique_labels = set(hdbscan_labels)
num_clusters_hdbscan = len(unique_labels) - (1 if -1 in unique_labels else 0)
num_noise_points = list(hdbscan_labels).count(-1) # Or np.sum(hdbscan_labels == -1)
print(f"HDBSCAN found {num_clusters_hdbscan} actual clusters and {num_noise_points} points were labeled as noise (-1).")
print("Stored HDBSCAN results in embed_adata.obs['HDBSCAN'].")

# --- 3e: OPTICS Clustering ---
from sklearn.cluster import OPTICS
print("Running OPTICS...")
# using HDBSCAN's min_cluster_size
optics_min_samples = min_cluster_size_hdbscan
print(f"Using min_samples={optics_min_samples} for OPTICS.")
optics = OPTICS(min_samples=optics_min_samples)
optics_labels = optics.fit_predict(embeddings)
embed_adata.obs['OPTICS'] = pd.Categorical([f'OPTICS_{c}' for c in optics_labels])
num_clusters_optics = len(set(optics_labels)) - (1 if -1 in optics_labels else 0)
print(f"OPTICS found {num_clusters_optics} clusters (excluding noise -1).")
print("Stored OPTICS results in embed_adata.obs['OPTICS'].")


# --- 3f: MeanShift Clustering ---
from sklearn.cluster import MeanShift
print("Running MeanShift...")
# MeanShift bandwidth estimation is computationally intensive.
meanshift = MeanShift(n_jobs=-1) # Use n_jobs=-1 for potential speedup for parallel computation
meanshift_labels = meanshift.fit_predict(embeddings)
embed_adata.obs['Mean_Shift'] = pd.Categorical([f'MeanShift_{c}' for c in meanshift_labels])
num_clusters_meanshift = len(set(meanshift_labels))
print(f"MeanShift found {num_clusters_meanshift} clusters.")
print("Stored MeanShift results in embed_adata.obs['Mean_Shift'].")

    # Save the newly cluster results
output_directory = './cluster_res'
output_filename = 'clustered_adata.h5ad'
local_path = os.path.join(output_directory, output_filename)
print(f"\nSaving embedded AnnData object to: {local_path}")
output_dir = os.path.dirname(output_directory)
if output_dir: # Ensure output_dir is not empty (e.g., saving in current dir)
    os.makedirs(output_directory, exist_ok=True)

embed_adata.write(local_path, compression='gzip')
print("Save complete.")


# --- Configuration for clustering---
# Choose the ground truth label column from embed_adata.obs
# 'cell1' (coarser) or 'cell2' (finer) are good candidates based on the data exploration cell
ground_truth_key = 'cell2'

# Choose the clustering results columns from embed_adata.obs to evaluate
# The notebook previously added: 'GMM_cluster', 'kmeans', 'louvain', 'leiden'
clustering_keys = ['GMM_cluster', 'kmeans', 'louvain', 'leiden','HDBSCAN','OPTICS','Mean_Shift']


# %%
# --- Evaluation Against Known Cell-Identity Labels ---
print("\n" + "="*50)
print("--- Starting Evaluation Against Known Cell Labels ---")
print("="*50 + "\n")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    confusion_matrix
)


# --- Check if keys exist ---
if ground_truth_key not in embed_adata.obs.columns:
    raise KeyError(f"Ground truth key '{ground_truth_key}' not found in embed_adata.obs. Available keys: {list(embed_adata.obs.columns)}")

valid_clustering_keys = []
for key in clustering_keys:
    if key in embed_adata.obs.columns:
        valid_clustering_keys.append(key)
    else:
        print(f"Warning: Clustering key '{key}' not found in embed_adata.obs. Skipping evaluation for this key.")
clustering_keys = valid_clustering_keys

if not clustering_keys:
    print("No valid clustering keys found to evaluate. Exiting evaluation section.")
else:

    true_labels = embed_adata.obs[ground_truth_key]

    # --- 1. Calculate and Print Metrics ---
    print(f"\n--- 1. Clustering Evaluation Metrics (vs '{ground_truth_key}') ---")
    metrics_results = {}
    for key in clustering_keys:
        print(f"\nEvaluating: {key}")
        predicted_labels = embed_adata.obs[key]

        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        homogeneity = homogeneity_score(true_labels, predicted_labels)
        completeness = completeness_score(true_labels, predicted_labels)
        v_measure = v_measure_score(true_labels, predicted_labels)

        metrics_results[key] = {
            'ARI': ari,
            'NMI': nmi,
            'Homogeneity': homogeneity,
            'Completeness': completeness,
            'V-Measure': v_measure
        }
        print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
        print(f"  Homogeneity: {homogeneity:.4f}")
        print(f"  Completeness: {completeness:.4f}")
        print(f"  V-Measure: {v_measure:.4f}")

    # Display metrics as a DataFrame
    metrics_df = pd.DataFrame(metrics_results).T
    print("\n--- Metrics Summary ---")
    print(metrics_df)
    print("-" * 25)


    # --- 2. Generate Confusion Matrices ---
    print(f"\n--- 2. Confusion Matrices (True Labels vs Predicted Clusters) ---")
    # Ensure true labels are categorical for proper ordering if they aren't already
    if not isinstance(true_labels, pd.CategoricalDtype):
        true_labels = true_labels.astype('category')
    true_label_names = true_labels.cat.categories

    for key in clustering_keys:
        print(f"\nGenerating Confusion Matrix for: {key}")
        predicted_labels = embed_adata.obs[key]

        if not isinstance(predicted_labels.dtype, pd.CategoricalDtype):
            predicted_labels = predicted_labels.astype('category')
        predicted_label_names = predicted_labels.cat.categories

        # Create confusion matrix dataframe using crosstab for correct labeling
        cm_df = pd.crosstab(true_labels, predicted_labels, rownames=[f'True ({ground_truth_key})'], colnames=[f'Predicted ({key})'])
        # Reindex to ensure all true labels are present and in consistent order, recheck this
        cm_df = cm_df.reindex(index=true_label_names, columns=predicted_label_names, fill_value=0)

        # Determine figure size dynamically
        fig_width = max(8, len(predicted_label_names) * 0.6)
        fig_height = max(6, len(true_label_names) * 0.4)
        annotation_size = max(6, 14 - max(len(predicted_label_names), len(true_label_names)) // 2) # Dynamically adjust annotation size

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d", # Format as integer
            cmap="Blues",
            linewidths=.5,
            annot_kws={"size": annotation_size} # Adjust annotation font size here
        )
        plt.title(f'Confusion Matrix: {key} vs {ground_truth_key}')
        # Labels are now set by crosstab rownames/colnames
        # plt.ylabel(f'True Labels ({ground_truth_key})') # No longer needed
        # plt.xlabel(f'Predicted Clusters ({key})') # No longer needed
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
    plt.show()


    # --- 3. Generate Stacked Bar Plots for Cluster Composition ---
    print(f"\n--- 3. Cluster Composition Bar Plots ---")
    # Using a qualitative colormap suitable for many categories
    try:
        # Try tab20 which has 20 distinct colors, fallback if fewer categories
        colors = plt.get_cmap('tab20').colors
    except ValueError:
        colors = plt.get_cmap('viridis').colors # Fallback

    for key in clustering_keys:
        print(f"\nGenerating Composition Plot for: {key}")
        # Create a cross-tabulation: counts of each true label within each predicted cluster
        ct = pd.crosstab(embed_adata.obs[key], embed_adata.obs[ground_truth_key])

        # Normalize by cluster (column sum = 1) to show percentage composition
        ct_norm = ct.apply(lambda x: x / x.sum(), axis=1)

        # Plotting
        ax = ct_norm.plot(kind='bar', stacked=True, figsize=(12, 7),
                          color=colors[:len(true_label_names)]) # Use subset of colors
        plt.title(f'Composition of {key} Clusters by {ground_truth_key} Labels')
        plt.xlabel(f'Predicted Cluster ({key})')
        plt.ylabel('Proportion of Cells')
        plt.xticks(rotation=45, ha='right')
        # Place legend outside the plot
        plt.legend(title=ground_truth_key, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend
        # plt.savefig(f'composition_plot_{key}.png', dpi=150, bbox_inches='tight')
        plt.show()


    # --- 4. Examine GMM Entropy Distribution per Known Label ---
    if 'GMM_entropy' in embed_adata.obs.columns and 'GMM_cluster' in clustering_keys:
        print(f"\n--- 4. GMM Entropy Distribution per '{ground_truth_key}' Label ---")

        # Create a DataFrame for easier plotting with seaborn
        entropy_df = embed_adata.obs[[ground_truth_key, 'GMM_entropy']].copy()

        # Calculate median entropy for sorting categories
        median_entropy = entropy_df.groupby(ground_truth_key)['GMM_entropy'].median().sort_values()
        ordered_labels = median_entropy.index

        plt.figure(figsize=(8, max(5, len(ordered_labels) * 0.3))) # Adjust height based on number of labels
        sns.boxplot(data=entropy_df, y=ground_truth_key, x='GMM_entropy', order=ordered_labels, palette='viridis')
        plt.title(f'GMM Assignment Entropy Distribution by {ground_truth_key}')
        plt.xlabel('GMM Entropy (bits)')
        plt.ylabel(f'True Label ({ground_truth_key})')
        plt.tight_layout()
        # plt.savefig(f'gmm_entropy_by_{ground_truth_key}.png', dpi=150, bbox_inches='tight') 
        plt.show()

        print("\nInterpretation Guide (?):")
        print(f"- Higher entropy for a cell indicates it has similar probabilities of belonging to multiple GMM clusters (more uncertainty/ambiguity).")
        print(f"- Lower entropy indicates the cell is strongly assigned to one GMM cluster.")
        print(f"- Examining entropy distribution within known '{ground_truth_key}' groups can reveal:")
        print(f"  - Groups with consistently low entropy: Likely well-separated and confidently clustered by GMM.")
        print(f"  - Groups with consistently high entropy: May represent intermediate states, poorly separated populations, or areas where GMM struggles.")
        print(f"  - Groups with wide entropy variance: Might contain subpopulations with different clustering certainties.")
    elif 'GMM_entropy' not in embed_adata.obs.columns:
         print(f"\n--- Skipping GMM Entropy analysis: 'GMM_entropy' not found in embed_adata.obs ---")
    else: # GMM_entropy exists but GMM_cluster wasn't evaluated
         print(f"\n--- Skipping GMM Entropy analysis: 'GMM_cluster' was not included in evaluated clustering_keys ---")


print("\n" + "="*50)
print("--- Evaluation Complete ---")
print("="*50 + "\n")

# %% [markdown]
# Zero-shot Integration copied and applied to the following embedding data.

# %%
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
os.makedirs("./bench_figures", exist_ok=True)
for rep in ['CancerGPT', 'X_pca', 'X_umap']:
    sc.pp.neighbors(embed_adata, use_rep=rep)
    sc.tl.umap(embed_adata)
    fig = sc.pl.umap(embed_adata, 
            color=["cell2", "sample"], 
            frameon=False, 
            palette=sc.pl.palettes.default_20,
            legend_loc=None,
            return_fig=True)
    fig.savefig(f"figures/{rep}_umap.png", dpi=300, bbox_inches="tight")



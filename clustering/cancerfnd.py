# %%
import sys
import os
# DEBUGGING:

print(f"--- PYTHON SCRIPT cancerfnd.py STARTED --- CWD: {os.getcwd()}", flush=True)
print(f"--- PYTHON SCRIPT cancerfnd.py --- Python Executable: {sys.executable}", flush=True)
print(f"--- PYTHON SCRIPT cancerfnd.py --- sys.path: {sys.path}", flush=True)




import yaml
import warnings
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score, homogeneity_score,
                             completeness_score, v_measure_score)
from scipy.stats import entropy, pearsonr
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
# Conditional import for pathway analysis
try:
    import gseapy
except ImportError:
    gseapy = None
    warnings.warn("gseapy not found. Pathway enrichment analysis will be skipped.")

# Append model directory to path if necessary (adjust based on your project structure)
# Assuming the 'model' package is in the parent directory relative to the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # If running as script
#sys.path.insert(0, "../") # If running interactively relative to project root
from model.embedding import embed

# %%
# --- Configuration Loading ---
print("--- Loading Configuration ---")
import argparse # Import the argparse library

parser = argparse.ArgumentParser(
    description="The embedding and clustering script that uses a YAML configuration file. You can specify a custom config file path."
)

# 2. Add an argument for the config file path
#    - '--config' is the long-form argument (e.g., --config my_config.yaml)
#    - '-c' is a short-form alias (e.g., -c my_config.yaml)
#    - `default` specifies the value to use if the argument isn't provided
#    - `help` provides a description for when the user runs the script with -h or --help
parser.add_argument(
    '--config',  # The name of the command-line option
    '-c',             # A short alias for the option
    type=str,         # The type of the argument (a string in this case)
    default='config_sn_tumor.yaml', # Default value if not specified
    help="Path to the YAML configuration file (e.g., 'config_alternative.yaml')."
)
args = parser.parse_args()
config_path = args.config
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
print(f"Loaded configuration for: {config.get('project_name', 'Unnamed Project')}")

# --- Setup Paths and Settings ---
print("--- Setting up Paths and Scanpy ---")
paths_cfg = config['paths']
output_cfg = config['output_paths']
viz_cfg = config['visualization']
data_keys_cfg = config['data_keys']
embed_cfg = config['embedding']
pp_cfg = config['preprocessing']
cluster_cfg = config['clustering']
eval_cfg = config['evaluation']
interp_cfg = config['interpretability']
bench_cfg = config['benchmarking']
RANDOM_SEED = config['random_seed']

# Create output directories
os.makedirs(output_cfg['adata_embedded_dir'], exist_ok=True)
os.makedirs(output_cfg['adata_clustered_dir'], exist_ok=True)
os.makedirs(output_cfg['scanpy_plots_output'], exist_ok=True)
os.makedirs(output_cfg['autosave_plots_output'], exist_ok=True)
os.makedirs(output_cfg['evaluation_plots_output'], exist_ok=True)
os.makedirs(output_cfg['interpretability_output_dir'], exist_ok=True)
os.makedirs(output_cfg['benchmarking_output_dir'], exist_ok=True)

# Scanpy settings
sc.settings.figdir = output_cfg['scanpy_plots_output']
sc.settings.autosave = True # Use autosave via monkey patch
sc.settings.autoshow = False # Prevent attempts to show inline
sc.settings.set_figure_params(dpi=viz_cfg['figure_dpi'], facecolor='white', frameon=False)
sns.set_style("whitegrid")
print(f"Scanpy figure directory set to: {sc.settings.figdir}")

# --- Monkey-patch matplotlib's show to save figures ---
_autosave_plot_dir = output_cfg['autosave_plots_output']
_autosave_plot_counter = 0

def _save_instead_of_show(*args, **kwargs):
    global _autosave_plot_counter, _autosave_plot_dir
    if _autosave_plot_counter == 0:
        try:
            os.makedirs(_autosave_plot_dir, exist_ok=True)
            print(f"--- Plot Autosave: Saving plots to directory '{_autosave_plot_dir}' ---")
        except OSError as e:
            warnings.warn(f"Could not create plot autosave directory '{_autosave_plot_dir}': {e}")
            _autosave_plot_dir = "." # Fallback to current dir
    filename = os.path.join(_autosave_plot_dir, f"autosaved_plot_{_autosave_plot_counter}.png")
    try:
        fig = plt.gcf()
        # Only save if the figure has axes, preventing empty saves
        if not fig.get_axes():
            print(f"Redirecting plt.show(): Skipping figure {_autosave_plot_counter} (no axes).")
            # plt.close(fig) # Close even if empty? Maybe not needed.
            return
        print(f"Redirecting plt.show(): Saving figure {_autosave_plot_counter} to {filename}")
        plt.savefig(filename, bbox_inches='tight', dpi=viz_cfg['figure_dpi'])
        _autosave_plot_counter += 1
        plt.close(fig) # Close the figure after saving
    except Exception as e:
        warnings.warn(f"Failed to autosave plot {_autosave_plot_counter} to {filename}: {e}")
        try:
            plt.close(plt.gcf()) # Attempt to close problematic figure
        except Exception:
            pass # Ignore if closing fails

plt.show = _save_instead_of_show
print("--- Matplotlib plt.show() patched to save figures instead. ---")

# %%
# --- Load or Generate Embeddings ---
print("--- Loading/Generating CancerFoundation Embeddings ---")
model_embedding_key = data_keys_cfg['embedding_key_model']
output_embedded_path = os.path.join(output_cfg['adata_embedded_dir'], output_cfg['adata_embedded_filename'])
print(f"output_embedded_path is: {output_embedded_path}")
adata = None
if not embed_cfg['force_regenerate'] and os.path.exists(output_embedded_path):
    print(f"Loading existing embedded AnnData: {output_embedded_path}")
    try:
        adata = sc.read_h5ad(output_embedded_path)
        print("Load complete.")
        if model_embedding_key not in adata.obsm_keys():
             warnings.warn(f"Loaded AnnData does not contain expected embedding key '{model_embedding_key}'. Will attempt regeneration.")
             adata = None # Force regeneration
        else:
             print(f"Found embedding key '{model_embedding_key}' in loaded data.")
    except Exception as e:
        warnings.warn(f"Failed to load existing AnnData ({e}). Will attempt regeneration.")
        adata = None

if adata is None:
    print(f"Embedding file not found or regeneration forced: {output_embedded_path}")
    print("Loading source AnnData and performing embedding...")
    if not os.path.exists(paths_cfg['adata_input']):
         raise FileNotFoundError(f"Source AnnData file not found: {paths_cfg['adata_input']}")

    adata_source = sc.read_h5ad(paths_cfg['adata_input'])
    print("Loaded source AnnData from {}".format(paths_cfg['adata_input']))
    print(adata_source)
    # --- Data Sanity Checks ---
    if data_keys_cfg['batch_key'] not in adata_source.obs.columns:
        raise KeyError(f"Batch key '{data_keys_cfg['batch_key']}' not found in source adata.obs.")
    print("Batch key is {}".format(data_keys_cfg['batch_key']))
    if data_keys_cfg['ground_truth_key'] not in adata_source.obs.columns:
        warnings.warn(f"Ground truth key '{data_keys_cfg['ground_truth_key']}' not found in source adata.obs. Evaluation might fail.")
    if data_keys_cfg['timepoint_key'] not in adata_source.obs.columns:
        warnings.warn(f"Timepoint key '{data_keys_cfg['timepoint_key']}' not found in source adata.obs. Interpretability analysis might fail.")
    else:
        # Check timepoint values
        expected_timepoints = {data_keys_cfg['timepoint_pre'], data_keys_cfg['timepoint_post']}
        actual_timepoints = set(adata_source.obs[data_keys_cfg['timepoint_key']].unique())
        if not expected_timepoints.issubset(actual_timepoints):
             warnings.warn(f"Expected timepoints '{expected_timepoints}' not fully found in column '{data_keys_cfg['timepoint_key']}'. Found: '{actual_timepoints}'.")
    ## Setup Embedding ##
    batch_key = data_keys_cfg['batch_key']
    if batch_key not in adata_source.obs.columns:
        raise KeyError(f"Batch key '{batch_key}' not found in source adata.obs.")
        
    ### START: Cleaning data ###
    # ── drop MALAT1 (a super‑abundant gene that often skews analyses) ─────
    if "MALAT1" in adata_source.var_names:
        adata_source = adata_source[:, adata.var_names != "MALAT1"].copy()

    # ── quick QC filters: keep good cells and reasonably common genes ─────
    sc.pp.filter_cells(adata_source, min_genes=200)
    sc.pp.filter_genes(adata_source, min_cells=3)
    ### END: Cleaning data ###

    adata_source.var_names = adata_source.var["gene_name"].values.copy() # make "gene_name" the var index"
    adata_source = adata_source[:, ~adata_source.var.index.isna()].copy() #remove genes with no name
    input_layer = embed_cfg['input_layer']
    if input_layer is not None:
        print("input_layer is {}".format(input_layer))
        adata_source.X = adata_source.layers[input_layer].copy().toarray() # convert sparse to dense
        print("adata_source.X has shape {} and values: {}".format(adata_source.X.shape,adata_source.X))
    #if adata_source.is_view: # loading and overwritting data
    
    adata = adata_source.copy() # Ensure we have a copy, not a view
    


    if adata is None:
        print("Calling embedding function...")
        adata = embed(
            adata_or_file=adata_source, # Pass the loaded AnnData object
            model_dir=paths_cfg['model_dir'],
            batch_key=data_keys_cfg['batch_key'],
            batch_size=embed_cfg['batch_size'],
            # Add any other necessary parameters for your specific 'embed' function
        )
        print("Embedding complete.")
        print(f"Resulting AnnData shape: {adata.shape}")
        print(f"Saving embedded AnnData object to: {output_embedded_path}")
        adata.write(output_embedded_path, compression='gzip')
        print("Save complete.")
        if model_embedding_key not in adata.obsm_keys():
            raise ValueError(f"Embedding function did not add the expected key '{model_embedding_key}' to adata.obsm")


print("\n--- Embedded AnnData Object Info ---")
print(adata)
if model_embedding_key in adata.obsm_keys():
    print(f"Embedding '{model_embedding_key}' shape: {adata.obsm[model_embedding_key].shape}")
else:
    warnings.warn(f"Model embedding key '{model_embedding_key}' unexpectedly missing after loading/generation.")



##### EMBEDDING FINISHED #####
# %%

#Filter for cells that have ground_truth
if pp_cfg['only_cells_with_ground_truth']:
    adata = adata[pd.isna(adata.obs[data_keys_cfg['ground_truth_key']]) == False].copy()
    print(f'adata filtered for cells that have a ground truth value, retaining: {adata.shape}')


# --- Calculate Neighbors and UMAP based on CancerFoundation Embedding ---
print(f"\n--- Calculating Neighbors and UMAP based on '{model_embedding_key}' ---")
if model_embedding_key not in adata.obsm_keys():
     raise KeyError(f"Cannot perform downstream analysis: embedding key '{model_embedding_key}' not found.")

# sc.pp.neighbors(adata, use_rep=model_embedding_key, n_neighbors=pp_cfg['n_neighbors'])
umap_key = viz_cfg['embedding_basis'] # e.g., "X_umap_cancerfnd"
# sc.tl.umap(adata, random_state=RANDOM_SEED, neighbors_key=None) # Use default neighbors calculation
# neighb_within_batch=3
# sc.external.pp.bbknn(adata, batch_key=data_keys_cfg['batch_key'], neighbors_within_batch=neighb_within_batch, metric="euclidean", use_rep=data_keys_cfg["embedding_key_model"])
sc.pl.umap(adata, color=[data_keys_cfg['ground_truth_key'], data_keys_cfg['batch_key'], data_keys_cfg["timepoint_key"]],frameon=False,
                 palette=sc.pl.palettes.default_102,
                 legend_loc="right margin",
                 return_fig=True,
                 title=[f"CancerGPT embedding for project: {config['project_name']}"], 
                ) # Use default neighbors calculation
# Rename the default 'X_umap' to our specific key
if 'X_umap' in adata.obsm_keys() and 'X_umap' != umap_key:
    adata.obsm[umap_key] = adata.obsm['X_umap'] 
    del adata.obsm['X_umap']
    print(f"UMAP calculated and stored in adata.obsm['{umap_key}']")
else:
     warnings.warn("sc.tl.umap did not produce 'X_umap' as expected.")


# --- Optional: Calculate PCA for comparison ---
if "X_pca" in bench_cfg['embedding_keys']:
    print("\n--- Calculating PCA for Benchmarking Comparison ---")
    sc.tl.pca(adata, n_comps=pp_cfg['n_pca_components'], svd_solver='arpack', random_state=RANDOM_SEED)
    print(f"PCA calculated and stored in adata.obsm['X_pca']")

# %%
# --- Clustering Algorithms ---
print("\n" + "="*50)
print("--- Running Clustering Algorithms ---")
print("="*50 + "\n")

embeddings = adata.obsm[model_embedding_key]
cluster_results = {} # To store labels temporarily if needed

# --- Gaussian Mixture Model (GMM) ---
optimal_n_clusters = None
if cluster_cfg['use_gmm']:
    print("--- Running Gaussian Mixture Model (GMM) ---")
    n_components_range = range(cluster_cfg['gmm']['n_components_min'], cluster_cfg['gmm']['n_components_max']+1)
    bic_scores = []
    aic_scores = []
    silhouette_scores_gmm = [] # GMM silhouette
    covariance_type = cluster_cfg['gmm']['covariance_type']
    gmm_n_init = cluster_cfg['gmm']['n_init']

    print("Calculating BIC/AIC for component selection...")
    for n_components in n_components_range:
        print(f"  Fitting GMM with {n_components} components...")
        gmm_temp = GaussianMixture(n_components=n_components, n_init=gmm_n_init,
                                   covariance_type=covariance_type, random_state=RANDOM_SEED)
        gmm_temp.fit(embeddings)
        bic_scores.append(gmm_temp.bic(embeddings))
        aic_scores.append(gmm_temp.aic(embeddings))
        # Calculate silhouette score for this number of components
        try:
            gmm_pred_labels = gmm_temp.predict(embeddings)
            if len(set(gmm_pred_labels)) > 1: # Silhouette needs > 1 cluster
                 score = silhouette_score(embeddings, gmm_pred_labels, metric='euclidean')
                 silhouette_scores_gmm.append(score)
            else:
                 silhouette_scores_gmm.append(np.nan) # Or handle as needed
        except Exception as e:
             print(f"    Warning: Could not calculate silhouette for {n_components} components: {e}")
             silhouette_scores_gmm.append(np.nan)


    # Plot BIC/AIC
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
    plt.plot(n_components_range, aic_scores, label='AIC', marker='x')
    # Plot Silhouette on secondary axis if calculated
    if not all(np.isnan(silhouette_scores_gmm)):
        ax2 = plt.gca().twinx()
        ax2.plot(n_components_range, silhouette_scores_gmm, label='Silhouette (GMM Pred)', color='green', marker='v')
        ax2.set_ylabel('Silhouette Score', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        # Combine legends
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')
    else:
        plt.legend(loc='best')

    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion Value')
    plt.title(f'GMM BIC/AIC ({covariance_type} covariance)')
    plt.xticks(list(n_components_range))
    plt.grid(True, linestyle='--', alpha=0.6)

    # Mark minimum BIC
    min_bic_index = np.argmin(bic_scores)
    optimal_n_clusters = n_components_range[min_bic_index]
    plt.axvline(optimal_n_clusters, color='purple', linestyle=':',
                label=f'Min BIC at {optimal_n_clusters} components')
    # Update legend if BIC line was added
    if not all(np.isnan(silhouette_scores_gmm)):
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Find the existing legend and update it
        leg = ax2.get_legend()
        if leg:
            leg.remove() # Remove the old combined legend
        ax2.legend(lines + lines2, labels + labels2, loc='best') # Recreate combined legend
    else:
         plt.legend(loc='best') # Recreate legend if only BIC/AIC

    print(f"Optimal number of clusters based on BIC: {optimal_n_clusters}")
    plt.show() # Uses patched show

    # Fit final GMM
    print(f"Fitting final GMM with {optimal_n_clusters} components...")
    gmm = GaussianMixture(n_components=optimal_n_clusters, random_state=RANDOM_SEED,
                          n_init=gmm_n_init, covariance_type=covariance_type)
    gmm.fit(embeddings)
    gmm_probabilities = gmm.predict_proba(embeddings)
    epsilon = 1e-9 # Add epsilon for numerical stability in entropy calculation
    cell_entropy = entropy(gmm_probabilities.T + epsilon, base=2)
    dominant_cluster = np.argmax(gmm_probabilities, axis=1)

    # Store results
    adata.obsm['GMM_probabilities'] = gmm_probabilities
    adata.obs['GMM_entropy'] = cell_entropy
    gmm_key = 'gmm_cluster'
    adata.obs[gmm_key] = pd.Categorical([f'GMM_{c}' for c in dominant_cluster])
    print(f"Added 'GMM_probabilities' to obsm, 'GMM_entropy' and '{gmm_key}' to obs.")
    if gmm_key not in cluster_cfg['clustering_keys_to_analyze']:
         cluster_cfg['clustering_keys_to_analyze'].append(gmm_key)

# --- K-Means Clustering ---
if cluster_cfg['use_kmeans']:
    if optimal_n_clusters is None:
        warnings.warn("Cannot run KMeans without an optimal cluster number from GMM (or specify manually). Skipping KMeans.")
    else:
        print(f"\n--- Running K-Means Clustering (k={optimal_n_clusters}) ---")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=RANDOM_SEED, n_init=cluster_cfg['kmeans']['n_init'])
        kmeans_labels = kmeans.fit_predict(embeddings)
        kmeans_key = 'kmeans'
        adata.obs[kmeans_key] = pd.Categorical([f'KMeans_{c}' for c in kmeans_labels])
        print(f"Stored K-Means results in adata.obs['{kmeans_key}']")
        if kmeans_key not in cluster_cfg['clustering_keys_to_analyze']:
             cluster_cfg['clustering_keys_to_analyze'].append(kmeans_key)


# --- Leiden Clustering ---
if cluster_cfg['use_leiden']:
    print("\n--- Running Leiden Clustering ---")
    leiden_key = 'leiden'
    sc.tl.leiden(adata, resolution=cluster_cfg['leiden']['resolution'],
                 random_state=RANDOM_SEED, key_added=leiden_key, neighbors_key=None) # Use default neighbors
    print(f"Stored Leiden results in adata.obs['{leiden_key}']")
    print(f"Found {len(adata.obs[leiden_key].cat.categories)} Leiden clusters.")
    if leiden_key not in cluster_cfg['clustering_keys_to_analyze']:
        cluster_cfg['clustering_keys_to_analyze'].append(leiden_key)

# --- Louvain Clustering ---
if cluster_cfg['use_louvain']:
    print("\n--- Running Louvain Clustering ---")
    louvain_key = 'louvain'
    sc.tl.louvain(adata, resolution=cluster_cfg['louvain']['resolution'],
                  random_state=RANDOM_SEED, key_added=louvain_key, neighbors_key=None) # Use default neighbors
    print(f"Stored Louvain results in adata.obs['{louvain_key}']")
    print(f"Found {len(adata.obs[louvain_key].cat.categories)} Louvain clusters.")
    if louvain_key not in cluster_cfg['clustering_keys_to_analyze']:
        cluster_cfg['clustering_keys_to_analyze'].append(louvain_key)

print("--- Clustering Finished ---")
print("Updated adata object is:")
print(adata)
adata.write_h5ad('midpoint_save.h5ad', compression='gzip')
# %%
# --- Visualization ---
# Necessary imports for this section (assume others like AnnData (adata) are pre-loaded)
import scanpy as sc
import warnings
import os
# import matplotlib # Not strictly needed if only using sc.pl.embedding with save parameter

print("\n" + "="*50)
print("--- Generating Visualizations ---")
print("="*50 + "\n")

# --- Configure Scanpy plot settings for saving ---
# (Assuming viz_cfg, data_keys_cfg, cluster_cfg, and adata are defined above this section)

# Define the output directory for plots.
# You can get this from viz_cfg or set a default.
# e.g., plot_output_dir = viz_cfg.get('plot_output_dir', 'scanpy_plots_output')
# For this example, let's assume it's 'concat/results/saved_plots' to match a potential existing structure
# or a new desired one.
plot_output_dir = viz_cfg.get('plot_output_dir', 'concat/results/my_saved_plots') # Or any other path

# Create the directory if it doesn't exist
if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir)
    print(f"Created plot output directory: {plot_output_dir}")

sc.settings.figdir = plot_output_dir
sc.settings.autoshow = False  # Important: Do not show plots interactively, rely on saving
sc.settings.autosave = False  # Disable global autosave if using explicit save in each plot call
sc.settings.file_format_figs = viz_cfg.get('plot_file_format', 'png') # e.g., 'png', 'pdf', 'svg'

print(f"Scanpy plots will be saved to: {os.path.abspath(sc.settings.figdir)}")
print(f"Scanpy plot file format: {sc.settings.file_format_figs}")
# --- End of Scanpy plot settings configuration ---

umap_key = viz_cfg.get('embedding_basis') # Use .get() for safety

if not umap_key:
    warnings.warn(f"'embedding_basis' not found or is None in viz_cfg. Skipping UMAP plots.")
elif umap_key not in adata.obsm:
    warnings.warn(f"UMAP key '{umap_key}' not found in adata.obsm. Skipping UMAP plots.")
else:
    print(f"Plotting UMAPs based on '{umap_key}' and saving to '{sc.settings.figdir}'...")

    # Helper to generate a save filename component for sc.pl.embedding
    # The 'save' parameter in sc.pl.embedding typically expects a suffix like "_description.png"
    # It will be prepended by sc.settings.figdir and the basis name (e.g., "umap_").
    def get_save_filename_suffix(description_parts):
        # Sanitize parts and join them for the suffix part of the filename
        # Replace problematic characters for filenames
        safe_parts = [str(p).replace(' ', '_').replace('/', '-').replace('.', '_') for p in description_parts if p]
        filename_core = "_".join(safe_parts)
        return f"_{filename_core}.{sc.settings.file_format_figs}"

    # Plot by Ground Truth
    ground_truth_obs_key = data_keys_cfg.get('ground_truth_key')
    if ground_truth_obs_key and ground_truth_obs_key in adata.obs:
        save_suffix = get_save_filename_suffix(["ground_truth", ground_truth_obs_key])
        print(f"Plotting UMAP colored by '{ground_truth_obs_key}', saving with suffix '{save_suffix}'...")
        sc.pl.embedding(adata, basis=umap_key, color=ground_truth_obs_key,
                       legend_loc='on data', legend_fontsize=8, title=f'Ground Truth ({ground_truth_obs_key})',
                       save=save_suffix)
    elif ground_truth_obs_key: # Key was configured but not found in adata.obs
        print(f"Skipping ground truth UMAP plot: key '{ground_truth_obs_key}' not found in adata.obs.")
    else: # Key was not configured
        print("Skipping ground truth UMAP plot: 'ground_truth_key' not configured or is None in data_keys_cfg.")

    # Plot by Batch Key
    batch_key_obs_key = data_keys_cfg.get('batch_key')
    if batch_key_obs_key and batch_key_obs_key in adata.obs:
        save_suffix = get_save_filename_suffix(["batch", batch_key_obs_key])
        print(f"Plotting UMAP colored by '{batch_key_obs_key}', saving with suffix '{save_suffix}'...")
        sc.pl.embedding(adata, basis=umap_key, color=batch_key_obs_key,
                       title=f'Batch Key ({batch_key_obs_key})',
                       save=save_suffix)
    elif batch_key_obs_key:
        print(f"Skipping batch key UMAP plot: key '{batch_key_obs_key}' not found in adata.obs.")
    else:
        print("Skipping batch key UMAP plot: 'batch_key' not configured or is None in data_keys_cfg.")

    # Plot by Timepoint Key
    timepoint_key_obs_key = data_keys_cfg.get('timepoint_key')
    if timepoint_key_obs_key and timepoint_key_obs_key in adata.obs:
        save_suffix = get_save_filename_suffix(["timepoint", timepoint_key_obs_key])
        print(f"Plotting UMAP colored by '{timepoint_key_obs_key}', saving with suffix '{save_suffix}'...")
        sc.pl.embedding(adata, basis=umap_key, color=timepoint_key_obs_key,
                       title=f'Timepoint ({timepoint_key_obs_key})',
                       save=save_suffix)
    elif timepoint_key_obs_key:
        print(f"Skipping timepoint key UMAP plot: key '{timepoint_key_obs_key}' not found in adata.obs.")
    else:
        print("Skipping timepoint key UMAP plot: 'timepoint_key' not configured or is None in data_keys_cfg.")

    # Plot by Generated Clusters
    clustering_keys_to_analyze = cluster_cfg.get('clustering_keys_to_analyze', [])
    if not isinstance(clustering_keys_to_analyze, list): # Basic check for iterability
        warnings.warn(f"'clustering_keys_to_analyze' in cluster_cfg (value: {clustering_keys_to_analyze}) is not a list or is missing. Skipping cluster plots.")
        clustering_keys_to_analyze = []
        
    for cluster_key_name in clustering_keys_to_analyze:
        if cluster_key_name in adata.obs:
            save_suffix = get_save_filename_suffix(["cluster", cluster_key_name])
            print(f"Plotting UMAP colored by '{cluster_key_name}', saving with suffix '{save_suffix}'...")
            sc.pl.embedding(adata, basis=umap_key, color=cluster_key_name,
                           legend_loc='on data', legend_fontsize=8, title=f'Cluster: {cluster_key_name}',
                           save=save_suffix)
        else:
            print(f"Skipping cluster UMAP plot: key '{cluster_key_name}' not found in adata.obs.")

    # Plot GMM Entropy if available
    gmm_entropy_obs_key = 'GMM_entropy' # Fixed key name
    if gmm_entropy_obs_key in adata.obs:
        save_suffix = get_save_filename_suffix([gmm_entropy_obs_key]) # Simpler name for single metric
        print(f"Plotting UMAP colored by GMM Entropy, saving with suffix '{save_suffix}'...")
        sc.pl.embedding(adata, basis=umap_key, color=gmm_entropy_obs_key, cmap='viridis',
                       title='GMM Assignment Entropy',
                       save=save_suffix)
    else:
        # This is not an error, just an optional plot
        print(f"Skipping GMM Entropy plot: '{gmm_entropy_obs_key}' not found in adata.obs.")

print("\n" + "="*50)
print("--- Visualization Generation and Saving Complete ---")
print(f"--- Plots have been saved in: {os.path.abspath(sc.settings.figdir)} ---")
print("="*50 + "\n")

# %%
# --- Evaluation Against Ground Truth ---
print("\n" + "="*50)
print(f"--- Evaluating Clustering vs '{data_keys_cfg['ground_truth_key']}' ---")
print("="*50 + "\n")

ground_truth_key = data_keys_cfg['ground_truth_key']
if ground_truth_key not in adata.obs.columns:
    warnings.warn(f"Ground truth key '{ground_truth_key}' not found. Skipping evaluation.")
else:
    true_labels = adata.obs[ground_truth_key]
    # Ensure categorical for consistent plotting/indexing
    if not pd.api.types.is_categorical_dtype(true_labels):
        true_labels = true_labels.astype('category')
    true_label_names = true_labels.cat.categories

    metrics_results = {}
    valid_clustering_keys_for_eval = []

    # --- Calculate Metrics ---
    print("--- 1. Calculating Evaluation Metrics ---")
    for key in cluster_cfg['clustering_keys_to_analyze']:
        if key in adata.obs.columns:
            valid_clustering_keys_for_eval.append(key)
            print(f"Evaluating: {key}")
            predicted_labels = adata.obs[key]
            ari = adjusted_rand_score(true_labels, predicted_labels)
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            homogeneity = homogeneity_score(true_labels, predicted_labels)
            completeness = completeness_score(true_labels, predicted_labels)
            v_measure = v_measure_score(true_labels, predicted_labels)

            # Silhouette Score (on the embedding used for clustering)
            silhouette = np.nan
            try:
                 # Ensure predicted labels are suitable (e.g., integers or strings)
                 # If categorical, get codes; otherwise, assume suitable format
                 if pd.api.types.is_categorical_dtype(predicted_labels):
                     pred_numeric = predicted_labels.cat.codes
                 else:
                     # Attempt conversion if needed, handle errors gracefully
                     try:
                         pred_numeric = pd.to_numeric(predicted_labels, errors='coerce')
                         if pred_numeric.isnull().any():
                             # If conversion fails or introduces NaNs, try factorize
                             pred_numeric, _ = pd.factorize(predicted_labels)
                     except:
                         pred_numeric, _ = pd.factorize(predicted_labels)

                 # Check for noise labels (e.g., -1 in HDBSCAN/OPTICS) - exclude if present?
                 # Or calculate silhouette only on clustered points. For now, include all.
                 if len(set(pred_numeric)) > 1: # Need at least 2 clusters
                     silhouette = silhouette_score(embeddings, pred_numeric, metric='euclidean')
                 else:
                     print(f"  Skipping Silhouette for {key}: Only one cluster found.")
            except Exception as e:
                 print(f"  Warning: Could not calculate Silhouette score for {key}: {e}")


            metrics_results[key] = {'ARI': ari, 'NMI': nmi, 'Homogeneity': homogeneity,
                                    'Completeness': completeness, 'V-Measure': v_measure,
                                    'Silhouette': silhouette}
            print(f"  ARI: {ari:.4f}, NMI: {nmi:.4f}, Homogeneity: {homogeneity:.4f}, Completeness: {completeness:.4f}, V-Measure: {v_measure:.4f}, Silhouette: {silhouette:.4f}")
        else:
            print(f"Skipping evaluation for {key}: Key not found in adata.obs")

    if metrics_results:
        metrics_df = pd.DataFrame(metrics_results).T
        print("\n--- Metrics Summary ---")
        print(metrics_df.to_markdown(floatfmt=".4f"))
        print("-" * 25)
        metrics_output_path = os.path.join(output_cfg['evaluation_plots_output'], 'clustering_metrics.csv')
        metrics_df.to_csv(metrics_output_path)
        print(f"Metrics saved to {metrics_output_path}")
    else:
        print("No valid clustering keys found to evaluate.")


    # --- Confusion Matrices ---
    print("\n--- 2. Generating Confusion Matrices ---")
    for key in valid_clustering_keys_for_eval:
        print(f"Generating Confusion Matrix for: {key}")
        predicted_labels = adata.obs[key]
        if not pd.api.types.is_categorical_dtype(predicted_labels):
            predicted_labels = predicted_labels.astype('category')
        predicted_label_names = predicted_labels.cat.categories

        cm_df = pd.crosstab(true_labels, predicted_labels, rownames=[f'True ({ground_truth_key})'], colnames=[f'Predicted ({key})'])
        # Ensure all categories are present
        cm_df = cm_df.reindex(index=true_label_names, columns=predicted_label_names, fill_value=0)

        # Dynamic figure size and annotation size
        fig_width = max(8, len(predicted_label_names) * 0.5 + 2)
        fig_height = max(6, len(true_label_names) * 0.4 + 2)
        annotation_size = max(6, 12 - max(len(predicted_label_names), len(true_label_names)) // 3)

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", linewidths=.5, annot_kws={"size": annotation_size})
        plt.title(f'Confusion Matrix: {key} vs {ground_truth_key}')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        # Save explicitly in this section for clarity
        cm_fig_path = os.path.join(output_cfg['evaluation_plots_output'], f'confusion_matrix_{key}_vs_{ground_truth_key}.png')
        plt.savefig(cm_fig_path, dpi=viz_cfg['figure_dpi'])
        plt.close() # Close after saving
        print(f"Saved confusion matrix to {cm_fig_path}")


    # --- Cluster Composition Bar Plots ---
    print(f"\n--- 3. Generating Cluster Composition Plots ---")
    try:
        # Use a perceptually uniform colormap if many categories, or tab20 if fewer
        if len(true_label_names) > 20:
             colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(true_label_names)))
        else:
             colors = plt.get_cmap('tab20').colors
    except ValueError:
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(true_label_names))) # Fallback


    for key in valid_clustering_keys_for_eval:
        print(f"Generating Composition Plot for: {key}")
        ct = pd.crosstab(adata.obs[key], true_labels)
        # Normalize by predicted cluster (rows sum to 1)
        ct_norm = ct.apply(lambda x: x / x.sum(), axis=1)

        ax = ct_norm.plot(kind='bar', stacked=True, figsize=(max(10, len(ct_norm.index)*0.5), 7),
                          color=colors[:len(true_label_names)])

        plt.title(f'Composition of {key} Clusters by {ground_truth_key} Labels')
        plt.xlabel(f'Predicted Cluster ({key})')
        plt.ylabel('Proportion of Cells')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=ground_truth_key, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust rect to make space for legend

        comp_fig_path = os.path.join(output_cfg['evaluation_plots_output'], f'composition_plot_{key}.png')
        plt.savefig(comp_fig_path, dpi=viz_cfg['figure_dpi'])
        plt.close()
        print(f"Saved composition plot to {comp_fig_path}")
"""
# --- Latent Dimension Interpretability ---
print("\n" + "="*50)
print("--- Latent Dimension Interpretability ---")
print("="*50 + "\n")

interp_dir = output_cfg['interpretability_output_dir']
timepoint_key = data_keys_cfg['timepoint_key']
timepoint_pre = data_keys_cfg['timepoint_pre']
timepoint_post = data_keys_cfg['timepoint_post'] # This can be a string or a list

dim_correlations = pd.DataFrame() # Initialize, will be populated if correlation runs
top_dims_info = [] # Store info about top correlated dimensions

if not interp_cfg.get('correlate_dims_with_timepoint', False) and not interp_cfg.get('perform_deg_analysis', False):
     print("Skipping interpretability analysis: 'correlate_dims_with_timepoint' and 'perform_deg_analysis' are both False in config.")
elif model_embedding_key not in adata.obsm:
     warnings.warn(f"Skipping interpretability: Embedding key '{model_embedding_key}' not found in adata.obsm.")
elif timepoint_key not in adata.obs:
     warnings.warn(f"Skipping interpretability: Timepoint key '{timepoint_key}' not found in adata.obs.")
else:
    latent_dims = adata.obsm[model_embedding_key]
    if not isinstance(latent_dims, np.ndarray): # Ensure it's a numpy array for processing
        warnings.warn(f"Latent dimensions at adata.obsm['{model_embedding_key}'] are not a NumPy array. Attempting to convert.")
        try:
            latent_dims = np.array(latent_dims)
        except Exception as e:
            # If conversion fails, we cannot proceed with numerical operations
            warnings.warn(f"Could not convert latent_dims to NumPy array: {e}. Skipping interpretability analysis.")
            # Set a flag or exit this block if this is critical
            latent_dims = None # Or handle error more gracefully

    if latent_dims is not None:
        n_dims = latent_dims.shape[1]

        # --- Correlate Latent Dims with Timepoint ---
        if interp_cfg.get('correlate_dims_with_timepoint', False):
            print(f"--- Correlating {n_dims} Latent Dimensions with Timepoint ({timepoint_key}) ---")

            unique_adata_timepoints = set(adata.obs[timepoint_key].astype(str).unique()) # Ensure string comparison
            pre_timepoint_str = str(timepoint_pre) # Ensure pre_timepoint is string for comparison

            pre_found = pre_timepoint_str in unique_adata_timepoints

            actual_post_labels_in_data = []
            if isinstance(timepoint_post, str):
                if str(timepoint_post) in unique_adata_timepoints:
                    actual_post_labels_in_data.append(str(timepoint_post))
            elif isinstance(timepoint_post, (list, tuple)):
                for p_val in timepoint_post:
                    if str(p_val) in unique_adata_timepoints:
                        actual_post_labels_in_data.append(str(p_val))

            if not pre_found or not actual_post_labels_in_data:
                warnings.warn(
                    f"Cannot perform correlation: Timepoint '{pre_timepoint_str}' (pre) or specified "
                    f"post-treatment timepoint(s) {timepoint_post} not adequately represented in '{timepoint_key}'. "
                    f"Unique timepoints in data: {list(unique_adata_timepoints)}. "
                    f"Ensure '{pre_timepoint_str}' and at least one of {actual_post_labels_in_data if actual_post_labels_in_data else timepoint_post} are present."
                )
            else:
                mapping_dict = {pre_timepoint_str: 0}
                for post_val in actual_post_labels_in_data: # Map all found post-treatment labels to 1
                    mapping_dict[post_val] = 1

                timepoint_numeric = adata.obs[timepoint_key].astype(str).map(mapping_dict)
                valid_idx = timepoint_numeric.notna() # Indices of cells belonging to pre or mapped post groups

                if valid_idx.sum() < 2: # Need at least two points to correlate
                     warnings.warn("Not enough valid timepoint data points (mapped to 0 or 1) for correlation.")
                else:
                    latent_dims_valid = latent_dims[valid_idx]
                    timepoint_numeric_valid = timepoint_numeric[valid_idx]

                    corrs = []
                    pvals = []
                    for i in range(n_dims):
                        dim_values_for_corr = latent_dims_valid[:, i]
                        # Ensure there's variance in both series for pearsonr
                        if np.isnan(dim_values_for_corr).any() or np.isinf(dim_values_for_corr).any():
                            corrs.append(np.nan)
                            pvals.append(np.nan)
                            warnings.warn(f"NaN or Inf found in latent dimension {i}, skipping correlation for this dim.")
                            continue
                        if len(np.unique(dim_values_for_corr)) < 2 or len(np.unique(timepoint_numeric_valid)) < 2:
                            corrs.append(np.nan)
                            pvals.append(np.nan)
                            # warnings.warn(f"Skipping correlation for dim {i}: Not enough variance in dimension values or timepoint values.")
                            continue

                        try:
                             corr, pval = pearsonr(dim_values_for_corr, timepoint_numeric_valid)
                             corrs.append(corr)
                             pvals.append(pval)
                        except ValueError as e:
                             warnings.warn(f"Could not calculate correlation for dimension {i}: {e}")
                             corrs.append(np.nan)
                             pvals.append(np.nan)

                    if any(not np.isnan(c) for c in corrs): # If any correlations were actually computed
                        dim_correlations = pd.DataFrame({
                            'Dimension': range(n_dims),
                            'Correlation': corrs,
                            'PValue': pvals
                        })
                        dim_correlations['AbsCorrelation'] = dim_correlations['Correlation'].abs()
                        dim_correlations = dim_correlations.sort_values('AbsCorrelation', ascending=False, na_position='last').reset_index(drop=True)

                        print("\nTop Latent Dimensions Correlated with Timepoint:")
                        n_top_dims_to_show_corr = interp_cfg.get('n_top_dims', 5)
                        print(dim_correlations.head(n_top_dims_to_show_corr).to_markdown(index=False, floatfmt=".4f"))

                        corr_output_filename = interp_cfg.get('correlation_filename', 'latent_dim_timepoint_correlations.csv')
                        corr_output_path = os.path.join(interp_dir, corr_output_filename)
                        dim_correlations.to_csv(corr_output_path, index=False)
                        print(f"Full correlation results saved to {corr_output_path}")

                        top_dims_info = dim_correlations.dropna(subset=['Correlation']).head(n_top_dims_to_show_corr)['Dimension'].tolist()
                    else:
                        warnings.warn("No valid correlations could be computed for any dimension.")
        else:
            print("Skipping correlation of latent dimensions with timepoint as 'correlate_dims_with_timepoint' is False.")


        # --- DEG Analysis and Pathway Enrichment for Top Dimensions ---
        if interp_cfg.get('perform_deg_analysis', False) and top_dims_info:
            print(f"\n--- Performing DEG Analysis for Top {len(top_dims_info)} Correlated Dimensions ---")

            attempt_pathway_enrichment_overall = False
            enrichr_libraries_to_use = []

            if interp_cfg.get('perform_pathway_enrichment', False):
                if gseapy is None:
                     warnings.warn("gseapy not installed. Pathway enrichment will be skipped for all dimensions.")
                else:
                    configured_enrichr_libs = interp_cfg.get('enrichr_libraries')
                    if configured_enrichr_libs and isinstance(configured_enrichr_libs, list) and len(configured_enrichr_libs) > 0:
                        attempt_pathway_enrichment_overall = True
                        enrichr_libraries_to_use = configured_enrichr_libs
                        print(f"INFO: Pathway enrichment is enabled and will use Enrichr libraries: {enrichr_libraries_to_use}")
                    else:
                        warnings.warn("WARNING: Pathway enrichment is enabled ('perform_pathway_enrichment: True'), "
                                      "but 'interpretability.enrichr_libraries' is missing, empty, or not a list in the config. "
                                      "Pathway enrichment will be skipped for all dimensions.")

            for dim_idx in top_dims_info:
                current_dim_corr_info = None
                if not dim_correlations.empty and 'Dimension' in dim_correlations.columns:
                    match_series = dim_correlations['Dimension'] == dim_idx
                    if match_series.any():
                        current_dim_corr_info = dim_correlations[match_series].iloc[0]

                if current_dim_corr_info is not None and not pd.isna(current_dim_corr_info['Correlation']):
                    corr_value = current_dim_corr_info['Correlation']
                    pval_value = current_dim_corr_info['PValue']
                    corr_direction = "Positively" if corr_value > 0 else "Negatively"
                    print(f"\nAnalyzing Dimension {dim_idx} ({corr_direction} correlated with post-treatment state, "
                          f"Corr={corr_value:.3f}, PVal={pval_value:.2E})")
                else:
                    print(f"\nAnalyzing Dimension {dim_idx} (correlation details might be unavailable or NaN)")

                dim_values_for_grouping = latent_dims[:, dim_idx]
                if np.std(dim_values_for_grouping) < 1e-9 or len(np.unique(dim_values_for_grouping)) < 10: # Increased threshold slightly
                    warnings.warn(f"Skipping DEG for Dim {dim_idx}: Low variance or too few unique values in dimension scores.")
                    continue

                q_low = np.percentile(dim_values_for_grouping, 25)
                q_high = np.percentile(dim_values_for_grouping, 75)
                temp_group_key = f'dim_{dim_idx}_group' # Temporary column name

                if q_low == q_high: # Handle cases where quantiles are identical
                    if len(np.unique(dim_values_for_grouping)) > 1: # If there's still some variance
                        median_val = np.median(dim_values_for_grouping)
                        # Create two groups based on median; this might result in uneven groups
                        adata.obs[temp_group_key] = np.where(dim_values_for_grouping > median_val, 'High', 'Low')
                        # Ensure 'High' and 'Low' are present, if not, this split is problematic
                        if 'High' not in adata.obs[temp_group_key].unique() or 'Low' not in adata.obs[temp_group_key].unique():
                             warnings.warn(f"Skipping DEG for Dim {dim_idx}: Median split did not yield distinct High/Low groups.")
                             if temp_group_key in adata.obs: del adata.obs[temp_group_key]
                             continue
                    else: # No variance at all
                        warnings.warn(f"Skipping DEG for Dim {dim_idx}: 25th and 75th percentiles are identical, and no variance in data to split by median.")
                        continue
                else:
                    adata.obs[temp_group_key] = 'Middle'
                    adata.obs.loc[dim_values_for_grouping <= q_low, temp_group_key] = 'Low'
                    adata.obs.loc[dim_values_for_grouping >= q_high, temp_group_key] = 'High'

                adata.obs[temp_group_key] = adata.obs[temp_group_key].astype('category')
                group_counts = adata.obs[temp_group_key].value_counts()

                if group_counts.get('High', 0) < 3 or group_counts.get('Low', 0) < 3:
                    warnings.warn(f"Skipping DEG for Dim {dim_idx}: Fewer than 3 cells in 'High' or 'Low' group. Counts: {group_counts.to_dict()}")
                    if temp_group_key in adata.obs: del adata.obs[temp_group_key]
                    continue

                print(f"  Running DEG analysis (High vs Low) for Dim {dim_idx}...")
                try:
                    deg_layer = config.get('embedding', {}).get('input_layer') # Use layer from embedding section if specified for source data
                    if deg_layer and deg_layer not in adata.layers:
                        warnings.warn(f"  Layer '{deg_layer}' specified for DEG not found in adata.layers. Using adata.X.")
                        deg_layer = None
                    elif deg_layer:
                        print(f"  Using layer '{deg_layer}' for DEG analysis.")

                    rank_genes_key = f'rank_genes_dim_{dim_idx}'
                    sc.tl.rank_genes_groups(adata, groupby=temp_group_key, use_raw=False, groups=['High'], reference='Low',
                                          method='wilcoxon', key_added=rank_genes_key,
                                          layer=deg_layer, n_genes=adata.n_vars, pts=True)

                    deg_results_df = sc.get.rank_genes_groups_df(adata, group='High', key=rank_genes_key)
                    deg_results_df['logfoldchanges'] = pd.to_numeric(deg_results_df['logfoldchanges'], errors='coerce')
                    deg_results_df = deg_results_df.dropna(subset=['logfoldchanges'])
                    deg_results_df = deg_results_df.sort_values('pvals_adj')

                    deg_pval_thresh = interp_cfg.get('deg_pval_threshold', 0.05)
                    deg_lfc_thresh = interp_cfg.get('deg_lfc_threshold', 0.25)
                    sig_genes_df = deg_results_df[
                        (deg_results_df['pvals_adj'] < deg_pval_thresh) &
                        (np.abs(deg_results_df['logfoldchanges']) > deg_lfc_thresh)
                    ]
                    print(f"  Found {len(sig_genes_df)} significant DEGs (adj. p < {deg_pval_thresh}, |LFC| > {deg_lfc_thresh}).")

                    n_top_degs_report = interp_cfg.get('n_top_genes_deg', 50)
                    top_degs_for_output = sig_genes_df.head(n_top_degs_report)

                    if not top_degs_for_output.empty and 'names' in top_degs_for_output.columns:
                        print(f"  Top {len(top_degs_for_output)} DEGs for Dim {dim_idx} (sorted by adj. p-val):")
                        print(top_degs_for_output[['names', 'logfoldchanges', 'pvals_adj']].to_markdown(index=False, floatfmt=".3G"))
                        deg_filename = f'dim_{dim_idx}_top_{len(top_degs_for_output)}_degs.csv'
                        deg_output_path = os.path.join(interp_dir, deg_filename)
                        top_degs_for_output.to_csv(deg_output_path, index=False)
                        print(f"  Top DEGs saved to {deg_output_path}")
                    elif sig_genes_df.empty:
                        print(f"  No DEGs met the significance criteria for Dim {dim_idx}.")

                    # --- Pathway Enrichment ---
                    genes_for_enrichment_df = sig_genes_df # Use all significant DEGs for enrichment

                    if attempt_pathway_enrichment_overall and not genes_for_enrichment_df.empty:
                        if 'names' in genes_for_enrichment_df.columns and not genes_for_enrichment_df['names'].empty:
                            gene_list_for_enrichment = genes_for_enrichment_df['names'].tolist()
                            gsea_organism = interp_cfg.get('gsea_organism', 'human')
                            print(f"  Performing GSEApy enrichment for Dim {dim_idx} using {len(gene_list_for_enrichment)} DEGs (Organism: {gsea_organism})...")
                            try:
                                enrichment_outdir_dim = os.path.join(interp_dir, f'enrichr_dim_{dim_idx}')
                                # gseapy.enrichr creates outdir if it doesn't exist
                                enr_results = gseapy.enrichr(gene_list=gene_list_for_enrichment,
                                                             gene_sets=enrichr_libraries_to_use,
                                                             organism=gsea_organism,
                                                             outdir=enrichment_outdir_dim,
                                                             cutoff=0.05, # p-value cutoff for results table
                                                             verbose=False)

                                if enr_results and hasattr(enr_results, 'results') and not enr_results.results.empty:
                                    print(f"  Enrichment Results from GSEApy (Top 10 by Adj. P-value):")
                                    sorted_enrich_results = enr_results.results.sort_values('Adjusted P-value').reset_index(drop=True)
                                    print(sorted_enrich_results.head(10).to_markdown(index=False, floatfmt=".2E"))
                                    combined_results_filename = interp_cfg.get('enrichment_results_filename', 'gseapy_enrichr_combined_results.csv')
                                    combined_results_path = os.path.join(enrichment_outdir_dim, combined_results_filename)
                                    enr_results.results.to_csv(combined_results_path, index=False)
                                    print(f"  Full GSEApy Enrichr results DataFrame saved to: {combined_results_path}")
                                    print(f"  Individual library plots/tables saved by GSEApy in: {enrichment_outdir_dim}")
                                else:
                                    print("  No significant enrichment results found by GSEApy or results table was empty.")
                            except Exception as e:
                                warnings.warn(f"  GSEApy Enrichr failed for Dim {dim_idx}: {e}")
                        else:
                            print(f"  Skipping pathway enrichment for Dim {dim_idx}: No valid gene names in DEGs list.")
                    elif interp_cfg.get('perform_pathway_enrichment', False) and genes_for_enrichment_df.empty:
                        print(f"  Skipping pathway enrichment for Dim {dim_idx}: No significant DEGs found.")

                except Exception as e:
                     warnings.warn(f"  DEG analysis or subsequent enrichment failed for Dim {dim_idx}: {e}")
                finally:
                    if temp_group_key in adata.obs:
                         del adata.obs[temp_group_key]
                    # Optionally clean up adata.uns[rank_genes_key] if space is a concern

        elif interp_cfg.get('perform_deg_analysis', False) and not top_dims_info:
             print("Skipping DEG analysis (and pathway enrichment): No top correlated dimensions identified or correlation step was skipped/failed.")
        elif not interp_cfg.get('perform_deg_analysis', False):
             print("Skipping DEG analysis (and pathway enrichment) as 'perform_deg_analysis' is False.")

    else: # if latent_dims was None due to conversion failure
        print("Skipping interpretability analysis because latent dimensions could not be processed.")

print("\n" + "="*50)
print("--- Interpretability Analysis Potentially Complete ---")
print("="*50 + "\n")
"""

# %%
# --- scIB Benchmark ---
print("\n" + "="*50)
print("--- Running scIB Benchmark ---")
print("="*50 + "\n")

if not bench_cfg['run_benchmark']:
    print("Skipping scIB benchmark as configured.")
elif not bench_cfg['embedding_keys']:
    warnings.warn("Skipping scIB benchmark: No embedding keys specified in config.")
else:
    # Verify keys exist in adata and config
    label_key = data_keys_cfg['ground_truth_key'] # Use same ground truth as eval
    batch_key = data_keys_cfg['batch_key']
    embedding_keys = [k for k in bench_cfg['embedding_keys'] if k in adata.obsm]
    missing_keys = [k for k in bench_cfg['embedding_keys'] if k not in adata.obsm]

    if not embedding_keys:
        warnings.warn("Skipping scIB benchmark: None of the specified embedding keys found in adata.obsm.")
    elif label_key not in adata.obs:
        warnings.warn(f"Skipping scIB benchmark: Label key '{label_key}' not found in adata.obs.")
    elif batch_key not in adata.obs:
        warnings.warn(f"Skipping scIB benchmark: Batch key '{batch_key}' not found in adata.obs.")
    else:
        if missing_keys:
            print(f"Warning: The following embedding keys specified for scIB were not found and will be skipped: {missing_keys}")

        print(f"Benchmarking embeddings: {embedding_keys}")
        print(f"Using label key: '{label_key}' and batch key: '{batch_key}'")

        # Define metric configurations based on YAML
        bio_conservation = BioConservation(
            nmi_ari_cluster_labels_kmeans=bench_cfg['bio_conservation_kmeans'],
            nmi_ari_cluster_labels_leiden=bench_cfg['bio_conservation_leiden']
            # Add other BioConservation metrics if needed
        )
        batch_correction = BatchCorrection(
            pcr_comparison=bench_cfg['batch_correction_pcr']
            # Add other BatchCorrection metrics if needed
        )

        # Run Benchmarker
        bm = Benchmarker(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            embedding_obsm_keys=embedding_keys,
            # Pre-computed neighbors and UMAP can be used if available, but Benchmarker recalculates as needed
            # neighbors_obsm_keys = {'X_pca': 'neighbors', 'X_umap': 'neighbors_umap'}, # Example if needed
            n_jobs=bench_cfg['n_jobs'],
            bio_conservation_metrics=bio_conservation,
            batch_correction_metrics=batch_correction
        )

        bm.benchmark()

        # Plot and save results
        print("Plotting scIB results table...")
        results_table_path = os.path.join(output_cfg['benchmarking_output_dir'], "scib_results_table")
        bm.plot_results_table(min_max_scale=False, save_dir=results_table_path) # Saves plot and csv

        # You can access the results dataframe directly:
        results_df = bm.get_results(min_max_scale=False)
        print("\n--- scIB Benchmark Results ---")
        print(results_df.to_markdown(floatfmt=".4f"))
        results_csv_path = os.path.join(output_cfg['benchmarking_output_dir'], 'scib_results_summary.csv')
        results_df.to_csv(results_csv_path)
        print(f"scIB results summary saved to {results_csv_path}")


# %%
# --- Final Save of Processed AnnData ---
print("\n" + "="*50)
print("--- Saving Final Processed AnnData ---")
print("="*50 + "\n")

final_adata_path = os.path.join(output_cfg['adata_clustered_dir'], output_cfg['adata_clustered_filename'])
print(f"Saving AnnData with embeddings, clusters, and other results to: {final_adata_path}")

# Optional: Clean up large intermediate data before saving if necessary
# e.g., remove GMM probabilities if not needed downstream
# if 'GMM_probabilities' in adata.obsm:
#     del adata.obsm['GMM_probabilities']

# Ensure the directory exists
os.makedirs(os.path.dirname(final_adata_path), exist_ok=True)

try:
    adata.write(final_adata_path, compression='gzip')
    print("Final AnnData saved successfully.")
except Exception as e:
    warnings.warn(f"Error saving final AnnData: {e}")

print("\n--- Analysis Script Completed ---")
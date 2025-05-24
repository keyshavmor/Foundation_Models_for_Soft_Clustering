import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_PATH = "/cluster/home/maurdu/Foundation_models_NB/clustering/resources/embeddings/embedding_processed.h5ad"
#DATA_PATH = "/cluster/home/maurdu/Foundation_models_NB/clustering/resources/embeddings/embedding.h5ad"
# EMBED_KEY = 'X_pca'
#PLOT_TITLE = 'Raw Combined Dataset'
PLOT_TITLE = 'CancerFoundation'

adata = sc.read_h5ad(DATA_PATH)
#print("Applying neighbors...")
#sc.pp.neighbors(adata, use_rep=EMBED_KEY)
#sc.tl.umap(adata)

adata.obs['cell type'] = adata.obs['cell_type'].combine_first(adata.obs['Cell_type']).copy()
adata.obs['stage code'] = adata.obs['Stage_Code'].combine_first(adata.obs['Timepoint']).copy()
adata.obs['cell state'] = adata.obs['cell_state'].copy()
adata.obs['sample'] = adata.obs['SAMPLES_JOINT'].copy()
adata.obs['stage code'].replace('relapse', 'post-treatment', inplace=True)
adata.obs['cell type'].replace('fibroblast', 'Fibroblast', inplace=True)
adata.obs['cell type'].replace('Schwann cell', 'Schwann', inplace=True)
adata.obs['cell type'].replace('neuroblast (sensu Vertebrata)', 'Neuroblast', inplace=True)

umap_obs = ['sample', 'cell type', 'cell state', 'stage code']

fig, axes = plt.subplots(1, len(umap_obs), figsize=(5 * len(umap_obs), 5))
print("Generating Umap plots for umap_obs: {}".format(umap_obs))
for i, feature in enumerate(tqdm(umap_obs, desc="Plotting UMAPs")):
    palette = None
    if feature == 'cell type':
        palette = {
            'Neuroblast': '#1f77b4',
            'Fibroblast': '#ff7f0e',
            'Schwann': '#2ca02c',
            'Endothelial': '#d62728',
            'Neuroendocrine': '#9467bd',
            'Stromal other': '#A9A9A9',  # Fix: Added missing key 'Stromal other' with a color
        }
    sc.pl.umap(
        adata,
        color=feature,
        ax=axes[i],
        show=False,
        legend_fontsize=7,
        title=feature,
        legend_loc=None if i == 0 else 'right margin',  # remove legend for first plot
        palette=palette
    )
plt.suptitle(PLOT_TITLE, fontsize=20, y=1.02) ###Change your model name
plt.tight_layout()
plt.savefig(f'{PLOT_TITLE}.png', dpi=300, bbox_inches='tight')
plt.show()
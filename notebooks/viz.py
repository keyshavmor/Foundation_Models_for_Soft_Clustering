import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.read_h5ad("/cluster/home/maurdu/Foundation_models_NB/clustering/resources/embeddings/embedding_processed.h5ad")
#sc.pp.neighbors(adata,use_rep="cancerfnd_pca_harmony", n_neighbors=15,random_state=0)
#sc.tl.umap(adata)
#adata = sc.read_h5ad("/cluster/home/maurdu/Foundation_models_NB/clustering/resources/embeddings/embedding.h5ad")
adata.obs['CELL_TYPE_JOINT'] = adata.obs['cell_type'].combine_first(adata.obs['Cell_type'])
adata.obs['STAGE_CODE_JOINT'] = adata.obs['Stage_Code'].combine_first(adata.obs['Timepoint'])
adata.obs['STAGE_CODE_JOINT'].replace('relapse', 'post-treatment', inplace=True)

umap_obs = ['SAMPLES_JOINT', 'CELL_TYPE_JOINT', 'cell_state', 'STAGE_CODE_JOINT']

fig, axes = plt.subplots(1, len(umap_obs), figsize=(5 * len(umap_obs), 5))

for i, feature in enumerate(umap_obs):
    sc.pl.umap(
        adata,
        color=feature,
        ax=axes[i],
        show=False,
        legend_fontsize=7,
        title=feature,
        legend_loc=None if i == 0 else 'right margin'  # remove legend for first plot
    )
plt.suptitle('CancerFoundation', fontsize=20, y=1.02) ###Change your model name
#plt.suptitle('Raw Combined Dataset', fontsize=20, y=1.02) ###Change your model name
plt.tight_layout()
plt.savefig("CancerFoundation_harmony_umap.png", dpi=300, bbox_inches='tight')
plt.show()
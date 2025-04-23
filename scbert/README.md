## File Descriptions

| File                     | Description |
|--------------------------|-------------|
| `preprocessing.ipynb`    | Reduce the number of genes using highly_variable_genes. Then do the preprocessing step that came with the model. |
| `embed.ipynb`            | Runs the Tencent scBERT model in batches to avoid memory limitations on the cluster. Once embeddings are computed, we only keep the embeddings corresponding to the genes that where present in our preprocessed data.|
| `postprocessing.ipynb`   | Combines batches, then applies PCA and UMAP for 2D visualization/analysis. |
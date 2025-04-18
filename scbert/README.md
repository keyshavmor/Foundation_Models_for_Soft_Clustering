## File Descriptions

| File                     | Description |
|--------------------------|-------------|
| `preprocessing.ipynb`    | Preprocessing script (originally from Tencent), optimized for faster execution. |
| `feature_selection.ipynb`| Performs feature selection on the sparse preprocessing output. Retains only high-variance attributes via an iteratively determined threshold (tested on a small subset to balance attribute reduction with model performance). |
| `embed.ipynb`            | Runs the Tencent scBERT model in batches to avoid memory limitations on the cluster. |
| `postprocessing.ipynb`   | Combines batches, then applies PCA and UMAP for 2D visualization/analysis. |

## Key Notes
- **Feature Selection**: Threshold was empirically determined using a small representative dataset.
- **Batching**: Necessary due to cluster memory constraints (full embedding output exceeds capacity).
import numpy as np
import os
import scanpy as sc
import sys
import tensorflow as tf
import warnings

import sfaira
from sfaira.consts import AdataIdsSfaira
from sfaira.data import clean_string


print(tf.__version__)

# Set global variables.
print("sys.argv", sys.argv)

model_class = str(sys.argv[1]).lower()
organism = str(sys.argv[2]).lower()
organ = str(sys.argv[3]).lower()
model_type = str(sys.argv[4]).lower()

organisation = str(sys.argv[5]).lower()
gridsearch_key = str(sys.argv[6])
base_path = str(sys.argv[7])
data_path = str(sys.argv[8])
config_path = str(sys.argv[9])

topology = "0.1"
version = "0.1"
gs_id = "0"

model_id = model_class + "_" + \
           clean_string(organism) + "-" + clean_string(organ) + "-" + model_type + "-" + topology + "-" + version + \
           "_" + organisation
fn_out = base_path + "/results/" + model_id + "_" + gs_id
config_fn = os.path.join(config_path, f"config_{clean_string(organism)}_{clean_string(organ)}.pickle")

# Train project:
np.random.seed(1)
data_store = sfaira.data.load_store(cache_path=data_path, store_format="h5ad")
data_store.load_config(fn=config_fn)
trainer = sfaira.train.TrainModelEmbedding(
    data=data_store,
    model_path=fn_out,
)
trainer.load_into_memory()  # trainer.data is now AnnData in memory.
if organism == "mouse" and "development_stage" in trainer.data.obs.columns:
    # Note: the data sets are loaded into memory here, only do this if the partitions are small enough.
    # In the case of mouse data, we need this to add a columns to .obs.
    split_key = "dataset_age"
    # Add new cell description column into data sets that allows splitting tabula muris senis objects by age:
    trainer.data.obs[split_key] = [
        str(x) + "_" + str(y) for x, y in zip(trainer.data.obs["dataset_id"].values,
                                              trainer.data.obs["development_stage"].values)
    ]
    splits = np.sort(np.unique(trainer.data.obs[split_key].values))
else:
    split_key = "id"
    splits = np.sort(np.unique(trainer.data.obs["dataset_id"].values))
print(splits)
adata_ids = AdataIdsSfaira()
n_pcs = 64
# Loop CV over data.
for i, x in enumerate(splits):
    print("CV iteration %i" % i)
    np.random.seed(i)
    adata = trainer.data[trainer.data.obs[split_key].values == x, :].copy()
    if adata.n_obs <= n_pcs:  # skip splits with too few observations
        warnings.warn(f"skipped dataset split {x} because there were not more than {n_pcs} observations in this dataset")
        continue
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=n_pcs)
    arr_pca = adata.obsm['X_pca']
    np.save(file=fn_out + "_cv" + str(i) + "_embedding", arr=arr_pca)
    ok = [getattr(adata_ids, k) for k in adata_ids.obs_keys if getattr(adata_ids, k) in adata.obs_keys()]
    df_summary = adata.obs[ok].copy()
    df_summary["ncounts"] = np.asarray(adata.X.sum(axis=1)).flatten()
    df_summary.to_csv(fn_out + "_cv" + str(i) + "_covar.csv")

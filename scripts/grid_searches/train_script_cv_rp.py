import numpy as np
import os
import sys
import tensorflow as tf
from sklearn import random_projection
import pickle

import sfaira
from sfaira.consts import AdataIdsSfaira
from sfaira.consts.utils import clean_id_str


def negll_nb(y_true, y_pred):
    x = tf.convert_to_tensor(y_true, dtype="float32")
    loc = tf.convert_to_tensor(y_pred, dtype="float32")
    scale = tf.ones_like(loc)

    eta_loc = tf.math.log(loc)
    eta_scale = tf.math.log(scale)

    log_r_plus_mu = tf.math.log(scale + loc)

    ll = tf.math.lgamma(scale + x)
    ll = ll - tf.math.lgamma(x + tf.ones_like(x))
    ll = ll - tf.math.lgamma(scale)
    ll = ll + tf.multiply(x, eta_loc - log_r_plus_mu) + tf.multiply(scale, eta_scale - log_r_plus_mu)

    ll = tf.clip_by_value(ll, -300, 300, "log_probs")
    neg_ll = -ll
    neg_ll = tf.reduce_mean(neg_ll)
    return neg_ll.numpy()


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

eps = 1e-10
n_comps = 64
topology = "0.1"
version = "0.1"
gs_id = "0"

model_id = model_class + "_" + \
           clean_id_str(organism) + "-" + clean_id_str(organ) + "-" + model_type + "-" + topology + "-" + version + \
           "_" + organisation
fn_out = base_path + "/results/" + model_id + "_" + gs_id
config_fn = os.path.join(config_path, f"config_{clean_id_str(organism)}_{clean_id_str(organ)}.pickle")

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


# Filter out splits that have less than n_comps observations
splits_filtered = []
for s in splits:
    if trainer.data[trainer.data.obs[split_key] == s].n_obs > n_comps:
        splits_filtered.append(s)

# Downsample splits:
if len(splits_filtered) > 3:
    np.random.seed(0)
    splits_filtered = np.random.choice(a=splits_filtered, size=3, replace=False)
elif len(splits_filtered) < 2:
    splits_filtered = [0.2, 0.2, 0.2]
    print(
        f"Less than 2 datasets detected, cannot do dataset-based cross validation. "
        f"Will use 3x random splits as testset: {splits_filtered}"
    )

adata_ids = AdataIdsSfaira()

# Loop CV over data.
for i, x in enumerate(splits_filtered):
    print("CV iteration %i" % i)
    print(x)
    np.random.seed(i)
    if isinstance(x, str):
        adata_train = trainer.data[trainer.data.obs[split_key].values != x, :].copy()
        adata_test = trainer.data[trainer.data.obs[split_key].values == x, :].copy()
    elif isinstance(x, float):
        idx = list(range(trainer.data.n_obs))
        idx_test = np.random.choice(a=idx, size=round(len(idx) * x), replace=False)
        idx_train = [i for i in idx if i not in idx_test]
        adata_test = trainer.data[idx_test].copy()
        adata_train = trainer.data[idx_train].copy()
    else:
        raise ValueError(f"split {x} is neither float nor string but of type {type(x)}")

    x_train = adata_train.X.copy()
    x_test = adata_test.X.copy()
    tr = random_projection.SparseRandomProjection(n_components=n_comps)
    tr = tr.fit(x_train)
    z = tr.transform(x_test)
    y = z.dot(tr.components_)
    y = y.A.clip(min=eps)
    x_test = x_test.A
    mse = ((x_test - y) ** 2).mean()
    negll = negll_nb(x_test, y)
    metrics = {"custom_mse": mse, "custom_negll": negll}
    with open(fn_out + "_cv" + str(i) + "_evaluation.pickle", 'wb') as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
    np.save(file=fn_out + "_cv" + str(i) + "_embedding", arr=z)
    ok = [getattr(adata_ids, k) for k in adata_ids.obs_keys if getattr(adata_ids, k) in adata_test.obs_keys()]
    df_summary = adata_test.obs[ok].copy()
    df_summary["ncells"] = np.asarray(adata_test.X.sum(axis=1)).flatten()
    df_summary.to_csv(fn_out + "_cv" + str(i) + "_covar.csv")

import numpy as np
import os
import sys
import tensorflow as tf
import time

import sfaira
from sfaira.data import clean_string
from sfaira.estimators.callbacks import WarmUpTraining

from sfaira_benchmarks import HyperparameterContainer

print(tf.__version__)

# Set global variables.
print("sys.argv", sys.argv)

model_class = str(sys.argv[1]).lower()
organism = str(sys.argv[2]).lower()
organ = str(sys.argv[3]).lower()
model_type = str(sys.argv[4]).lower()

depth_key = str(sys.argv[5])
width_key = str(sys.argv[6])
learning_rate_keys = str(sys.argv[7])
dropout_key = str(sys.argv[8])
l1_key = str(sys.argv[9])
l2_key = str(sys.argv[10])

organisation = str(sys.argv[11]).lower()
topology = str(sys.argv[12]).lower()
version = str(sys.argv[13]).lower()

gridsearch_key = str(sys.argv[14])
base_path = str(sys.argv[15])
data_path = str(sys.argv[16])
config_path = str(sys.argv[17])
target_path = str(sys.argv[18])

# change model class name to mlp if linear was given
# linear is used up to here only so that linear models are saved in separate folder
if model_class.lower() == "celltype":
    if topology.split(".")[1] == "0":
        if model_type.lower() == "linear":
            model_type = "mlp"
model_id = model_class + "_" + \
           clean_string(organism) + "-" + clean_string(organ) + "-" + model_type + "-" + topology + "-" + version + \
           "_" + organisation
for learning_rate_key in learning_rate_keys.split("+"):
    gs_id = depth_key + "_" + width_key + "_" + learning_rate_key + "_" + dropout_key + "_" + l1_key + "_" + l2_key

    if model_type[:3] == 'vae':
        callbacks = [WarmUpTraining()]
        patience = 150  # guarantees 50 epochs after warm-up
    else:
        callbacks = None
        patience = 50

    fn_tensorboard = base_path + "/logs/" + model_id + "_" + gs_id
    fn_out = base_path + "/results/" + model_id + "_" + gs_id
    config_fn = os.path.join(config_path, f"config_{clean_string(organism)}_{clean_string(organ)}.pickle")
    fn_target_universe = os.path.join(target_path, f"targets_{clean_string(organism)}_{clean_string(organ)}.csv")

    # Train project:
    # Assemble hyperparameter that mask topology version presets:
    hpcontainer = HyperparameterContainer()
    override_hyperpar = {
        "l1_coef": hpcontainer.l1_coef[l1_key],
        "l2_coef": hpcontainer.l2_coef[l2_key],
        "dropout_rate": hpcontainer.dropout[dropout_key]
    }

    t0 = time.time()
    np.random.seed(1)
    data_store = sfaira.data.load_store(cache_path=data_path, store_format="h5ad")
    data_store.load_config(fn=config_fn)
    if model_class == "embedding":
        trainer = sfaira.train.TrainModelEmbedding(
            data=data_store,
            model_path=fn_out,
        )
    elif model_class == "celltype":
        trainer = sfaira.train.TrainModelCelltype(
            data=data_store,
            model_path=fn_out,
            fn_target_universe=fn_target_universe,
        )
    else:
        raise ValueError("model_class %s not recognized" % model_class)
    print(f"TRAINING SCRIPT: time for initialising data access {time.time() - t0}s.")

    trainer.zoo.model_id = model_id
    t0 = time.time()
    trainer.init_estim(override_hyperpar=override_hyperpar)
    print(f"TRAINING SCRIPT: time initialising estimator {time.time() - t0}s.")
    t0 = time.time()
    if organism == "mouse" and np.any([
        "development_stage" in trainer.data.adata_by_key[k].obs.columns
        for k in trainer.data.indices.keys()
    ]):
        # Note: the data sets are loaded into memory here, only do this if the partitions are small enough.
        # In the case of mouse data, we need this to add a columns to .obs.
        trainer.load_into_memory()  # trainer.data is now AnnData in memory.
        split_key = "dataset_age"
        # Add new cell description column into data sets that allows splitting tabula muris senis objects by age:
        trainer.data.obs[split_key] = [
            str(x) + "_" + str(y) for x, y in zip(trainer.data.obs["dataset_id"].values,
                                                  trainer.data.obs["development_stage"].values)
        ]
        splits = np.sort(np.unique(trainer.data.obs[split_key].values))
        keep_data_in_memory = True  # Define whether to keep data set in memory.
        cache_full = True
    else:
        split_key = "id"
        splits = np.sort(np.unique(list(trainer.data.indices.keys())))
        # Define whether to keep data set in memory.
        if trainer.data.n_obs < 5*1e4:
            keep_data_in_memory = True
            cache_full = True
        elif organism == "human":
            keep_data_in_memory = True
            cache_full = False  # Do not increase memory load further.
        else:
            keep_data_in_memory = False
            cache_full = False
        if keep_data_in_memory:
            trainer.load_into_memory()  # trainer.data is now AnnData in memory.
    print(splits)
    print(f"TRAINING SCRIPT: time for defining splits {time.time() - t0}s.")

    # Filter out splits that have less than 65 observations
    splits_filtered = []
    for s in splits:
        if trainer.data[trainer.data.obs[split_key] == s].n_obs > 64:
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

    for i, x in enumerate(splits_filtered):
        print(f"TRAINING SCRIPT: CV iteration {i}: {x}")
        np.random.seed(i)
        t0 = time.time()
        trainer.init_estim(override_hyperpar=override_hyperpar)
        print(f"TRAINING SCRIPT: time initialising estimator {time.time() - t0}s.")
        t0 = time.time()
        trainer.estimator.train(
            optimizer="adam",
            lr=hpcontainer.learning_rate[learning_rate_key],
            epochs=2000,
            max_steps_per_epoch=20,
            batch_size=256,
            validation_split=0.1,
            test_split={split_key: x} if isinstance(x, str) else x,
            validation_batch_size=256,
            max_validation_steps=4,
            patience=patience,
            shuffle_buffer_size=int(1e4) if keep_data_in_memory else None,
            cache_full=cache_full,
            randomized_batch_access=not keep_data_in_memory,
            retrieval_batch_size=512,
            prefetch=10,
            lr_schedule_min_lr=1e-10,
            lr_schedule_factor=0.2,
            lr_schedule_patience=25,
            weighted=False,
            callbacks=callbacks,
            log_dir=None  # fn_tensorboard + "_cv" + str(i)
        )
        print(f"TRAINING SCRIPT: time for training {time.time() - t0}s.")
        t0 = time.time()
        trainer.save(
            fn=fn_out + "_cv" + str(i),
            model=False,
            specific=True
        )
        print(f"TRAINING SCRIPT: time for testing {time.time() - t0}s.")

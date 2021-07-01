import numpy as np
import os
import sys
import json

import sfaira
from sfaira.consts.utils import clean_id_str
from sfaira.estimators.callbacks import WarmUpTraining

# Set global variables.
print("sys.argv", sys.argv)

model_class = str(sys.argv[1]).lower()
base_path = str(sys.argv[2])
data_path = str(sys.argv[3])
config_path = str(sys.argv[4])
target_path = str(sys.argv[5])
best_param_file = str(sys.argv[6])

model_class2, model_id, organisation = best_param_file.split('_')[:3]
# Sanity check on file.
assert model_class == model_class2
del model_class2
organism, organ, model_type, topology, version = model_id.split('-')


# Load best embeddings hyperparameter
with open(os.path.join(base_path, 'hyperparameter', best_param_file), 'r') as file:
    hyparam_all = json.load(file)
hyparam_model = hyparam_all["model"]
hyparam_optim = hyparam_all["optimizer"]


# change model class name to mlp if linear was given
# linear is used up to here only so that linear models are saved in separate folder
if model_class.lower() == "celltype":
    if topology.split(".")[1] == "0":
        if model_type.lower() == "linear":
            model_type = "mlp"
model_id = model_class + "_" + \
           clean_id_str(organism) + "-" + clean_id_str(organ) + "-" + model_type + "-" + topology + "-" + version + \
           "_" + organisation

if model_type[:3] == 'vae':
    callbacks = [WarmUpTraining()]
else:
    callbacks = None

fn_tensorboard = os.path.join(base_path, "logs", best_param_file[:-4])
fn_out = os.path.join(base_path, "results", best_param_file[:-4])
config_fn = os.path.join(config_path, f"config_{clean_id_str(organism)}_{clean_id_str(organ)}.pickle")
fn_target_universe = os.path.join(target_path, f"targets_{clean_id_str(organism)}_{clean_id_str(organ)}.csv")

# Train project:
# Assemble hyperparameter that mask topology version presets:
override_hyperpar = {
    "l1_coef": hyparam_model["l1_coef"],
    "l2_coef": hyparam_model["l2_coef"],
    "dropout_rate": hyparam_model["dropout_rate"]
}
np.random.seed(1)
data_store = sfaira.data.load_store(cache_path=data_path, store_format="h5ad")
data_store.load_config(fn=config_fn)
keep_data_in_memory = True  # Define whether to keep data set in memory.
cache_full = False
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
if keep_data_in_memory:
    trainer.load_into_memory()  # trainer.data is now AnnData in memory.

trainer.zoo.model_id = model_id
assert trainer.zoo.model_id is not None, "choose model in zoo first"
trainer.init_estim(override_hyperpar=override_hyperpar)
trainer.estimator.train(
    optimizer=hyparam_optim["optimizer"],
    lr=hyparam_optim["lr"],
    epochs=hyparam_optim["epochs"],
    max_steps_per_epoch=hyparam_optim["max_steps_per_epoch"],
    batch_size=hyparam_optim["batch_size"],
    validation_split=0.1,
    test_split=0.0,
    validation_batch_size=hyparam_optim["validation_batch_size"],
    max_validation_steps=hyparam_optim["max_validation_steps"],
    patience=hyparam_optim["patience"],
    shuffle_buffer_size=int(1e4) if keep_data_in_memory else None,
    cache_full=cache_full,
    randomized_batch_access=not keep_data_in_memory,
    retrieval_batch_size=512,
    prefetch=10,
    lr_schedule_min_lr=hyparam_optim["lr_schedule_min_lr"],
    lr_schedule_factor=hyparam_optim["lr_schedule_factor"],
    lr_schedule_patience=hyparam_optim["lr_schedule_patience"],
    callbacks=callbacks,
    log_dir=None  # fn_tensorboard
)
trainer.save(
    fn=fn_out,
    model=True,
    specific=True
)

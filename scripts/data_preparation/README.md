# Preparing the sfaira data/config/target store using a container
## Prepare container for sfaira
We usually run data preparation scripts inside containers.
You may choose to do this differently, important is that an R and a python installation are available in this environment.
The following installation scripts have to be executed in the environment of your choice:
```bash
sh .setup_container.sh
```

## Download raw data
You can choose which sfaira data sets you want to include in your fits.
Use the sfaira dataset documentation to build dataset groups and download the raw files.

## Write data store
This writes a store of acccess optimized on-disk representations of your data library.

```bash
sh .create_datastore.sh
```

## Write config store
This writes config files based on your data store which assign data sets to target anatomic partitions and organisms.

```bash
sh .create_configstore.sh
```

## Write target store
This defines the target (observed) leaf cell types for a cell type classifier given a config for the defined data scenario.
```bash
sh .create_targetstore.sh
```

# Ablation Experiments

This folder contains the batch of experiments used for the ablation study.

## Overview

-   This batch compares various configurations of the GNN-based heuristics.
-   In particular, we test the following configurations:
    -   **BITMASK representation**
    -   **HASHED representation**
    -   **MAPPED representation**
    -   For each representation, we test:
        -   GNN with HFS* mode
        -   GNN with HFS mode
-   All models of interest are located in the `_models/{domain_name}` subfolder.
-   All results are stored in the `_results` subfolder.

------------------------------------------------------------------------

## Usage

All commands should be run from the root directory of the repository.

> To fully reproduce the results, run the script for each dataset type.  
> If skipping data and model generation, make sure to place the correct model (identified by the subfolder it is stored in) in the `_models/{domain_name}` folder so that inference can locate it.

```console
python3 scripts/gnn_exp/run_batch.py --bin cmake-build-release-nn/bin/deep --exp_dir exp/gnn_exp/batch_ablation --dataset_type "BITMASK" --ablation
```

Replace `--bin` with the path to your compiled `deep` binary.

### Arguments

The main arguments to this script are:

- `--bin` : path to the deep executable (e.g. `cmake-build-release-nn/bin/deep`)
- `--exp_dir` : path to the experiment folder (e.g. `exp/gnn_exp/batch_ablation/`)
- `--dataset_type` : dataset type to use (`BITMASK`, `HASHED`, or `MAPPED`)
- `--ablation` : flag indicating this is an ablation study

Optional arguments:

- `--skip_gen` : skip dataset generation if already provided
- `--skip_train` : skip model training if models are already provided in `_models/{domain_name}`

For further details, check the script `scripts/gnn_exp/run_batch.py`.
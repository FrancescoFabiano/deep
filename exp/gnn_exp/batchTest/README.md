# Experiment Batch Test
------------------------------------------------------------------------

## Usage

All commands should be run from the root directory of the repository.
> While Steps 1 and 2 can be run, to emulate the results of the paper, you
can skip them and directly run Step 3 since models are already provided
in the `_models` folder.


### Run entire batch

``` console
python3 scripts/gnn_exp/create_all_training_data.py exp/gnn_exp/batchTest --deep_exe cmake-build-debug-nn/bin/deep
```



### 1. Generate training data

This creates training datasets for each domain:

``` console
python3 scripts/gnn_exp/create_all_training_data.py exp/gnn_exp/batchTest --deep_exe cmake-build-debug-nn/bin/deep
```

Replace --deep_exe with the path to your compiled deep binary.

#### Failures and Alternative Representations

If the script fails to generate training data for a specific domain, it
will skip that domain and continue with the others. The generation
process involves randomness, so retrying may succeed on a second
attempt (the first 12 attempts are seeded).
If some domains consistently fail to produce training data, you
can try adjusting the following options:

-   `--depth n`: Sets the maximum depth of the search tree to explore.
    The default is `25`. Increasing this value improves the chance of
    generating meaningful training data.
-   `--discard_factor x`: Specifies the maximum discard factor (a float
    in the interval `[0, 1)`). This controls how quickly the dataset
    generator abandons a subtree to explore another. The default is
    `0.4`. Lowering this value results in deeper and more exhaustive
    exploration. Increasing it makes the exploration more "jumpy" and
    likely to skip over parts of the tree.

Example command with adjusted parameters:

``` console
python3 scripts/gnn_exp/create_all_training_data.py exp/gnn_exp/batchTest --deep_exe cmake-build-debug-nn/bin/deep --depth 40 --discard_factor 0.2
```

Similarly, if you want to generate training data with a different
representation, you can use the `--dataset_type` option with one of the
following choices: `["MAPPED", "HASHED", "BITMASK"]` (default is
`HASHED`).\
To ensure consistency during inference and training, use the
same `dataset_type` in the training and execution phases
(as illustrated in Steps 2 and 3).

Example command with a different dataset type:

``` console
python3 scripts/gnn_exp/create_all_training_data.py exp/gnn_exp/batchTest --deep_exe cmake-build-debug-nn/bin/deep --dataset_type BITMASK
```

### 2. Train GNN models

This trains one model per domain using the previously generated training
data.
> This will overwrite existing models in the `_models` folder.

``` console
python3 scripts/gnn_exp/train_models.py exp/gnn_exp/batchTest
```

If you generated the training data with a specific representation, you must
use the same representation (`--dataset_type`) when training the model.\
The available options are: `["MAPPED", "HASHED", "BITMASK"]` (default: `HASHED`).

Example command with a different dataset type:

```console
python3 scripts/gnn_exp/train_models.py exp/gnn_exp/batchTest --dataset_type BITMASK
```

### 3. Run evaluation and aggregate results

Run inference using the trained models and aggregate the results into
the `_results` folder.

#### GNN heuristic

This command runs inference using the GNN heuristic with the appropriate
model generated in the previous step.

``` console
python3 scripts/gnn_exp/bulk_coverage_run.py cmake-build-debug-nn/bin/deep exp/gnn_exp/batchTest/ --threads 8 --binary_args "-s Astar -u GNN -c -b" --timeout 600
```

Example with changed dataset type:

``` console
python3 scripts/gnn_exp/bulk_coverage_run.py cmake-build-debug-nn/bin/deep exp/gnn_exp/batchTest/ --threads 8 --binary_args "-s Astar -u GNN -c -b --dataset_type BITMASK" --timeout 600
```

#### Breadth-First Search

This command runs inference using the BFS heuristic, which is the
baseline for comparison.

``` console
python3 scripts/gnn_exp/bulk_coverage_run.py cmake-build-debug-nn/bin/deep exp/gnn_exp/batchTest/ --threads 8 --binary_args "-c -b" --timeout 600
```

##### Arguments

The arguments to this script are:
- the path to the deep executable (`cmake-build-debug-nn/bin/deep` in
  the example)
- the path to the experiment folder (`exp/gnn_exp/batchTest/` which
  is this folder)
- the number of threads to use to speed up the testing (`8` in the
  example)
- `--binary_args` option allows you to pass additional arguments to the
  deep executable, such as specifying the search algorithm and whether
  to enable GNN heuristics.
- `--timeout` specifies the maximum time in
  seconds for each instance to be solved (`600` seconds in the example).

### Advanced Setup

For more complex scenarios---such as training a single GNN model across
multiple domains to improve generalization---we do not provide a
dedicated script to avoid clutter.\
However, you can adapt the existing scripts in the `/scripts` folder (in
the project root) or create a custom folder structure to achieve this.\
Models trained on merged domains are also included in `batch4`.

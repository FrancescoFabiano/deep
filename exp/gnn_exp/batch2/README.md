# Experiment Batch 2: Same Goals, Different Initial States

This folder contains the third batch of experiments for the project.

- This batch tests GNN-based heuristics against problem with identical goals but varying initial states.
- For each goal, a dedicated model is trained and then used to solve all instances that share that same goal.
- All models of interest are located in the `_models/{specific_goal}` subfolder.
- All the results are stored in the `_results` subfolder.

---

## Usage

All commands should be run from the root directory of the repository.

> While Steps 1 and 2 can be run, to emulate the results of the paper, you can skip them and directly run Step 3 since models are already provided in the `_models` folder.


### 1. Generate training data

This creates training datasets for each domain:

```console
python3 scripts/gnn_exp/create_all_training_data.py exp/gnn_exp/batch2 --deep_exe cmake-build-release-nn/bin/deep --no_goal
```
Replace --deep_exe with the path to your compiled deep binary.
The `--no_goal` flag is used to ensure that the training data does not include goal information, as the models are trained on instances with the same goal but different initial states.

#### Failings
If the script fails to generate training data for a specific domain, it will skip that domain and continue with the others.
The generation process involves randomness, so retrying may succeed on a second attempt.
In case some domains consistently fail to produce training data, you can try adjusting the following options:

- `--depth n`: Sets the maximum depth of the search tree to explore.
  The default is `25`.
  Increasing this value improves the chance of generating meaningful training data.
- `--discard_factor x`: Specifies the maximum discard factor (a float in the interval `[0, 1)`).
  This controls how quickly the dataset generator abandons a subtree to explore another.
  The default is `0.4`.
  Lowering this value results in deeper and more exhaustive exploration.
  Increasing it makes the exploration more "jumpy" and likely to skip over parts of the tree.

Example command with adjusted parameters:
```console
python3 scripts/gnn_exp/create_all_training_data.py exp/gnn_exp/batch2 --deep_exe cmake-build-release-nn/bin/deep --no_goal --depth 40 --discard_factor 0.2
```

### 2. Train GNN models
This trains one model per domain using the previously generated training data.
> This will overwrite existing models in the `_models` folder.


```console
python3 scripts/gnn_exp/train_models.py exp/gnn_exp/batch2 --no_goal
```
The `--no_goal` flag is used to ensure that the training data does not include goal information, as the models are trained on instances with the same goal but different initial states.


### 3. Run evaluation and aggregate results
Run inference using the trained models and aggregate the results into the `_results` folder.


#### GNN heuristic
This command runs inference using the GNN heuristic with the appropriate model generated in the previous step.
```console
python3 scripts/gnn_exp/bulk_coverage_run.py cmake-build-release-nn/bin/deep exp/gnn_exp/batch2/ --threads 8 --binary_args "-s Astar -u GNN -c -b --dataset_separated" --timeout 600
```
> Note that the `--dataset_separated` argument is used here (in the `--binary_args`), as the models are trained on instances with the same goal but different initial states.

#### Breadth-First Search
This command runs inference using the BFS heuristic, which is the baseline for comparison.
```console
python3 scripts/gnn_exp/bulk_coverage_run.py cmake-build-release-nn/bin/deep exp/gnn_exp/batch2/ --threads 8 --binary_args "-c -b" --timeout 600
```

##### Arguments
The arguments to this script are:
- the path to the deep executable (`cmake-build-release-nn/bin/deep` in the example)
- the path to the experiment folder (`exp/gnn_exp/batch2/` which is this folder)
- the number of threads to use to speed up the testing (`8` in the example)
- `--binary_args` option allows you to pass additional arguments to the deep executable, such as specifying the search algorithm and whether to enable GNN heuristics.
- `--timeout` specifies the maximum time in seconds for each instance to be solved (`600` seconds in the example).

### Advanced Setup
For more complex scenarios—such as training a single GNN model across multiple domains to improve generalization—we do not provide a dedicated script to avoid clutter.
However, you can adapt the existing scripts in the `/scripts` folder (in the project root) or create a custom folder structure to achieve this.
Models trained on merged domains are also included in this folder for convenience.
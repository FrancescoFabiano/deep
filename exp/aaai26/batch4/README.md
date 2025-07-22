# Experiment Batch 4: Scalability Testing

This folder contains the fourth batch of experiments for the project.

- This batch tests GNN-based heuristics in terms of scalability.
- We have generated instances with increasing goal length to see if GNN is able to generalize and allow the solver to scale.
- All models of interest are located in the `_models/{domain}` subfolder.
- All the results are stored in the `_results` subfolder.

---

## Usage

All commands should be run from the root directory of the repository.

### 1. Generate training data

This creates training datasets for each domain:

```bash
$ python scripts/aaai26/create_all_training_data.py exp/aaai26/batch4 --deep_exe cmake-build-release-nn/bin/deep
```
Replace --deep_exe with the path to your compiled deep binary.

### 2. Train GNN models
This trains one model per domain using the previously generated training data.
> This will overwrite existing models in the `_models` folder.


```bash
$ python scripts/aaai26/train_models.py exp/aaai26/batch4
```

### 3. Run evaluation and aggregate results
Run inference using the trained models and aggregate the results into the `_results` folder.


#### GNN heuristic
This command runs inference using the GNN heuristic with the appropriate model generated in the previous step.
```bash
$ python scripts/aaai26/aaai_coverage_run.py cmake-build-release-nn/bin/deep exp/aaai26/batch4/ --threads 8 --binary_args "-s Astar -u GNN -c -b" --timeout 600
```

#### Breadth-First Search
This command runs inference using the BFS heuristic, which is the baseline for comparison.
```bash
$ python scripts/aaai26/aaai_coverage_run.py cmake-build-release-nn/bin/deep exp/aaai26/batch4/ --threads 8 --binary_args "-c -b" --timeout 600
```

##### Arguments
The arguments to this script are:
- the path to the deep executable (`cmake-build-release-nn/bin/deep` in the example)
- the path to the experiment folder (`exp/aaai26/batch4/` which is this folder)
- the number of threads to use to speed up the testing (`8` in the example)
- `--binary_args` option allows you to pass additional arguments to the deep executable, such as specifying the search algorithm and whether to enable GNN heuristics.
- `--timeout` specifies the maximum time in seconds for each instance to be solved (`600` seconds in the example).

### Advanced Setup
For more complex scenarios—such as training a single GNN model across multiple domains to improve generalization—we do not provide a dedicated script to avoid clutter.
However, you can adapt the existing scripts in the `/scripts` folder (in the project root) or create a custom folder structure to achieve this.
Models trained on merged domains are also included in this folder for convenience.
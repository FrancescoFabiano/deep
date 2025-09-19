
## About the Project
This project develops an estimator that predicts how far a given state is from the goal within dynamic epistemic search
trees. Each search space is modeled as a set of Kripke structures.

## 1. Installation
1. Create a new python virtual environment (with 'python > 3.10'):
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   > The use of a virtual environment to keep dependencies isolated, is not mandatory, but helps avoid contaminating the global Python environment.
   > If you create one, be sure to activate it before using any of this repository’s functionality.
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Install setup:
   ```
   python setup.py install
   ```

## 2. Usage

### Training Data Generation
Before creating the dataloader and building the model, training data has to be generated.
An example of data generation is:
```console
../../cmake-build-release-nn/bin/deep ../../exp/example.txt --dataset
```

### Launch
Launch `__main__.py` and customize your process with the available argparse options:

```console
python __main__.py [OPTIONS]
```

### Data Options

* `--subset-train <str>...`
  Name(s) of problem subsets to use for training.
  **Default:** `[]`
* `--folder-raw-data <str>`
  Path where raw or intermediate data lives (or will be built).
  **Default:** `out/NN/Training`
* `--dir-save-data <str>`
  Directory into which processed data will be saved.
  **Default:** `data`
* `--unreachable-state-value <int>`
  Numeric value to assign unreachable states in your distance matrix.
  **Default:** `10^6`
* `--test-size <float>`
  Fraction of the dataset to reserve for testing (between 0.0 and 1.0).
  **Default:** `0.2`
* `--max-percentage-per-class <float>`
  Highest possible percentage for one class in the target variable..
  **Default:** `0.2`

### Model & Output Options

* `--model-name <str>`
  Name for the distance estimator model (used for filenames and logging).
  **Default:** `distance_estimator`
* `--normalization-constants-name <str>`
  Name of the normalization constants file for rescaling outputs.
  **Default:** `C`
* `--dir-save-model <str>`
  Directory into which trained models will be saved.
  **Default:** `models`
* `--experiment-name <str>`
  Identifier for this experiment's data and model outputs.
  **Default:** ``

### Training Parameters

* `--n-train-epochs <int>`
  Number of epochs for training the neural network.
  **Default:** `500`
* `--batch-size <int>`
  Batch size for the training loop.
  **Default:** `2048`
* `--seed <int>`
  Random seed for reproducibility of splits and initialization.
  **Default:** `42`

### Boolean Flags

Each of these accepts `true` or `false` (case-insensitive):

* `--build-data <bool>`
  Whether to (re)build the processed dataset before training.
  **Default:** `true`
* `--train <bool>`
  Whether to actually train the model after data is available.
  **Default:** `true`
* `--if-try-example <bool>`
  Run inference on example samples using PyTorch and ONNX.
  **Default:** `false`
* `--use-goal <bool>`
  Include goal information as part of the input features.
  **Default:** `false`
* `--use-depth <bool>`
  Include depth information (e.g., search depth) as a feature.
  **Default:** `false`
* `--verbose <bool>`
  Print detailed evaluation errors and progress logs.
  **Default:** `false`

### Feature Options

* `--kind-of-ordering <hash|map>`
  Strategy for ordering your state representations.
  **Default:** `hash`
* `--kind-of-data <merged|separated>`
  Whether to merge all data into a single set or keep splits separate.
  **Default:** `merged`

---

### Example Commands

Build data and train with goal and depth features enabled:

```console
python __main__.py \
  --build-data true \
  --train true \
  --use-goal true \
  --use-depth true \
  --n-train-epochs 300 \
```

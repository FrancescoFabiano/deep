# scripts

`trial.py` is the only entry point. One experiment = one folder under `exp/trials/`
holding a `trial.toml`; the script runs four stages over it and writes only inside
that folder.

```
python scripts/trial.py <stage> exp/trials/<trial> [--domains D ...] [--strategies S ...] [--models rl|gnn ...] [--dry-run]
```

| stage    | reads                          | writes                                   | what it does |
|----------|--------------------------------|------------------------------------------|--------------|
| `data`   | `instances/`, `split.csv`      | `data/<domain>/<STRAT>/<problem>/`       | `deep --dataset` once per (train problem, strategy) |
| `train`  | `data/` train-split trees      | `models/<domain>/*.onnx`                 | RL fringe ranker and/or GNN distance estimator, one model per F |
| `infer`  | `instances/`, `models/`        | `results/results.csv`                    | BFS and every model on every problem, one row per (problem, method, F) |
| `report` | `results/results.csv`          | `report/tables/*.tex *.csv`, `report/figures/*.png` | coverage, IQM nodes, IQM time, per-instance tables; per-domain figures |
| `all`    | | | the four in order |

Flags: `--domains` restricts any stage to some domains; `--strategies` restricts
`data` to some generation strategies; `--models` restricts `train` to `rl` and/or
`gnn`; `--dry-run` prints the planner/trainer
commands and runs nothing. Stages skip what is already on disk (a tree, a model,
a results row), so rerunning after a crash resumes.

Typical use, one domain, one model kind per command (the second command reuses the
trees and results of the first and only adds the missing models and rows):

```
python scripts/trial.py all exp/trials/basic --domains gossip --models rl
python scripts/trial.py all exp/trials/basic --domains gossip --models gnn
```

## A trial folder

```
exp/trials/<trial>/
  trial.toml                        the config (sections below)
  instances/act_lib.epddl           action library
  instances/<domain>/domain.epddl
  instances/<domain>/problems/*.epddl
  split.csv                         domain,problem,train|test  (written on first run)
  data/  models/  results/  report/ stage outputs (data, models, results are git-ignored)
```

To create a trial: copy `exp/trials/basic/trial.toml`, add domains under `instances/`
(one `domain.epddl` plus a `problems/` folder each, the library as `act_lib.epddl`),
run `data`. The split is computed once per domain from `[split]` -- problems sorted
by name, the first `train_pct`% train, the rest test (`shuffle_seed` randomises
instead) -- and frozen in `split.csv`. Edit that file to change the split by hand.

## trial.toml

| section       | keys | notes |
|---------------|------|-------|
| `[trial]`     | `deep_exe`, `workers`, `mem_gb` | binary path (repo-relative), parallel planner runs, RSS kill limit per run |
| `[split]`     | `train_pct`, `shuffle_seed` (optional) | |
| `[data]`      | `strategies`, `hfs_heuristic`, `depth`, `depth_overrides`, `seed`, `max_retries`, `discard_factor`, `max_generation`, `max_creation`, `dataset_type`, `timeout_s`, `generate_test` | strategies among BFS, DFS, S_DFS, HFS; `hfs_heuristic` only for HFS, `discard_factor` only for S_DFS; DFS/S_DFS retry with seed+1.. on failure, BFS/HFS are deterministic and never retry; `generate_test = true` also builds test trees (passed to the trainers as held-out diagnostics) |
| `[train]`     | `models`, `strategies`, `fringe_sizes`, `epochs`, `batch_size`, `seed` | `models` among `rl`, `gnn`; `strategies = "all"` or a subset of `[data].strategies` |
| `[train.rl]`, `[train.gnn]` | `extra = [...]` | flags forwarded verbatim to `lib/rl_handler/offline_main.py` / `lib/gnn_handler/__main__.py` |
| `[inference]` | `timeout_s`, `rl_exploration`, `rl_exploitation`, `gnn_searches` | exploration + exploitation < 100; `gnn_searches` among `Astar`, `HFS` |

The `[data]` keys that shape a tree are fingerprinted in `data/.dataspec`; changing
them requires an empty `data/` (the stage refuses to mix).

## Methods at inference

Which methods run is decided by the files in `models/<domain>/`, never by config:

| method        | planner flags | model file |
|---------------|---------------|------------|
| `BFS`         | `-s BFS` | none (always run) |
| `RL@F`        | `-s RL -u RL_H --RL_model ... --RL_fringe_size F` | `rl_F<F>.onnx` |
| `GNN_RL@F`    | same, with the GNN ranking the beam | `gnn_F<F>.onnx` |
| `GNN_Astar@F` / `GNN_HFS@F` | `-s Astar|HFS -u GNN --GNN_model ... --GNN_constant_file ...` | `gnn_F<F>_state.onnx` + `gnn_F<F>_state_C.txt` |

Every run adds `--act_lib`, `-b -c -r`.

## Code map

```
scripts/pipeline/
  config.py      trial.toml -> Config (paths, validation)       REQUIRED keys, checks in _validate
  instances.py   domains, problems, split.csv                   Instance(domain, problem, split, files)
  deep.py        planner argv, time/memory limits, -r parsing   base_argv(), run(), Result.field()
  data.py        stage 1                                        _argv() builds the --dataset flags
  train.py       stage 2                                        TRAINERS, FIXED_FLAGS, EXPORTS
  infer.py       stage 3                                        methods(), COLUMNS
  report.py      stage 4                                        tables, _write() (LaTeX), plot_*()
  tests/         config validation, split rule, trainer flag contract, report on a toy CSV
```

## How to extend

- **New generation flag**: add the key to `trial.toml [data]`, read it in `data._argv`;
  add it to `data.FINGERPRINT` if it changes the tree.
- **New trainer flag, once**: `[train.rl].extra` / `[train.gnn].extra`. Permanently:
  add it to the command in `train._train` and to the `sent` set in `tests/test_pipeline.py`.
- **New method / planner flags at inference**: append a `Method(name, F, flags)` in
  `infer.methods`; give it a color in `report.COLORS`. A new model kind also needs an
  entry in `train.TRAINERS` and `train.EXPORTS` (file the trainer writes, installed name).
- **New table or figure**: a function in `report.py` returning a DataFrame (tables go
  through `_write`, which emits `.tex` and `.csv`) or saving a PNG, called from `report.run`.
- **New results column**: add it to `infer.COLUMNS` and fill it in `infer._one`
  (`Result.field("<label>")` reads any `label: <int>` line of the planner's `-r` output).
- **Per-domain generation depth**: `[data].depth_overrides = { gossip = 30 }`.

Run the tests with `.venv/bin/python -m pytest scripts/pipeline/tests -q`.

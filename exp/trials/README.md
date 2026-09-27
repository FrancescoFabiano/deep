# Trials

One folder per experiment, one `trial.toml` per folder, four stages:

```
python scripts/trial.py data   exp/trials/<trial>    # generation trees   -> data/
python scripts/trial.py train  exp/trials/<trial>    # RL + GNN models    -> models/
python scripts/trial.py infer  exp/trials/<trial>    # BFS + every model  -> results/results.csv
python scripts/trial.py report exp/trials/<trial>    # tables + figures   -> report/
```

`all` runs the four in order. `--domains`, `--strategies` (data only) and `--dry-run`
apply to any stage. Stages skip what is already on disk, so a crashed run resumes.

Layout of a trial:

```
instances/act_lib.epddl, <domain>/domain.epddl, <domain>/problems/*.epddl   the frozen inputs
split.csv                    train/test per problem, written once from [split]
data/<domain>/<STRAT>/<problem>/                                             stage 1
models/<domain>/{rl,gnn}_F<F>.onnx, gnn_F<F>_state.onnx (+_C.txt)            stage 2
results/results.csv          one row per (problem, method, F)                stage 3
report/tables/*.tex *.csv, report/figures/*.png                              stage 4
```

Methods at inference: `BFS`; `RL@F` (RL search, RL model); `GNN_RL@F` (RL beam ranked
by the GNN); `GNN_Astar@F` / `GNN_HFS@F` (the per-state GNN as heuristic).

To make a new trial copy `basic/trial.toml`, drop instances under `instances/`
(one `domain.epddl` + `problems/` per domain, the action library as `act_lib.epddl`).

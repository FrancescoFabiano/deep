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
(one `domain.epddl` + `problems/` per domain, the action library as `act_lib.epddl`,
or per domain as `<domain>/act_lib.epddl` when the domains need different ones).

## The four trials

`basic`, `intermediate`, `hard` hold one IPC 2026 tier each (copied from
`exp/ipc2026-benchmarks/`); `all` holds every tier, with domain and problem names
prefixed by the tier (`basic-gossip/basic-gos-03-all`), because gossip and
blocks-world appear in every tier with the same problem names and the trainers key
trees by problem name. `all` sets `[train] pooled = true`: one RL and one GNN model
per F from the train trees of every domain, installed under `models/pooled/` and
used on every test problem.

Every `split.csv` comes from `scripts/make_splits.py` over `deep_solutions.csv`, the
baseline planner's results on the whole benchmark (1200 s CPU, 16 GB): a problem the
baseline solved is train, every other one (memout, timeout, no plan found) is test.
So test measures coverage beyond the baseline, and BFS solves none of it by
construction. Re-run the script if `deep_solutions.csv` changes.

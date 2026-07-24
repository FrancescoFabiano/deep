# rl_handler — Offline fringe-ranking RL for the epistemic planner

## About the Package

`rl_handler` trains a GNN that, at every expansion step of the C++ planner's
RL best-first search, scores the whole **fringe** (up to F candidate states
packed into one disconnected graph) and ranks which state to expand next. The
trainer is an offline **Double-DQN** over the fringe MDP simulated inside
search trees reconstructed from the planner's generation tables: reward −1
per expansion, 0 on goal ⇒ the return is −(#expansions), so **node economy is
optimized directly**. Design rationale: [DESIGN.md](DESIGN.md). Experimental
results: [../../REPORT_offline_rl.md](../../REPORT_offline_rl.md).

## The C++ consumer and contract (5 lines)

`SpaceSearcher` → `RL_BestFirst` → `FringeEvalRL` (ONNX Runtime) feeds, in
order: `node_features int64 [N]`, `edge_index int64 [2,E]`, `edge_attr int64
[E]`, `membership int64 [N]`, `mask uint8 [F]` (last) and reads `logits float
[F]` — higher logit = expand sooner; `F` must equal `--RL_fringe_size` and the
file must be named `_models/<domain>/frontier_policy_<F>.onnx`.
Export goes through `src/trainer.py::RLFrontierTrainer.to_onnx` **only** —
that path is the contract guarantee (byte-identical exports verified).

## Layout

```
offline_main.py        training entry point (Double-DQN)
offline_analysis.py    multi-seed plots + comparison vs the supervised champion
src/
  models/frontier_policy.py  deployed architecture (do not rename)
  trainer.py                 contract surface: save/load checkpoints, to_onnx
  offline/                   encoder (C++-exact packing), tree_env, replay,
                             dqn (trainer), plots
  common/no_pyg_loader.py    pydot reference loader (test parity)
tests/                 test_offline_encoder.py, test_offline_env.py
```

The legacy supervised trainer (softmax-over-frontier, the original producer
of the production `frontier_policy_<F>.onnx` baselines under
`exp/rl_exp/*/_models/`) was removed in this commit; recover it from git
history (commits `84ce260`/`783758c`) if the production frontier_policy
models ever need retraining. The exported baselines under `exp/` are data
and remain usable as planner-smoke references.

## Usage

### Primary entry point — `scripts/rl_exp/train_models.py` (run from repo root)

Per-experiment driver (twin of `scripts/gnn_exp/train_models.py`): enumerates
domains under `<exp_dir>/_models/<domain>/training_data/`, builds each domain's
train/val split, runs the trainer per (domain, seed), and — for a single-seed
run — installs the exported model at
`<exp_dir>/_models/<domain>/frontier_policy_<F>.onnx`, exactly where the eval
consumer `scripts/rl_exp/bulk_coverage_run.py` looks for it. Any
`offline_main.py` flag after the orchestration args (optionally past a `--`)
is forwarded verbatim:

```bash
# all domains, defaults, single seed (42) -> installs each domain's model
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged

# one domain, forwarding trainer flags through to offline_main.py
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged \
    --domains CC -- --frames 100000 --n-checkpoints 20 --fringe-size 32

# multi-seed (each seed its own subdir; pick one with offline_analysis.py)
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged \
    --domains CC --seeds 0 1 2 -- --frames 100000 --gamma 0.99
```

Then evaluate the freshly installed model:

```bash
python3 scripts/rl_exp/bulk_coverage_run.py ./cmake-build-release-nn/bin/deep \
    exp/rl_exp/batch0_merged/CC/Test 32 false "--search RL"
```

### Low-level interface — `offline_main.py` (from `lib/rl_handler`)

`train_models.py` shells out to this; call it directly for ad-hoc runs over
explicit CSVs:

```bash
python offline_main.py --frames 100000 --n-checkpoints 20 --seed 0 \
    --train-csv <a.csv> <b.csv> --val-csv <c.csv> \
    --dir-save-model ../../exp/rl_exp/offline_rl/seed0
```

Analyze a multi-seed run (plots + champion comparison + summary json):

```bash
python offline_analysis.py --runs-root ../../exp/rl_exp/offline_rl --seeds 0 1 2
```

Tests:

```bash
python tests/test_offline_encoder.py   # contract packing + parser parity
python tests/test_offline_env.py       # tree env vs oracle/BFS/random
```

## Training flags (`offline_main.py`)

| flag | default | effect |
|---|---|---|
| `--train-csv` / `--val-csv` | CC tables under `out/NN/Training` | generation tables; split is at instance level |
| `--frames` | `100000` | env steps (frames) |
| `--n-checkpoints` | `20` | evenly spaced eval checkpoints (val greedy #expansions, Spearman, score stats) |
| `--seed` | `42` | seeds env, replay, torch |
| `--gamma` | `0.99` | discount; with −1/step rewards Q ∈ [−1/(1−γ), 0] |
| `--epsilon-schedule` | `1.0,0.05,0.5` | linear `start,end,frac-of-frames` |
| `--fringe-size` | `32` | F: beam size, ONNX logits length, mask length |
| `--batch-size` | `64` | replay transitions per update |
| `--lr` | `1e-4` | Adam |
| `--replay-capacity` | `50000` | compact transitions (state ids, not tensors) |
| `--warmup` | `1000` | transitions before updates start |
| `--target-sync` | `1000` | hard target-net sync period (frames) |
| `--update-every` | `4` | env frames per gradient update (sets the fps) |
| `--eval-expansion-cap` | `2000` | censoring cap for greedy eval rollouts |
| `--max-grad-norm` | `1.0` | gradient clipping |
| `--dir-save-model` | required | run directory |
| `--device` | auto | `cuda` if available |
| `--export-onnx` / `--no-export-onnx` | on | export best checkpoints per the contract |

## Artifacts (per run directory)

```
exp/rl_exp/offline_rl/seedN/
  args.json, history.json            # config + training/checkpoint history
  best_by_expansions.pt              # best by val greedy #expansions (objective)
  best_by_spearman.pt                # best by val Spearman(score, -d*)
  last.pt
  frontier_policy_32_best_by_*.onnx  # contract exports
exp/rl_exp/offline_rl/plots/         # per-seed curves, IQM bands, scatters
exp/rl_exp/offline_rl/analysis_summary.json
```

The `[train] ...` / `[ckpt] ...` / `[export] ...` / `[done]` log lines are a
stable interface — monitors and `offline_analysis.py` grep for them. The tqdm
progress bar appears only on a TTY (auto-disabled under nohup/pipes).

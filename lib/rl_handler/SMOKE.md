# Post-launcher smoke plan

Everything below is prepared. **Do not run until `final_launcher.sh` finishes** —
generation and the smoke must not contend with it for the machine.

Check first:

```bash
pgrep -fa "final_launcher.sh|bulk_coverage_run|create_.*training_data" || echo "machine free"
```

---

## 1. Generate faithful data (one command)

```bash
cd /home/giovanni-briglia/CLionProjects/deep_forked
.venv/bin/python scripts/rl_exp/generate_faithful.py \
    exp/rl_exp/faithful/_models/CC/training_data \
    --domains CC --families CC_2_3_4 CC_2_2_3 --nice 15
```

Sets, per domain: `--dataset_discard_factor 0`, `--dataset_max_creation 50000`,
`--dataset_depth 25` (CC) / `40` (SC, SCRich). Writes `generation_manifest.json`,
then runs the faithfulness check and writes **`faithful_pool.json`**.

**Dry-run it first** (`--dry-run`) to see the exact `deep` argv per instance.

### What must be true before training

Read `faithful_pool.json`:

- `n_faithful > 0` — otherwise nothing trains. If **every** instance is excluded
  for `delta_root != known optimal`, the depth bound is cutting the solution path
  (raise it for that domain) or discarding is still on.
- `n_usable_for_fidelity > 0` — otherwise the fidelity gate cannot score anything
  and will report `NOT ARMED: no faithful instance reaches 20 expansions`.
  **Expected difficulty here**: the faithful tree is EASIER (restoring the optimal
  path dropped `CC_2_2_3__pl_4`'s BFS from 58 expansions to 6), so the small CC
  instances are likely to be flagged too-small. The fidelity cohort probably needs
  the larger `pl_N` (`CC_2_3_4__pl_7`: optimal 7, live BFS 9031).

---

## 2. The smoke (F=4, one within-config family)

```bash
cd lib/rl_handler
CUBLAS_WORKSPACE_CONFIG=:4096:8 nice -n 10 ../../.venv/bin/python offline_main.py \
    --exp-dir ../../exp/rl_exp/faithful \
    --domain CC \
    --fringe-sizes 4 \
    --model dqn \
    --kind-of-data merged \
    --context-mode mean_pool \
    --frames 2000 --n-checkpoints 5 \
    --eval-seeds 5 \
    --deep-exe ../../cmake-build-release-nn/bin/deep
```

`CUBLAS_WORKSPACE_CONFIG` must be in the environment **before** the CUDA context is
created; `determinism.py` sets it at import as a backstop, but the shell export is
the reliable path.

### Produces

| artifact | where |
|---|---|
| `telemetry.jsonl` | `exp/rl_exp/faithful/_models/CC/run_F4_dqn_merged/` |
| checkpoints (all) | `.../run_F4_dqn_merged/checkpoints/` |
| selected ONNX | `exp/rl_exp/faithful/_models/CC/frontier_policy_4.onnx` |
| selection sidecar | `.../frontier_policy_4.selection.json` |
| faithful pool | `.../CC/faithful_pool.json` |

### The three gates, in the sidecar and on stdout

1. **onnx_parity** — green already (55 contract tests).
2. **env_fidelity** — expected **PASS** on faithful data. On the shipped
   `discard 0.4` tables the env and planner were 229 vs 9031; on regenerated
   `discard 0` data they were 5 vs 7. **If it still FAILS on faithful data with
   instances above the 20-expansion floor, STOP** — that is a real modelling error
   in the env, and the gate earning its keep a third time.
3. **beats_baselines** — evaluated, warns rather than blocks. On HASHED data the
   pre-flight says the RL band is empty (0/15), so a loss here is *expected* and is
   not a pipeline failure — it is the shakedown data being what it is.

Exit code is non-zero if any gate fails.

---

## 3. The baseline, same harness, one flag

```bash
... offline_main.py --model two_head   # everything else identical
```

Same env, same seeds, same telemetry, same selection. Report each on its own before
comparing.

---

## 4. The four cells

```bash
for kind in merged separated; do
  for ctx in mean_pool none; do
    ... --kind-of-data $kind --context-mode $ctx
  done
done
```

`--kind-of-data separated` requires the tables generated with `--dataset_separated`
(the default in `generate_faithful.py`) **and** exports a 9-input ONNX;
`FringeEvalRL.tpp:413` rejects a merged 5-input model when the planner is given
`--dataset_separated`, so the pairing is not optional.

---

## Notes for whoever runs this

- **Multi-seed for any claim.** Determinism is solved (a seed reproduces exactly),
  but a seed is a model, not a dice roll — `--seed 0..4` and report the spread.
  Single-run A/Bs measured the RNG for a whole day.
- **This is a shakedown, not the experiment.** Its purpose is to prove the
  instrument runs end to end on faithful HASHED data. The RL-vs-baseline verdict on
  HASHED is already known (empty band); the point is that the moment BITMASK is
  enabled in `fringe_to_tensor_minimal`, the identical command reruns against the
  representation that could actually fill the band — no pipeline change, because
  nothing in trainer/telemetry/selection/export branches on the dataset type (ast-
  guarded).

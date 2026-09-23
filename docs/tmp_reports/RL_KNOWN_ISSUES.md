# RL pipeline — known issues / tech-debt ledger

Living list of "bad" training/model/infra things to fix. Status tags:
[FIXING NOW] in the current init+occupancy prompt · [PENDING] not yet addressed ·
[DECISION] needs Giovanni's call · [NOTE] landed/aware, confirm intended.
Committable as `lib/rl_handler/KNOWN_ISSUES.md` whenever you want it in-repo.

## Correctness & methodology
1. [FIXING NOW] Per-fringe model init not controlled. `model = _build_model()`
   runs before `OfflineDQNTrainer(..., seed=...)`, and the global torch RNG
   advances across the fringe loop, so F=32 and F=64 start from DIFFERENT inits
   → any fringe-size difference is confounded with init variance.
2. [PENDING] RNG hygiene generally loose — same root cause as #1; worth a small
   audit that a full run is reproducible end-to-end from `--seed`.
3. [DECISION] Equal frame budget across fringes. F=64 has a larger action space
   / harder credit assignment; at equal frames it may be under-trained. For the
   diversity round, either scale `--frames` with F or document the choice as a
   deliberate conservative control.

## Experimental design / data
4. [FIXING NOW] The fringe-size IV is INERT on current instances — live fringe
   peaks ~1 (CC) / ~11 (SC), never reaching 32. F=32 ≡ F=64 at deployment there.
   Occupancy logging + occupancy-based selection address this.
5. [PENDING] CC val instance is degenerate (near-linear, optimal = BFS = 12,
   max fringe = 1) → zero discriminative signal. Need higher-branching domains.
6. [PENDING] n = 1 val instance per domain → point estimates; can't separate
   signal from variance. Diversity round needs more val instances AND ≥3 seeds.

## Logging / UX
7. [PENDING] tqdm off-TTY guard broken. `disable=not sys.stderr.isatty()` is
   commented out and `import sys` was removed (committed in 12d15ea), so the
   training bar no longer auto-disables under the driver's capturing pipe — it
   emits newline-less `\r` updates → block-buffering / the "silent for minutes"
   symptom. Fix: uncomment the guard, re-add `import sys`.
8. [PENDING] Durable unbuffering not built. The `PYTHONUNBUFFERED=1`-in-`run_one`
   (+ `sys.stdout.flush()` after each `pbar.write` + a flushed `[prep]` line)
   fix was diagnosed but not implemented; currently relying on the env-var
   prefix as a manual workaround.

## Test / CI hygiene
9. [PENDING] `test_offline_encoder.py::test_onnx_compat` hard-requires a deployed
   `frontier_policy_32.onnx` fixture and tracebacks when it's absent (it's
   deleted in the working tree) instead of skipping. Bit two sessions in a row.
   Fix: commit a tiny DOT/ONNX fixture under `tests/`, or add a graceful skip.

## Model / export quirks
10. [PENDING] Exported ONNX has a FIXED scatter/output dim = its training fringe
    size, despite the logits being labeled symbolic `['F']`. A 32-model rejects
    N > 32 at inference (ScatterElements out-of-range). Consequence: you cannot
    cross-deploy a model across beam widths via ONNX — only via the fringe-
    agnostic `.pt`. Document; decide whether a truly dynamic export is wanted.
11. [DECISION] Two unresolved design `# TODO`s baked into `_build_model()`
    (committed): `dataset_type="HASHED"` and `use_goal_separate_input=False`.
    Resolve or document the intent.
12. [NOTE] `--batch-size` default silently changed 64 → 512 and committed inside
    the fringe-sweep commit (12d15ea). You'd been running 512, so likely
    intended — confirm, since it rode in with an unrelated change.

## Housekeeping
13. [PENDING] Stale `seed42/last.pt` artifact lingers from a pre-`last.pt`-removal
    run. Harmless clutter; delete when convenient.

---
Most-load-bearing for the science: #4 + #1 (this prompt) gate every fringe-size
claim, then #5/#6 (diversity round) gate any general claim. #7–#9 are quality-of-
life/CI and can be swept in a single cleanup pass. #10/#11 are model-design
decisions to make deliberately rather than by default.

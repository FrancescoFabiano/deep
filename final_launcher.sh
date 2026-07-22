#!/usr/bin/env bash
set -uo pipefail
# NOTE: no 'set -e' — steps are checked explicitly.

# ============================================================
#  Single-config experiment driver
#    data generation -> training -> evaluation pipeline
#
#  ONE configuration per run. No grids, no nested loops.
#  Edit the CONFIG block below, or override from the shell:
#
#      BATCH=batch2 ALGO=cql CTX=mean_pool ./run_batch.sh
#
#  Data reuse: point DATA_SOURCE at a previous batch to symlink
#  its training_data instead of regenerating (saves hours).
#  A .dataspec marker guards against reusing data generated with
#  incompatible flags (e.g. merged data in a separated run).
# ============================================================

source .venv/bin/activate
REPO="$(pwd)"

# ============================================================
#  CONFIG — the only part you normally touch
# ============================================================

BATCH="${BATCH:-batch1}"                  # -> exp/rl_exp/batch1

# --- what varies between runs (one value each) ---
ALGO="${ALGO:-dqn}"                       # dqn | cql
MODE="${MODE:-separated}"                 # separated | merged
STRICT="${STRICT:-yes}"                   # yes | no          (--strong_equality)
CTX="${CTX:-self_attention}"              # self_attention | mean_pool

# --- data handling ---
DRY_RUN="${DRY_RUN:-false}"               # true -> echo every stage command, run nothing
DATA_SOURCE="${DATA_SOURCE:-}"            # e.g. "batch1" -> symlink its training_data; "" -> generate
GENERATE_TEST_DATA="${GENERATE_TEST_DATA:-false}"
FORCE_REGEN="${FORCE_REGEN:-false}"       # true -> regenerate even if data exists

# --- what stays fixed across runs ---
FRINGE_SIZES="${FRINGE_SIZES:-4 8 16 32}"
DOMAINS="${DOMAINS:-CC}"                  # domains to symlink/check, e.g. "CC SC SCRich"

DEEP_EXE="cmake-build-release-nn/bin/deep"
BATCH_SIZE=64
# BUDGET IS EPOCHS, NOT STEPS. FRAMES=100000 meant 100k optimizer steps, which at
# batch 64 was 321 epochs at F=4 but only 44 at F=32 -- the dataset grows with F,
# so the same step count trained each F a different amount and no F-sweep was
# comparable. Steps are now derived per run: S = ceil(EPOCHS * |train_rows| / batch).
EPOCHS=100
N_CHECKPOINTS=20
# Draw distribution: capped (production default) | proportional (historical,
# reachable via SAMPLER=proportional for ablation). Capped draws an instance from
# min(c*(k), k*p_i) then a row within it, so no single instance dominates the
# gradient (on CC F=8, proportional gave pl_7 62% of rows / 3.2 effective
# instances; capped bounds that). The CLI/RunConfig default stays proportional --
# only this production launcher opts in -- so existing scripts and reruns are
# unaffected.
SAMPLER="${SAMPLER:-capped}"
SAMPLER_K="${SAMPLER_K:-4}"
CQL_ALPHA=1.0
# ---- FAITHFUL GENERATION ----
# The discard is BIASED, not uniform: its probability rises with depth and gains
# +0.2 immediately after a goal is found (TrainingDataset.tpp:714-730), so it
# preferentially deletes the SHALLOW GOALS that make an instance easy, while
# writing the discarded states to the CSV as childless leaves. Measured on
# CC_2_2_3__pl_4 (planner BFS true optimal = 4):
#     discard 0.4 -> delta_root 10, sterile 30.0%   (the optimal path is ABSENT)
#     discard 0   -> delta_root  4, sterile  2.9%
# Passed EXPLICITLY: create_all_training_data.py is shared with gnn_exp and its
# default must not move under that pipeline's feet.
DISCARD_FACTOR="${DISCARD_FACTOR:-0}"

# A depth bound that CONTAINS the optimal keeps the whole solution path while cutting
# the tree where it no longer matters. CC optimals are ~4-7; SC needs the depth.
DEPTH_MAP="${DEPTH_MAP:-CC:25,SC:40,SCRich:40,Grapevine:25,Assemble:25,CoinBox:25}"
# No MAX_DEPTH fallback: DEPTH_MAP is AUTHORITATIVE. A domain absent from it fails
# loudly rather than silently inheriting 40 -- which is the UNFAITHFUL setting for
# CC and is what blew past the ceiling and poisoned the tree.
#
# DEPTH ALONE DOES NOT BOUND THE TREE. TrainingDataset.tpp:677 poisons on EITHER
# ceiling -- m_current_nodes >= max_generation (VISITS) OR m_added_to_dataset >=
# max_creation (WRITES) -- and past it non-goals are dropped AND their parents
# inherit 1e6 = unreachable, WITHOUT recursing, so goals below them are never
# reached. Measured 2026-07-15: CC_2_3_4__pl_7 at depth 25 estimates 1.15e40 nodes
# -> "Decision: using SPARSE DFS" -> the DFS burns the VISIT budget in the deep
# region, records a goal at depth 22, and never reaches the true optimal at 7
# (delta_root 22 vs 7; strict BFS confirms 7).
#
# So BOTH ceilings are set HERE and fingerprinted, not left at a C++ default nobody
# sees. The VISIT ceiling is the binding one and it is the one that was invisible.
# Beware: a table stranded UNDER max_creation is the SYMPTOM of the visit ceiling
# biting (it stops all further additions), not evidence that no ceiling bit.
TRAIN_MAX_CREATION=50000
TEST_MAX_CREATION=5000
# 100000 is the historical C++ default, set explicitly so it is visible and
# fingerprinted. SETTLED 2026-07-15: raising it does NOT buy a shortest-path tree.
# On CC_2_3_4__pl_7 at depth 25, 100k -> 250k bought 4.2x the goals and moved
# delta_root by ZERO (22 both), while poisoned_frac FELL (0.0056 -> 0.0042) -- there
# is no starvation signature. Budget was never the lever; the generator's memo is
# depth-blind, so its distances are DFS-discovery depths (see usability.py).
# Also: do not raise this far. The DOT file counter increments per VISIT and
# overflows at 999,999 (std::length_error), and dots-per-visit is depth-dependent
# (~3.3x at depth 25, ~1580x at depth 9). Watch the file count, not a ratio.
TRAIN_MAX_GENERATION="${TRAIN_MAX_GENERATION:-100000}"
TEST_MAX_GENERATION="${TEST_MAX_GENERATION:-100000}"
# The generation seed is LOAD-BEARING: the tree is a seed-dependent DFS sample, so a
# different seed is different data. Fingerprinted for exactly that reason.
SEED="${SEED:-42}"

# The gamma=0.99 stability trio (--target-tau/--lr-schedule/--lr-min) is RETIRED.
# The objective is undiscounted (gamma=1): every policy is proper (completeness
# proposition), so this is a stochastic shortest path problem. The trainer uses a
# hard target sync + a |Q| > 3x cap divergence alarm and honours no LR schedule, so
# those flags would be dead surface -- a flag the entry point ignores is the same
# latent bug as one it rejects, it just fails silently.
STABILITY_FLAGS=""

# ============================================================
#  DERIVED — nothing below needs editing
# ============================================================

BATCH_DIR="exp/rl_exp/${BATCH}"

# ---- generation flags (these define data compatibility) ----
GEN_FLAG=""
[[ "$MODE"   == "separated" ]] && GEN_FLAG+=" --no_goal"
[[ "$STRICT" == "yes"       ]] && GEN_FLAG+=" --strong_equality"
GEN_FLAG="${GEN_FLAG# }"

# Data fingerprint: any run reusing this data must match it.
# discard=, depth_map=, max_generation= and seed= are NOT optional. Without discard=,
# a discard=0 run would happily symlink a discard=0.4 batch, pass this check, train on
# trees whose optimal path was deleted, and report clean self-consistent numbers --
# the worst failure mode available here. Without depth_map=, a CC-25 run would reuse
# CC-40 data. Without max_generation=, a run at one VISIT budget would reuse a batch
# generated at another. Without seed=, a run would reuse a DIFFERENT SAMPLE of the
# state space: the tree is a seed-dependent DFS walk, and on CC_2_2_3__pl_4 the seed
# alone moves delta_root 14/6/7 (seeds 42/43/44) on a fixed true optimal of 4.
DATASPEC="mode=${MODE};strict=${STRICT};discard=${DISCARD_FACTOR};depth_map=${DEPTH_MAP};max_creation=${TRAIN_MAX_CREATION};max_generation=${TRAIN_MAX_GENERATION};seed=${SEED}"

# ---- training flags ----
TRAIN_FLAG=""
PIPE_FLAG=""
if [[ "$MODE" == "separated" ]]; then
    TRAIN_FLAG="--no_goal"
    PIPE_FLAG="--separated"
fi

# --no-pad-closed and --stratified-replay are RETIRED. Padding the beam with CLOSED
# states gave the model actions the planner can never take -- a train/deploy mismatch
# the fidelity gate would flag, i.e. a defect rather than a feature.
TRAIN_EXTRA="${STABILITY_FLAGS}"
TRAIN_EXTRA+=" --context-mode ${CTX}"
[[ "$ALGO"  == "cql" ]] && TRAIN_EXTRA+=" --cql-alpha ${CQL_ALPHA}"
# ALLOW_CROSS_CONFIG=true opts into a GLOBAL (multi-configuration) model. Default
# false: the guard hard-raises, because node ids are fluent-set hashes and two
# configurations share 0.0% of them. On HASHED this run is a STRUCTURE-ONLY FLOOR --
# the node channel contributes nothing, so any RL-vs-baseline gap comes from topology
# and edge labels alone. Stamped exploratory=true in the selection sidecar.
[[ "${ALLOW_CROSS_CONFIG:-false}" == "true" ]] && TRAIN_EXTRA+=" --allow-cross-config"
TRAIN_EXTRA="${TRAIN_EXTRA# }"

# ============================================================
#  BANNER
# ============================================================
echo ""
echo "============================================================"
echo "  BATCH:   ${BATCH_DIR}"
echo "  algo=${ALGO}  mode=${MODE}  strict=${STRICT}"
echo "  ctx=${CTX}"
echo "  fringes: ${FRINGE_SIZES}"
echo "  gen  flags: ${GEN_FLAG:-<none>}  discard=${DISCARD_FACTOR}  depth_map=${DEPTH_MAP}"
echo "  train flags: ${TRAIN_FLAG} ${TRAIN_EXTRA}"
echo "  data source: ${DATA_SOURCE:-<generate here>}"
echo "============================================================"

[[ "${DRY_RUN}" == "true" ]] || mkdir -p "${BATCH_DIR}"

fail() { echo "[FAIL] ${BATCH_DIR} — $1"; rm -rf out; exit 1; }

# Every stage goes through this, so DRY_RUN proves the wiring without touching the
# machine (no generation, no GPU, no `deep` calls).
run_stage() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[DRY] $*"
        return 0
    fi
    "$@"
}

# ============================================================
#  1. TRAINING DATA  (reuse | skip | generate)
# ============================================================

data_dir_for() { echo "${BATCH_DIR}/_models/$1/training_data"; }

if [[ -n "${DATA_SOURCE}" ]]; then
    # ---- reuse: symlink from another batch ----
    SRC_DIR="exp/rl_exp/${DATA_SOURCE}"
    SRC_SPEC="${SRC_DIR}/.dataspec"

    [[ -d "${SRC_DIR}" ]] || fail "DATA_SOURCE '${SRC_DIR}' does not exist"

    if [[ -f "${SRC_SPEC}" ]]; then
        HAVE="$(cat "${SRC_SPEC}")"
        # Compared by the TESTED module (lib/rl_handler/src/offline/dataspec.py), not
        # by string equality here: a legacy spec WITHOUT discard= must be REFUSED
        # (old batches were generated at 0.4), which a bare != cannot express.
        WHY="$(python3 - "${HAVE}" "${DATASPEC}" <<'PYEOF'
import sys
sys.path.insert(0, "lib/rl_handler")
from src.offline.dataspec import compatible
ok, why = compatible(sys.argv[1], sys.argv[2])
print("" if ok else why)
PYEOF
)"
        if [[ -n "${WHY}" ]]; then
            echo "[!] data spec mismatch: ${WHY}"
            echo "    source: ${HAVE}"
            echo "    wanted: ${DATASPEC}"
            fail "incompatible DATA_SOURCE (regenerate, or fix MODE/STRICT/DISCARD)"
        fi
    else
        # No spec => provenance unknown => almost certainly a discard=0.4 batch.
        # Linking anyway would silently reintroduce the artifact.
        fail "${SRC_SPEC} missing — refusing to reuse data of unknown provenance"
    fi

    for dom in ${DOMAINS}; do
        SRC="${REPO}/${SRC_DIR}/_models/${dom}/training_data"
        TGT="$(data_dir_for "${dom}")"
        [[ -d "${SRC}" ]] || fail "no training_data for domain ${dom} in ${SRC_DIR}"
        mkdir -p "$(dirname "${TGT}")"
        [[ -L "${TGT}" ]] && rm "${TGT}"
        [[ -d "${TGT}" && ! -L "${TGT}" ]] && fail "${TGT} is a real dir — refusing to clobber"
        ln -sfn "${SRC}" "${TGT}"
        echo "[1/3] linked ${dom}: ${TGT} -> ${SRC}"
    done
    echo "${DATASPEC}" > "${BATCH_DIR}/.dataspec"

else
    # ---- generate in place (unless already present) ----
    NEED_GEN=false
    for dom in ${DOMAINS}; do
        D="$(data_dir_for "${dom}")"
        if [[ ! -d "${D}" ]] || [[ -z "$(ls -A "${D}" 2>/dev/null)" ]]; then
            NEED_GEN=true
        fi
    done
    [[ "${FORCE_REGEN}" == "true" ]] && NEED_GEN=true

    if [[ "${NEED_GEN}" == "true" ]]; then
        echo "[1/3] generating training data ..."
        run_stage python3 scripts/gnn_exp/create_all_training_data.py "${BATCH_DIR}" \
            --deep_exe "${DEEP_EXE}" \
            --depth-map "${DEPTH_MAP}" \
            --discard_factor "${DISCARD_FACTOR}" \
            --dataset-max-creation "${TRAIN_MAX_CREATION}" \
            --dataset-max-generation "${TRAIN_MAX_GENERATION}" \
            --seed "${SEED}" \
            ${GEN_FLAG} \
            || fail "step 1 (generate train data)"
        [[ "${DRY_RUN}" == "true" ]] || { echo "${DATASPEC}" > "${BATCH_DIR}/.dataspec"; rm -rf out; }
    else
        echo "[1/3] training data already present — skipping generation."
        # refresh spec if absent
        [[ -f "${BATCH_DIR}/.dataspec" ]] || echo "${DATASPEC}" > "${BATCH_DIR}/.dataspec"
    fi

    # ---- optional test data ----
    if [[ "${GENERATE_TEST_DATA}" == "true" ]]; then
        echo "[1b] generating test data ..."
        run_stage python3 scripts/gnn_exp/create_all_training_data.py "${BATCH_DIR}" \
            --deep_exe "${DEEP_EXE}" \
            --dataset-name "test_data" \
            --training-folder "Test" \
            --depth-map "${DEPTH_MAP}" \
            --discard_factor "${DISCARD_FACTOR}" \
            --dataset-max-creation "${TEST_MAX_CREATION}" \
            --dataset-max-generation "${TEST_MAX_GENERATION}" \
            --seed "${SEED}" \
            ${GEN_FLAG} \
            || fail "step 1b (generate test data)"
        rm -rf out
    fi
fi

# ============================================================
#  1b. PATH STALENESS CHECK  (before anything reads a .dot)
# ============================================================
# The generation tables store the .dot paths as they existed WHEN GENERATED, and
# nothing rewrites them when a tree is moved or a domain dir renamed. batch2's
# Grapevine_5__pl_4 recorded `batch2/_models/Grapevine/...` (the dir it was
# generated into, later moved to batch1/_models/Grapevine and hardlinked into
# batch2/_models/CC-Grapevine) and training died 12 CSVs deep with a bare
# FileNotFoundError. Worse, the surviving CSVs resolve only because their SOURCE
# batch still exists on disk -- a hardlinked batch is not self-contained, so
# deleting the donor silently breaks it. Fail here instead, with the prefix named.
echo "[1b] path staleness check ..."
if [[ "${DRY_RUN}" == "true" ]]; then echo "[DRY] path staleness check -> verifies every .dot prefix in the generation tables resolves"; else
python3 - "${BATCH_DIR}" ${DOMAINS} <<'PYEOF' || fail "step 1b (path staleness check)"
import csv, sys
from pathlib import Path

batch_dir, domains = sys.argv[1], sys.argv[2:]
rc = 0
for dom in domains:
    root = Path(batch_dir) / "_models" / dom / "training_data"
    broken = []
    for csv_path in sorted(root.glob("*/*_depth_*.csv")):
        with csv_path.open(newline="") as fh:
            rdr = csv.DictReader(fh)
            cols = [c for c in (rdr.fieldnames or []) if "Path" in c or c == "Goal"]
            dirs = {}   # recorded parent dir -> one example file
            for row in rdr:
                for c in cols:
                    v = (row.get(c) or "").strip()
                    # `init.dot` is the root's predecessor sentinel: it is never
                    # written to disk and never opened (tree.py builds the parent
                    # map with .get()), so it is not evidence of a stale prefix.
                    if v.endswith(".dot") and Path(v).name != "init.dot":
                        dirs.setdefault(str(Path(v).parent), v)
        for d, example in sorted(dirs.items()):
            if not Path(example).exists():
                broken.append((csv_path.parent.name, example))
    if broken:
        rc = 1
        print(f"[1b] {dom}: {len(broken)} recorded path prefix(es) do not resolve --")
        for inst, example in broken:
            print(f"       {inst}: {example}")
        print("     These tables were generated into a directory that has since moved or\n"
              "     been renamed; nothing rewrites the path column on a move. Fix by\n"
              "     rewriting the column IN PLACE -- open(p,'w'), NOT sed -i, so hardlinked\n"
              "     copies of the same table in other batches are corrected too.")
    # A resolving-but-foreign prefix is legal (hardlinked batches) yet fragile:
    # report the donor trees once per domain so the dependency is visible.
    donors = set()
    for csv_path in sorted(root.glob("*/*_depth_*.csv")):
        with csv_path.open(newline="") as fh:
            rdr = csv.DictReader(fh)
            first = next(rdr, None)
        if not first:
            continue
        v = (first.get("File Path") or "").strip()
        if v.endswith(".dot") and not Path(v).resolve().is_relative_to(csv_path.parent.resolve()):
            donors.add(str(Path(v).parents[3]))
    if donors:
        print(f"[1b] {dom}: WARNING tables point at {len(donors)} foreign tree(s): "
              f"{', '.join(sorted(donors))}\n"
              f"     This batch is NOT self-contained -- deleting a donor breaks it.")
sys.exit(rc)
PYEOF
fi

# ============================================================
#  1c. FAITHFULNESS GATE  (between generation and training)
# ============================================================
# Nothing trains on data that fails this. An unfaithful tree is a WRONG PROBLEM,
# not noisy data: the discard=0.4 tables have their shallow goals deleted
# (delta_root 10 vs a true optimal of 4 on CC_2_2_3__pl_4), so a model trained on
# them produces clean, self-consistent, meaningless numbers. Runs here so the gate
# applies however the launcher is invoked.
echo "[1c] usability gate ..."
if [[ "${DRY_RUN}" == "true" ]]; then echo "[DRY] usability gate -> ${BATCH_DIR}/_models/<dom>/usable_pool.json (refuses training if n_usable==0)"; else
REQUIRE_SCORABLE="${REQUIRE_SCORABLE:-false}" \
python3 - "${BATCH_DIR}" "${MODE}" ${DOMAINS} <<'PYEOF' || fail "step 1c (usability gate)"
import os
import sys
from pathlib import Path
sys.path.insert(0, "lib/rl_handler")
from src.offline.tree import load_tree_instance, partition_solvable
from src.offline.usability import build_usable_pool

# Scorability is a FLAG by default: a metric-blind instance still contributes
# gradient, and excluding it would silently change the training pool of every
# existing batch. REQUIRE_SCORABLE=true promotes it to an exclusion -- use it for
# runs whose whole point is a held-out ranking claim, and expect a smaller pool.
require_scorable = os.environ.get("REQUIRE_SCORABLE", "false") == "true"
batch_dir, mode, domains = sys.argv[1], sys.argv[2], sys.argv[3:]
rc = 0
for dom in domains:
    root = Path(batch_dir) / "_models" / dom / "training_data"
    csvs = sorted(root.glob("*/*_depth_*.csv"))
    if not csvs:
        print(f"[1c] {dom}: no generation tables under {root}"); rc = 1; continue
    insts = [load_tree_instance(p, name=p.parent.name, kind_of_data=mode) for p in csvs]
    ok, _ = partition_solvable(insts)
    pool = build_usable_pool(ok, out_path=Path(batch_dir) / "_models" / dom / "usable_pool.json",
                             require_scorable=require_scorable)
    if pool["n_usable"] == 0:
        print(f"[1c] {dom}: NO USABLE INSTANCE -- refusing to train."); rc = 1
    if pool["n_usable_for_ranking"] == 0 and pool["n_usable"] > 0:
        print(f"[1c] {dom}: WARNING every usable instance is METRIC-BLIND; held-out "
              f"ranking metrics and checkpoint selection will be vacuous. Training "
              f"still runs (gradient is fine) -- do not read the ranking numbers.")
    if pool["n_usable_for_fidelity"] == 0:
        print(f"[1c] {dom}: WARNING no instance reaches 20 expansions; the "
              f"env-fidelity gate cannot score anything and will report NOT ARMED.")
sys.exit(rc)
PYEOF
fi

# ============================================================
#  2. TRAIN
# ============================================================
echo "[2/3] training (model=${ALGO}, F=${FRINGE_SIZES}, batch=${BATCH_SIZE}, epochs=${EPOCHS}, sampler=${SAMPLER}${SAMPLER:+ k=${SAMPLER_K}}) ..."
# DOMAINS is forwarded: without it train_models.py falls back to find_domains(),
# i.e. EVERY subdir of _models with a training_data/ -- so DOMAINS="CoinBox" on
# batch1_1 gated CoinBox and then trained all five domains, including Assemble
# (n_usable=0, ungated because 1c only checks the named domains, so it raised deep
# inside load_pool). Single-domain batches like batch2 masked this.
# (DOMAINS is never empty: the `:-CC` default above substitutes on unset AND empty.)
run_stage python3 scripts/rl_exp/train_models.py "${BATCH_DIR}" \
    --domains ${DOMAINS} \
    --fringe-sizes ${FRINGE_SIZES} \
    --model "${ALGO}" \
    ${TRAIN_FLAG} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --sampler ${SAMPLER} \
    --sampler-k ${SAMPLER_K} \
    --n-checkpoints ${N_CHECKPOINTS} \
    ${TRAIN_EXTRA} \
    || fail "step 2 (train)"

# ============================================================
#  3. EVALUATE
# ============================================================
echo "[3/3] evaluation pipeline ..."
run_stage python3 scripts/rl_exp/pipeline.py "${BATCH_DIR}" \
    --domains ${DOMAINS} \
    ${PIPE_FLAG} \
    || fail "step 3 (pipeline)"

[[ "${DRY_RUN}" == "true" ]] || rm -rf out

echo ""
echo "============================================================"
echo "  [DONE] ${BATCH_DIR}"
echo "============================================================"

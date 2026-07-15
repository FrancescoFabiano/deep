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
PAD="${PAD:-no}"                          # yes | no          (pad fringe with closed nodes)
STRAT="${STRAT:-no}"                      # yes | no          (d*-stratified replay)
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
FRAMES=100000
N_CHECKPOINTS=20
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

# Depth is what keeps generation under the ceiling -- do NOT raise the ceiling.
# --dataset_max_creation is a hard stop that POISONS (TrainingDataset.tpp:677-684):
# past it non-goals are dropped AND their parents inherit 1e6 = unreachable. A
# depth bound that CONTAINS the optimal keeps the whole solution path while cutting
# the tree where it no longer matters. CC optimals are ~4-7; SC needs the depth.
DEPTH_MAP="${DEPTH_MAP:-CC:25,SC:40,SCRich:40}"
MAX_DEPTH=40                              # fallback for domains absent from DEPTH_MAP
TRAIN_MAX_CREATION=50000
TEST_MAX_CREATION=5000

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
# discard= and depth_map= are NOT optional. Without discard=, a faithful run would
# happily symlink a discard=0.4 batch, pass this check, train on trees whose optimal
# path was deleted, and report clean self-consistent numbers -- the worst failure
# mode available here. Without depth_map=, a CC-25 run would reuse CC-40 data.
DATASPEC="mode=${MODE};strict=${STRICT};discard=${DISCARD_FACTOR};depth_map=${DEPTH_MAP};max_creation=${TRAIN_MAX_CREATION}"

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
TRAIN_EXTRA="${TRAIN_EXTRA# }"

# ============================================================
#  BANNER
# ============================================================
echo ""
echo "============================================================"
echo "  BATCH:   ${BATCH_DIR}"
echo "  algo=${ALGO}  mode=${MODE}  strict=${STRICT}"
echo "  pad=${PAD}  strat=${STRAT}  ctx=${CTX}"
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
            --depth "${MAX_DEPTH}" \
            --depth-map "${DEPTH_MAP}" \
            --discard_factor "${DISCARD_FACTOR}" \
            --dataset-max-creation "${TRAIN_MAX_CREATION}" \
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
            --depth "${MAX_DEPTH}" \
            --depth-map "${DEPTH_MAP}" \
            --discard_factor "${DISCARD_FACTOR}" \
            --dataset-max-creation "${TEST_MAX_CREATION}" \
            ${GEN_FLAG} \
            || fail "step 1b (generate test data)"
        rm -rf out
    fi
fi

# ============================================================
#  1c. FAITHFULNESS GATE  (between generation and training)
# ============================================================
# Nothing trains on data that fails this. An unfaithful tree is a WRONG PROBLEM,
# not noisy data: the discard=0.4 tables have their shallow goals deleted
# (delta_root 10 vs a true optimal of 4 on CC_2_2_3__pl_4), so a model trained on
# them produces clean, self-consistent, meaningless numbers. Runs here so the gate
# applies however the launcher is invoked.
echo "[1c] faithfulness gate ..."
if [[ "${DRY_RUN}" == "true" ]]; then echo "[DRY] faithfulness gate -> ${BATCH_DIR}/_models/<dom>/faithful_pool.json (refuses training if n_faithful==0)"; else
python3 - "${BATCH_DIR}" "${MODE}" ${DOMAINS} <<'PYEOF' || fail "step 1c (faithfulness gate)"
import sys
from pathlib import Path
sys.path.insert(0, "lib/rl_handler")
from src.offline.faithfulness import build_faithful_pool
from src.offline.tree import load_tree_instance, partition_solvable

batch_dir, mode, domains = sys.argv[1], sys.argv[2], sys.argv[3:]
rc = 0
for dom in domains:
    root = Path(batch_dir) / "_models" / dom / "training_data"
    csvs = sorted(root.glob("*/*_depth_*.csv"))
    if not csvs:
        print(f"[1c] {dom}: no generation tables under {root}"); rc = 1; continue
    insts = [load_tree_instance(p, name=p.parent.name, kind_of_data=mode) for p in csvs]
    ok, _ = partition_solvable(insts)
    pool = build_faithful_pool(ok, out_path=Path(batch_dir) / "_models" / dom / "faithful_pool.json")
    if pool["n_faithful"] == 0:
        print(f"[1c] {dom}: NO FAITHFUL INSTANCE -- refusing to train."); rc = 1
    if pool["n_usable_for_fidelity"] == 0:
        print(f"[1c] {dom}: WARNING no instance reaches 20 expansions; the "
              f"env-fidelity gate cannot score anything and will report NOT ARMED.")
sys.exit(rc)
PYEOF
fi

# ============================================================
#  2. TRAIN
# ============================================================
echo "[2/3] training (model=${ALGO}, F=${FRINGE_SIZES}, batch=${BATCH_SIZE}, frames=${FRAMES}) ..."
run_stage python3 scripts/rl_exp/train_models.py "${BATCH_DIR}" \
    --fringe-sizes ${FRINGE_SIZES} \
    --model "${ALGO}" \
    ${TRAIN_FLAG} \
    --batch-size ${BATCH_SIZE} \
    --frames ${FRAMES} \
    --n-checkpoints ${N_CHECKPOINTS} \
    ${TRAIN_EXTRA} \
    || fail "step 2 (train)"

# ============================================================
#  3. EVALUATE
# ============================================================
echo "[3/3] evaluation pipeline ..."
run_stage python3 scripts/rl_exp/pipeline.py "${BATCH_DIR}" \
    ${PIPE_FLAG} \
    || fail "step 3 (pipeline)"

[[ "${DRY_RUN}" == "true" ]] || rm -rf out

echo ""
echo "============================================================"
echo "  [DONE] ${BATCH_DIR}"
echo "============================================================"

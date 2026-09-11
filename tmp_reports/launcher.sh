#!/usr/bin/env bash
# ============================================================
#  launcher.sh -- data generation -> training -> planner inference
#
#  Usage:  ./launcher.sh [ACTION] [BATCH] [DOMAIN ...]
#
#    ACTION  check | gen | train | infer | all      (default: all)
#    BATCH   folder name under exp/rl_exp/          (default: batch2)
#    DOMAIN  domain folder(s) inside the batch; omit or 'all' = every
#            subfolder that has a Training/ dir     (default: all)
#
#      ./launcher.sh                      # batch2, every domain (= Mix): check+gen+train+infer
#      ./launcher.sh check                # preflight only, runs nothing heavy
#      ./launcher.sh all batch1 CC        # batch1, domain CC only
#      ./launcher.sh all batch1           # batch1, every domain (CC SC SCRich ...)
#      ./launcher.sh train batch1 CC SC   # training only, two domains
#
#  The same three settings are also env vars (BATCH, DOMAINS), as is every
#  other CONFIG value below:
#      BATCH=batch1 DOMAINS="CC SC" EPOCHS=50 ./launcher.sh train
#
#  Stage logs: exp/rl_exp/<BATCH>/_models/launcher_logs/<stage>_<timestamp>.log
# ============================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO="$(pwd)"

# ============================================================
#  CONFIG
# ============================================================
# Positional args override the env vars: ./launcher.sh [ACTION] [BATCH] [DOMAIN ...]
ACTION="${1:-all}"
[[ $# -ge 2 ]] && BATCH="$2"
[[ $# -ge 3 ]] && DOMAINS="${*:3}"
BATCH="${BATCH:-batch2}"                     # -> exp/rl_exp/<BATCH>
DOMAINS="${DOMAINS:-all}"                    # domain folder(s) under the batch, space-separated; 'all' = discover
DEEP_EXE="${DEEP_EXE:-cmake-build-release-nn/bin/deep}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"   # the venv carries torch; system python3 does not
# The stage scripts spawn sub-steps with a bare `python3` (generator workers, the
# pipeline's aggregate/analyze/plot chain), so that name must resolve to the same
# interpreter: put its bin dir first on PATH.
export PATH="$(dirname "$PYTHON"):$PATH"

# --- state representation (must agree across all three stages) ---
MODE="${MODE:-separated}"                    # separated (--no_goal / --separated) | merged
STRICT="${STRICT:-yes}"                      # yes -> --strong_equality

# --- generation ---
STRATEGIES="${STRATEGIES:-all}"              # BFS DFS S_DFS HFS | all
HEURISTICS="${HEURISTICS:-SUBGOALS}"         # HFS heuristic
# Depth is PER DOMAIN. Keys are domain FOLDER names (CC, SC, SCRich) or, for a
# mixed folder such as Mix/, the instance-name domain TOKEN (CC_2_2_3__pl_3 -> CC,
# SC_R_10_10__pl_2 -> SC_R). A token without an entry fails loudly before any
# tree is generated -- never inherits a default.
DEPTH_MAP="${DEPTH_MAP:-CC:25,SC:40,SC_R:40,SCRich:40,Grapevine:25,Assemble:25,CoinBox:25,Coin_in_the_Box:25}"
MAX_CREATION="${MAX_CREATION:-50000}"        # --dataset-max-creation (WRITE cap)
MAX_GENERATION="${MAX_GENERATION:-50000}"    # --dataset-max-generation (VISIT cap, the binding one)
SEED="${SEED:-42}"
GENERATE_TEST_DATA="${GENERATE_TEST_DATA:-false}"   # true -> also build _models/<dom>/test_data from Test/
FORCE_REGEN="${FORCE_REGEN:-false}"          # true -> regenerate even if training_data exists

# --- training ---
MODEL="${MODEL:-dqn}"                        # dqn | cql
FRINGE_SIZES="${FRINGE_SIZES:-4 8 16 32}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-100}"
N_CHECKPOINTS="${N_CHECKPOINTS:-20}"
CTX="${CTX:-self_attention}"                 # self_attention | mean_pool
ALLOW_CROSS_CONFIG="${ALLOW_CROSS_CONFIG:-true}"   # a mixed folder has several configs -> required
TRAIN_EXTRA="${TRAIN_EXTRA:-}"               # anything else forwarded verbatim to offline_main.py

# ============================================================
#  DERIVED
# ============================================================
BATCH_DIR="exp/rl_exp/${BATCH}"
LOG_DIR="${BATCH_DIR}/_models/launcher_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"

GEN_FLAGS=(); TRAIN_FLAGS=(); PIPE_FLAGS=()
if [[ "$MODE" == "separated" ]]; then
    GEN_FLAGS+=(--no_goal); TRAIN_FLAGS+=(--no_goal); PIPE_FLAGS+=(--separated)
elif [[ "$MODE" != "merged" ]]; then
    echo "[launcher] MODE must be separated|merged, got '$MODE'" >&2; exit 2
fi
[[ "$STRICT" == "yes" ]] && GEN_FLAGS+=(--strong_equality)
[[ "$ALLOW_CROSS_CONFIG" == "true" ]] && TRAIN_FLAGS+=(--allow-cross-config)

log()  { printf '[launcher %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { log "ERROR: $*" >&2; exit 1; }
run()  { log "\$ $*"; "$@"; }

# Domains: explicit list, or 'all' = every subfolder of the batch with a Training/ dir
# (same rule as pipeline.py; _models/, dot-dirs and __pycache__ are never domains).
discover_domains() {
    local d
    for d in "$BATCH_DIR"/*/; do
        d="${d%/}"; d="${d##*/}"
        [[ "$d" == _* || "$d" == .* || "$d" == __pycache__ ]] && continue
        [[ -d "$BATCH_DIR/$d/Training" ]] && printf '%s\n' "$d"
    done
}
if [[ "$DOMAINS" == all ]]; then
    [[ -d "$BATCH_DIR" ]] || die "batch dir not found: $BATCH_DIR"
    mapfile -t DOMAIN_LIST < <(discover_domains)
    (( ${#DOMAIN_LIST[@]} > 0 )) || die "no domain (subfolder with Training/) found under $BATCH_DIR"
else
    # shellcheck disable=SC2206
    DOMAIN_LIST=($DOMAINS)
fi
# shellcheck disable=SC2206
FRINGE_LIST=($FRINGE_SIZES); STRAT_LIST=($STRATEGIES)

# Run a stage with its output tee'd to a log file; fail if the command fails.
# NOT written as `if ! ( ... ) | tee`: a command inside an `if` test runs with
# errexit DISABLED, so a failing step in the subshell would be ignored and the
# stage would report success (seen: analyze_results.py crashed, stage 'done').
stage() {
    local name="$1"; shift
    mkdir -p "$LOG_DIR"
    local logf="${LOG_DIR}/${name}_${STAMP}.log"
    log "=== stage: $name  (log: $logf) ==="
    local t0=$SECONDS rc
    set +e
    ( set -e; "$@" ) 2>&1 | tee "$logf"
    rc=${PIPESTATUS[0]}
    set -e
    (( rc == 0 )) || die "stage '$name' FAILED (exit $rc) after $((SECONDS - t0))s -- see $logf"
    log "=== stage '$name' done in $((SECONDS - t0))s ==="
}

# ============================================================
#  STAGES
# ============================================================
do_check() {
    local ok=true
    log "repo=$REPO batch=$BATCH_DIR domains=[${DOMAIN_LIST[*]}] mode=$MODE strict=$STRICT"
    [[ -x "$DEEP_EXE" ]]      || { log "missing/non-executable planner binary: $DEEP_EXE (run ./build.sh nn)"; ok=false; }
    [[ -x "$PYTHON" ]]        || { log "missing python: $PYTHON"; ok=false; }
    [[ -d "$BATCH_DIR" ]]     || { log "missing batch dir: $BATCH_DIR"; ok=false; }
    for s in scripts/rl_exp/create_all_training_data.py scripts/rl_exp/train_models.py \
             scripts/rl_exp/pipeline.py scripts/gnn_exp/create_training_data.py \
             lib/rl_handler/offline_main.py; do
        [[ -f "$s" ]] || { log "missing script: $s"; ok=false; }
    done
    for d in "${DOMAIN_LIST[@]}"; do
        if [[ -d "$BATCH_DIR/$d/Training" ]]; then
            log "domain $d: $(ls "$BATCH_DIR/$d/Training" | wc -l) training instance(s), $(ls "$BATCH_DIR/$d/Test" 2>/dev/null | wc -l) test instance(s)"
        else
            log "domain $d: missing $BATCH_DIR/$d/Training"; ok=false
        fi
    done
    if [[ "$ok" == true ]]; then
        "$PYTHON" -c "import torch, onnx" 2>/dev/null \
            || { log "$PYTHON cannot import torch/onnx"; ok=false; }
        # Depth-map coverage: fail here, not after an hour of generation.
        "$PYTHON" - "$BATCH_DIR" "$DEPTH_MAP" "${DOMAIN_LIST[@]}" <<'PY' || ok=false
import os, sys
sys.path.insert(0, "scripts/gnn_exp")
from create_training_data import instance_domain_token, parse_depth_map
batch, spec, domains = sys.argv[1], sys.argv[2], sys.argv[3:]
dm = parse_depth_map(spec); bad = False
for d in domains:
    if d in dm:
        print(f"[check] depth {d}: {dm[d]} (folder-level)"); continue
    folder = os.path.join(batch, d, "Training")
    toks = sorted({instance_domain_token(f) for f in os.listdir(folder)})
    miss = [t for t in toks if t not in dm]
    if miss:
        print(f"[check] depth-map MISSING token(s) {miss} for mixed folder {d} (found {toks})"); bad = True
    else:
        print(f"[check] depth {d} (mixed): " + ", ".join(f"{t}={dm[t]}" for t in toks))
sys.exit(1 if bad else 0)
PY
        for F in "${FRINGE_LIST[@]}"; do
            [[ "$F" =~ ^(4|8|16|32)$ ]] || log "warning: planner inference sweeps fringes 4 8 16 32 only; F=$F will be trained but never evaluated"
        done
        run "$PYTHON" scripts/rl_exp/pipeline.py "$BATCH_DIR" --domains "${DOMAIN_LIST[@]}" "${PIPE_FLAGS[@]}" --dry-run >/dev/null \
            || { log "pipeline.py --dry-run failed"; ok=false; }
    fi
    [[ "$ok" == true ]] || die "preflight check failed"
    log "preflight OK"
}

do_gen() {
    local have_all=true
    for d in "${DOMAIN_LIST[@]}"; do
        [[ -d "$BATCH_DIR/_models/$d/training_data" ]] || have_all=false
    done
    if [[ "$have_all" == true && "$FORCE_REGEN" != true ]]; then
        log "training_data already present for [${DOMAIN_LIST[*]}]; skipping generation (FORCE_REGEN=true to redo)"
        return 0
    fi
    local gen_cmd=("$PYTHON" scripts/rl_exp/create_all_training_data.py "$BATCH_DIR"
        --deep_exe "$DEEP_EXE" --domains "${DOMAIN_LIST[@]}"
        --dataset-generation "${STRAT_LIST[@]}" --heuristics "$HEURISTICS"
        --depth-map "$DEPTH_MAP"
        --dataset-max-creation "$MAX_CREATION" --dataset-max-generation "$MAX_GENERATION"
        --seed "$SEED" "${GEN_FLAGS[@]}")
    run "${gen_cmd[@]}"
    [[ "$GENERATE_TEST_DATA" == true ]] && run "${gen_cmd[@]}" --test-data
    rm -rf out                      # planner scratch output (out/NN, out/plan_exec)
    # The generator reports a per-domain failure on stderr but exits 0: verify the tables exist.
    for d in "${DOMAIN_LIST[@]}"; do
        local n
        n=$(find "$BATCH_DIR/_models/$d/training_data" -name '*.csv' 2>/dev/null | wc -l)
        (( n > 0 )) || die "no training CSV produced for domain $d (see $BATCH_DIR/_models/$d/_failed)"
        log "domain $d: $n training CSV table(s)"
    done
    # Fingerprint the data so a later run can tell what generated it.
    local strict_s="no"; [[ "$STRICT" == yes ]] && strict_s="yes"
    local discard="0"; for s in "${STRAT_LIST[@]}"; do [[ "$s" =~ ^(all|S_DFS|s_dfs|S-DFS)$ ]] && discard="0.6"; done
    PYTHONPATH=lib/rl_handler "$PYTHON" -c "
from src.offline.dataspec import make_dataspec
print(make_dataspec('$MODE', '$strict_s', '$DEPTH_MAP', discard_factor=$discard,
      max_creation=$MAX_CREATION, max_generation=$MAX_GENERATION, seed=$SEED,
      generation='${STRAT_LIST[*]}'.replace(' ', ','), heuristics='$HEURISTICS'))" \
        > "$BATCH_DIR/.dataspec"
    log "wrote $BATCH_DIR/.dataspec: $(cat "$BATCH_DIR/.dataspec")"
}

do_train() {
    for d in "${DOMAIN_LIST[@]}"; do
        [[ -d "$BATCH_DIR/_models/$d/training_data" ]] || die "no training_data for $d; run './launcher.sh gen' first"
    done
    # shellcheck disable=SC2086
    run "$PYTHON" scripts/rl_exp/train_models.py "$BATCH_DIR" \
        --domains "${DOMAIN_LIST[@]}" --strategies "${STRAT_LIST[@]}" \
        --fringe-sizes "${FRINGE_LIST[@]}" --model "$MODEL" "${TRAIN_FLAGS[@]}" \
        --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --n-checkpoints "$N_CHECKPOINTS" \
        --context-mode "$CTX" $TRAIN_EXTRA
    for d in "${DOMAIN_LIST[@]}"; do for F in "${FRINGE_LIST[@]}"; do
        [[ -f "$BATCH_DIR/_models/$d/frontier_policy_${F}.onnx" ]] \
            || die "model not installed: $BATCH_DIR/_models/$d/frontier_policy_${F}.onnx"
    done; done
    log "models installed under $BATCH_DIR/_models/<domain>/frontier_policy_<F>.onnx"
}

do_infer() {
    for d in "${DOMAIN_LIST[@]}"; do for F in "${FRINGE_LIST[@]}"; do
        [[ -f "$BATCH_DIR/_models/$d/frontier_policy_${F}.onnx" ]] \
            || die "missing $BATCH_DIR/_models/$d/frontier_policy_${F}.onnx; run './launcher.sh train' first"
    done; done
    run "$PYTHON" scripts/rl_exp/pipeline.py "$BATCH_DIR" --domains "${DOMAIN_LIST[@]}" "${PIPE_FLAGS[@]}"
    log "results under combined_results/${BATCH}/<domain>/"
}

# ============================================================
#  MAIN
# ============================================================
case "$ACTION" in
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    check) stage check do_check ;;
    gen)   stage check do_check; stage gen   do_gen ;;
    train) stage check do_check; stage train do_train ;;
    infer) stage check do_check; stage infer do_infer ;;
    all)   stage check do_check; stage gen do_gen; stage train do_train; stage infer do_infer ;;
    *) echo "unknown action '$ACTION' (check|gen|train|infer|all)" >&2; exit 2 ;;
esac
log "all requested stages complete"

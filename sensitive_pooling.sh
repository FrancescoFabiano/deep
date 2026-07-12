#!/usr/bin/env bash
set -uo pipefail
# NOTE: no 'set -e' — we trap errors per batch, not globally.

# ============================================================
#  Ablation: pad × strat × context × mode (separated + merged)
#
#  Step 0: Generate training data ONCE PER MODE
#          (separated needs --no_goal; merged does not)
#  Step 1: Symlink data within each mode group
#  Step 2: Train + evaluate each config sequentially
#
#  Folders: exp/rl_exp/dqn_{no_pad|pad}_{no_strat|strat}_{mean|self_attention}_{separated|merged}/
# ============================================================

source .venv/bin/activate

REPO="$(pwd)"
DEEP_EXE="cmake-build-release-nn/bin/deep"
FRINGE_SIZES="16"
BATCH_SIZE=64
FRAMES=50000
N_CHECKPOINTS=20
RANDOM_PCT=0.3
TRAIN_MAX_CREATION=50000

# ---- grid ----
PADS=("no_pad" "pad") #  "pad"
STRATS=("no_strat" "no_strat")  # add "strat" to test stratified replay
CTXS=("self_attention" "mean") # "mean" 
MODES=("separated" "merged")  # "merged"

# ---- tracking ----
PASSED=()
FAILED=()

# ============================================================
#  STEP 0: Generate training data ONCE PER MODE
#  separated → --no_goal --strong_equality
#  merged    → --strong_equality (goal inlined in the graph)
# ============================================================
for mode in "${MODES[@]}"; do

    # First batch folder for this mode = canonical data source
    DATA_BATCH="exp/rl_exp/dqn_${PADS[0]}_${STRATS[0]}_${CTXS[0]}_${mode}"

    # Mode-dependent generation flags
    GEN_FLAG="--strong_equality"
    if [[ "$mode" == "separated" ]]; then
        GEN_FLAG+=" --no_goal"
    fi

    echo ""
    echo "============================================================"
    echo "  DATA GENERATION [${mode}] into: ${DATA_BATCH}"
    echo "  flags: ${GEN_FLAG}"
    echo "============================================================"

    if [[ ! -d "${DATA_BATCH}/_models/CC/training_data" ]] || \
       [[ -z "$(ls -A "${DATA_BATCH}/_models/CC/training_data/" 2>/dev/null)" ]]; then
        echo "[0] generating training data (${mode}) ..."
        if ! python3 scripts/gnn_exp/create_all_training_data.py "${DATA_BATCH}" \
                --deep_exe "${DEEP_EXE}" \
                --dataset-max-creation "${TRAIN_MAX_CREATION}" \
                ${GEN_FLAG}; then
            echo "[FATAL] data generation failed for ${mode} — cannot continue."
            exit 1
        fi
        echo "[0] data generation complete (${mode})."
    else
        echo "[0] training data already exists for ${mode} — skipping."
    fi

    rm -rf out

    # ---- Symlink to all other folders of the SAME mode ----
    DATA_SRC="${REPO}/${DATA_BATCH}/_models/CC/training_data"

    echo ""
    echo "  SYMLINKING [${mode}] training_data from: ${DATA_SRC}"

    for pad in "${PADS[@]}"; do
        for strat in "${STRATS[@]}"; do
            for ctx in "${CTXS[@]}"; do
                BATCH="exp/rl_exp/dqn_${pad}_${strat}_${ctx}_${mode}"
                TARGET="${BATCH}/_models/CC/training_data"
                if [[ "${BATCH}" == "${DATA_BATCH}" ]]; then
                    echo "    ${BATCH} — source (skip)"
                    continue
                fi
                mkdir -p "${BATCH}/_models/CC"
                if [[ -L "${TARGET}" ]]; then
                    rm "${TARGET}"
                fi
                ln -sfn "${DATA_SRC}" "${TARGET}"
                echo "    ${BATCH} — linked"
            done
        done
    done

done

# ============================================================
#  STEP 2: Train + evaluate each config
# ============================================================
for mode in "${MODES[@]}"; do
    for pad in "${PADS[@]}"; do
        for strat in "${STRATS[@]}"; do
            for ctx in "${CTXS[@]}"; do

                BATCH="exp/rl_exp/dqn_${pad}_${strat}_${ctx}_${mode}"
                echo ""
                echo "============================================================"
                echo "  BATCH: ${BATCH}"
                echo "  mode=${mode}  pad=${pad}  strat=${strat}  ctx=${ctx}"
                echo "============================================================"

                # ---- mode-dependent flags ----
                TRAIN_FLAG=""
                PIPE_FLAG=""
                if [[ "$mode" == "separated" ]]; then
                    TRAIN_FLAG="--no_goal"
                    PIPE_FLAG="--separated"
                fi

                # ---- common training-stability flags (all arms) ----
                # Polyak soft target updates + cosine LR decay: prevents the
                # unbounded Q-value divergence (q_mean -> -34.5) seen with the
                # hard target sync + constant LR. Architecture-agnostic; applied
                # before the per-config pad/strat/ctx flags below.
                TRAIN_EXTRA=" --target-tau 0.005 --lr-schedule cosine --lr-min 1e-5"

                # ---- config-dependent flags ----
                if [[ "$pad" == "no_pad" ]]; then
                    TRAIN_EXTRA+=" --no-pad-closed"
                fi

                if [[ "$strat" == "strat" ]]; then
                    TRAIN_EXTRA+=" --stratified-replay"
                fi

                if [[ "$ctx" == "self_attention" ]]; then
                    TRAIN_EXTRA+=" --context-mode self_attention"
                else
                    TRAIN_EXTRA+=" --context-mode mean_pool"
                fi

                ok=true

                # ---- TRAIN ----
                echo "[1/2] training (F=${FRINGE_SIZES}, batch=${BATCH_SIZE}, frames=${FRAMES}) ..."
                if ! python3 scripts/rl_exp/train_models.py "${BATCH}" \
                        --fringe-sizes ${FRINGE_SIZES} \
                        --model dqn \
                        ${TRAIN_FLAG} \
                        --batch-size ${BATCH_SIZE} \
                        --frames ${FRAMES} \
                        --n-checkpoints ${N_CHECKPOINTS} \
                        --random-pct ${RANDOM_PCT} \
                        ${TRAIN_EXTRA}; then
                    echo "[FAIL] ${BATCH} (train)"
                    FAILED+=("${BATCH}:train")
                    ok=false
                fi

                # ---- EVAL PIPELINE ----
                if $ok; then
                    echo "[2/2] evaluation pipeline ..."
                    if ! python3 scripts/rl_exp/pipeline.py "${BATCH}" \
                            ${PIPE_FLAG}; then
                        echo "[FAIL] ${BATCH} (pipeline)"
                        FAILED+=("${BATCH}:pipeline")
                        ok=false
                    fi
                fi

                rm -rf out

                if $ok; then
                    echo "[DONE] ${BATCH}"
                    PASSED+=("${BATCH}")
                else
                    echo "[SKIPPED] ${BATCH} — failed, moving on"
                fi

            done
        done
    done
done

# ============================================================
#  SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "  SUMMARY  (${#PASSED[@]} passed, ${#FAILED[@]} failed)"
echo "============================================================"
for b in "${PASSED[@]+"${PASSED[@]}"}"; do echo "  ✓ $b"; done
for b in "${FAILED[@]+"${FAILED[@]}"}"; do echo "  ✗ $b"; done
echo "============================================================"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi

#!/usr/bin/env bash
set -uo pipefail
# NOTE: no 'set -e' — we trap errors per batch, not globally.

# ============================================================
#  Full study driver: generate data → train models → evaluate
#  One batch folder at a time, sequentially.
#  Safe: a failed batch is logged and skipped, not fatal.
# ============================================================

source .venv/bin/activate

DEEP_EXE="cmake-build-release-nn/bin/deep"
FRINGE_SIZES="1 4 8 16 32 64"
BATCH_SIZE=24
FRAMES=50000
N_CHECKPOINTS=10
CQL_ALPHA=1.0
RANDOM_PCT=0.5
TRAIN_MAX_CREATION=50000
MAX_DEPTH=40
TEST_MAX_CREATION=5000
# The VISIT ceiling (--dataset_max_generation). 100000 is the historical C++
# default, now set explicitly because it is the BINDING one: past it non-goals are
# dropped and their parents poisoned to 1e6 WITHOUT recursing, so goals below are
# never reached. See scripts/gnn_exp/create_all_training_data.py's docstring.
TRAIN_MAX_GENERATION=100000
TEST_MAX_GENERATION=100000

IF_GENERATE_DATA=true
IF_TRAIN_MODEL=true
GENERATE_TEST_DATA=false

ALGOS=("dqn") # "cql" 
MODES=("separated")
STRICTS=("strict")

# ---- tracking ----
PASSED=()
FAILED=()

for algo in "${ALGOS[@]}"; do
    for mode in "${MODES[@]}"; do
        for strict in "${STRICTS[@]}"; do

		BATCH="exp/rl_exp/batch_${mode}_${algo}_${strict}"
		echo ""
		echo "============================================================"
		echo "  BATCH: ${BATCH}  (algo=${algo}, mode=${mode}, strict=${strict})"
		echo "============================================================"

		# ---- mode-dependent flags ----
		GEN_FLAG=""
		TRAIN_FLAG=""
		PIPE_FLAG=""
		if [[ "$mode" == "separated" ]]; then
		    GEN_FLAG="--no_goal"
		    TRAIN_FLAG="--no_goal"
		    PIPE_FLAG="--separated"
		fi
		
		if [[ "$strict" == "strict" ]]; then
		    GEN_FLAG+=" --strong_equality"
		fi

		# ---- algo-dependent flags ----
		MODEL_FLAG="--model ${algo}"
		ALGO_FWD="--target-tau 0.005 --lr-schedule cosine --lr-min 1e-5"
		if [[ "$algo" == "cql" ]]; then
		    ALGO_FWD=" --cql-alpha ${CQL_ALPHA}"
		fi

		# ---- run all 4 steps; break to next batch on any failure ----
		ok=true

		# 1. GENERATE TRAINING DATA
		if $IF_GENERATE_DATA; then
		    echo "[1/4] generating training data ..."
		    if ! python3 scripts/gnn_exp/create_all_training_data.py "${BATCH}" \
		            --deep_exe "${DEEP_EXE}" \
		            ---depth 40 "${MAX_DEPTH}" \
		            --dataset-max-creation "${TRAIN_MAX_CREATION}" \
		            --dataset-max-generation "${TRAIN_MAX_GENERATION}" \
		            ${GEN_FLAG}; then
		        echo "[FAIL] ${BATCH} step 1 (generate train)"
		        FAILED+=("${BATCH}:generate_train")
		        ok=false
		    fi
		fi
		
		# 2. GENERATE TEST DATA
		if $IF_GENERATE_DATA && $GENERATE_TEST_DATA && $ok; then
		    echo "[2/4] generating test data ..."
		    if ! python3 scripts/gnn_exp/create_all_training_data.py "${BATCH}" \
			    --deep_exe "${DEEP_EXE}" \
			    --dataset-name "test_data" \
			    ---depth 40 "${MAX_DEPTH}" \
			    --dataset-max-creation "${TEST_MAX_CREATION}" \
			    --dataset-max-generation "${TEST_MAX_GENERATION}" \
			    --training-folder "Test" \
			    ${GEN_FLAG}; then
			echo "[FAIL] ${BATCH} step 2 (generate test)"
			FAILED+=("${BATCH}:generate_test")
			ok=false
		    fi
		fi
		
		rm -rf out

		# 3. TRAIN MODELS
		if $ok && $IF_TRAIN_MODEL; then
		    echo "[3/4] training models (algo=${algo}, F=${FRINGE_SIZES}) ..."
		    if ! python3 scripts/rl_exp/train_models.py "${BATCH}" \
		            --fringe-sizes ${FRINGE_SIZES} \
		            ${MODEL_FLAG} \
		            ${TRAIN_FLAG} \
		            --batch-size ${BATCH_SIZE} \
		            --frames ${FRAMES} \
		            --n-checkpoints ${N_CHECKPOINTS} \
		            --random-pct ${RANDOM_PCT} \
		            ${ALGO_FWD}; then
		        echo "[FAIL] ${BATCH} step 3 (train)"
		        FAILED+=("${BATCH}:train")
		        ok=false
		    fi
		fi

		# 4. EVALUATION PIPELINE
		if $ok; then
		    echo "[4/4] running evaluation pipeline ..."
		    if ! python3 scripts/rl_exp/pipeline.py "${BATCH}" \
		            ${PIPE_FLAG}; then
		        echo "[FAIL] ${BATCH} step 4 (pipeline)"
		        FAILED+=("${BATCH}:pipeline")
		        ok=false
		    fi
		fi
		
		rm -rf out

		if $ok; then
		    echo "[DONE] ${BATCH}"
		    PASSED+=("${BATCH}")
		else
		    echo "[SKIPPED] ${BATCH} — failed at step above, moving on"
		fi

        done
    done
done

# ============================================================
#  SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "  PASSED: ${#PASSED[@]}"
for b in "${PASSED[@]+"${PASSED[@]}"}"; do echo "    ✓ $b"; done
echo "  FAILED: ${#FAILED[@]}"
for b in "${FAILED[@]+"${FAILED[@]}"}"; do echo "    ✗ $b"; done
echo "============================================================"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi

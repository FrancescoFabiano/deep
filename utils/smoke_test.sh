#!/bin/bash

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <binary_path>"
  exit 1
fi

TIMEOUT_SECONDS=300
KILL_DELAY=5
TIMEOUT_BIN=""
BIN_PATH="$1"
CASES_FILE="utils/smoke_cases.tsv"

if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="gtimeout"
fi

if [[ ! "$BIN_PATH" = /* && ! -x "$BIN_PATH" && -x "./$BIN_PATH" ]]; then
  BIN_PATH="./$BIN_PATH"
fi

if [[ ! -x "$BIN_PATH" ]]; then
  echo "Binary not found or not executable: $BIN_PATH"
  exit 1
fi

if [[ ! -f "$CASES_FILE" ]]; then
  echo "Cases file not found: $CASES_FILE"
  exit 1
fi

run_with_optional_timeout() {
  local description="$1"
  shift

  echo "Running: $description"

  set +e
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" --kill-after=${KILL_DELAY} ${TIMEOUT_SECONDS} "$@"
  else
    echo "Warning: timeout utility not found; running without a time limit."
    "$@"
  fi
  local ret=$?
  set -e

  if [ $ret -eq 124 ]; then
    echo "Error: $description timed out after ${TIMEOUT_SECONDS} seconds. Forced kill after ${KILL_DELAY}s."
    exit $ret
  elif [ $ret -ne 0 ]; then
    echo "Error: $description failed (exit code $ret)"
    exit $ret
  fi
}

run_planning_checks() {
  local label="$1"
  local domain="$2"
  local problem="$3"
  local library="$4"
  local -a base_cmd=("$BIN_PATH" "$domain" "$problem")

  if [ "$library" != "-" ]; then
    base_cmd+=(--act_lib "$library")
  fi

  run_with_optional_timeout \
    "deep on ${label} with HFS, SUBGOALS, and fast comparison" \
    "${base_cmd[@]}" -s HFS -u SUBGOALS --fast-comparison

  run_with_optional_timeout \
    "deep on ${label} with Astar, SUBGOALS, bisimulation, visited-state checking, and fast comparison" \
    "${base_cmd[@]}" -s Astar -u SUBGOALS -b -c --fast-comparison
}

run_execution_checks() {
  local label="$1"
  local domain="$2"
  local problem="$3"
  local library="$4"
  local actions="$5"
  local -a base_cmd=("$BIN_PATH" "$domain" "$problem")

  if [ "$library" != "-" ]; then
    base_cmd+=(--act_lib "$library")
  fi

  read -r -a action_array <<< "$actions"

  run_with_optional_timeout \
    "deep on ${label} with actions [${actions}] (basic execution with fast comparison)" \
    "${base_cmd[@]}" --fast-comparison -e -a "${action_array[@]}"

  run_with_optional_timeout \
    "deep on ${label} with actions [${actions}] (visited-state check with fast comparison)" \
    "${base_cmd[@]}" --fast-comparison -c -e -a "${action_array[@]}"

  run_with_optional_timeout \
    "deep on ${label} with actions [${actions}] (bisimulation (FB) check with fast comparison)" \
    "${base_cmd[@]}" --fast-comparison -b -e -a "${action_array[@]}"

  run_with_optional_timeout \
    "deep on ${label} with actions [${actions}] (bisimulation (PT) check with fast comparison)" \
    "${base_cmd[@]}" --fast-comparison -b --bisimulation_type PT -e -a "${action_array[@]}"

  run_with_optional_timeout \
    "deep on ${label} with actions [${actions}] (bisimulation and visited-state check with fast comparison)" \
    "${base_cmd[@]}" --fast-comparison -b -c -e -a "${action_array[@]}"
}

while IFS=$'\t' read -r label domain problem library actions; do
  [[ -z "${label}" ]] && continue
  [[ "${label}" =~ ^# ]] && continue

  if [[ ! -f "$domain" ]]; then
    echo "Domain file not found for ${label}: $domain"
    exit 1
  fi
  if [[ ! -f "$problem" ]]; then
    echo "Problem file not found for ${label}: $problem"
    exit 1
  fi
  if [[ -n "$library" && "$library" != "-" && ! -f "$library" ]]; then
    echo "Library file not found for ${label}: $library"
    exit 1
  fi

  run_planning_checks "$label" "$domain" "$problem" "${library:--}"

  if [[ -n "$actions" ]]; then
    run_execution_checks "$label" "$domain" "$problem" "${library:--}" "$actions"
  fi
done < "$CASES_FILE"

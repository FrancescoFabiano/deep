#!/bin/bash

# Function to run a command and check result
run_deep() {
  local description="$1"
  shift
  echo "Running: deep on $FILENAME with actions $actions ($description)"
  ./cmake-build-debug/bin/deep "$FILENAME" "$@"
  local ret=$?
  if [ $ret -ne 0 ]; then
    echo "Error: deep execution failed during '$FILENAME' with actions '$actions' ('$description') (exit code $ret)"
    exit $ret
  fi
  sleep 1
}

# Input validation
if [ $# -ne 1 ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

FILENAME="$1"

# Extract actions
actions=$(grep -oP '%%% Executed actions:\s*\K.*?(?=\s*%%%$)' "$FILENAME")
if [ -z "$actions" ]; then
  echo "No executable actions found."
  exit 1
fi

# Run tests
run_deep "basic execution" -e -a $actions
run_deep "visited states check" -c -e -a $actions
run_deep "bisimulation (FB) check" -b -e -a $actions
run_deep "bisimulation (PT) check" -b --bisimulation_type PT -e -a $actions
run_deep "bisimulation and visited states check" -b -c -e -a $actions

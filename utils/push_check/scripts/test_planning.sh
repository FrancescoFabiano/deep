#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

# Custom timeout value
TIMEOUT=300
KILL_DELAY=5

run_with_timeout() {
  desc="$1"
  shift

  echo "Running: $desc"

  # Run the command with timeout
  timeout --kill-after=${KILL_DELAY} ${TIMEOUT} "$@"
  ret=$?

  if [ $ret -eq 124 ]; then
    echo "Error: $desc timed out after ${TIMEOUT} seconds. Forced kill after ${KILL_DELAY}s."
    exit $ret
  elif [ $ret -ne 0 ]; then
    echo "Error: $desc failed (exit code $ret)"
    exit $ret
  fi

  sleep 2
}

FILENAME="$1"

run_with_timeout "deep on $FILENAME with portfolio (5 threads)" \
  ./cmake-build/bin/deep "$FILENAME" -p 5

run_with_timeout "deep on $FILENAME with portfolio (5 threads) bisimulation and visited state check" \
  ./cmake-build/bin/deep "$FILENAME" -b -c -p 5

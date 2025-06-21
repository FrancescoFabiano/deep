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

  # Start command in a new process group so we can kill everything cleanly
  timeout --foreground --kill-after=${KILL_DELAY} ${TIMEOUT} \
    bash -c "exec setsid \"$@\""

  ret=$?
  if [ $ret -eq 124 ]; then
    echo "Error: $desc timed out after ${TIMEOUT} seconds. Forced kill after ${KILL_DELAY}s."
    exit $ret
  elif [ $ret -ne 0 ]; then
    echo "Error: $desc failed (exit code $ret)"
    exit $ret
  fi

  sleep 5

}

FILENAME="$1"

run_with_timeout "deep on $FILENAME with portfolio (5 threads)" ./cmake-build-debug/bin/deep "$FILENAME" -p 5
run_with_timeout "deep on $FILENAME with portfolio (5 threads) bisimulation and visited state check" ./cmake-build-debug/bin/deep "$FILENAME" -b -c -p 5

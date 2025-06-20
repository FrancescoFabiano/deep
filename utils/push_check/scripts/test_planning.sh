#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

run_with_timeout() {
  desc="$1"
  shift
  echo "Running: $desc"
  timeout 300 "$@"
  ret=$?
  if [ $ret -eq 124 ]; then
    echo "Error: $desc timed out after 300 seconds. Should not happen."
    exit $ret
  elif [ $ret -ne 0 ]; then
    echo "Error: $desc failed (exit code $ret)"
    exit $ret
  fi
}

run_with_timeout "deep on $1 with portfolio (5 threads)" ./cmake-build-debug/bin/deep "$1" -p 5
run_with_timeout "deep on $1 with portfolio (5 threads) bisimulation and visited state check" ./cmake-build-debug/bin/deep "$1" -b -c -p 5
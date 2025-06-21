#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

actions=$(grep -oP '%%% Executed actions:\s*\K.*?(?=\s*%%%$)' "$1")
if [ -z "$actions" ]; then
  echo "No executable actions found."
  exit 1
fi

ret=0

echo "Running: deep on $1 with actions $actions"
./cmake-build-debug/bin/deep "$1" -e -a $actions
ret=$?
if [ $ret -ne 0 ]; then
  echo "Error: deep execution failed (exit code $ret)"
  exit $ret
fi

echo "Running: deep on $1 with actions $actions (visited states check)"
./cmake-build-debug/bin/deep "$1" -c -e -a $actions
ret=$?
if [ $ret -ne 0 ]; then
  echo "Error: deep execution failed (exit code $ret)"
  exit $ret
fi

echo "Running: deep on $1 with actions $actions (bisimulation (FB) check)"
./cmake-build-debug/bin/deep "$1" -b -e -a $actions
ret=$?
if [ $ret -ne 0 ]; then
  echo "Error: deep execution failed (exit code $ret)"
  exit $ret
fi

echo "Running: deep on $1 with actions $actions (bisimulation (PT) check)"
./cmake-build-debug/bin/deep "$1" -b --bisimulation_type PT -e -a $actions
ret=$?
if [ $ret -ne 0 ]; then
  echo "Error: deep execution failed (exit code $ret)"
  exit $ret
fi

echo "Running: deep on $1 with actions $actions (bisimulation and visited states check)"
./cmake-build-debug/bin/deep "$1" -b -c -e -a $actions
ret=$?
if [ $ret -ne 0 ]; then
  echo "Error: deep execution failed (exit code $ret)"
  exit $ret
fi
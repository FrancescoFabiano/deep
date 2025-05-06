#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <dot_file> <number>"
  exit 1
fi

DOT_FILE="$1"
DEPTH="$2"

source ~/py_envs/bin/activate

python3 lib/RL/main.py "$DOT_FILE" "$DEPTH"
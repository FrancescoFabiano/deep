#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Check for binary path argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <binary_path>"
    exit 1
fi

BIN_PATH="$1"

# If BIN_PATH is not an absolute path and not found, try prepending ./
if [[ ! "$BIN_PATH" = /* && ! -x "$BIN_PATH" && -x "./$BIN_PATH" ]]; then
    BIN_PATH="./$BIN_PATH"
fi

FOLDER="utils/push_check/experiments"
NUM_FILES_SET1=5
NUM_FILES_SET2=5

FILES_SET1=($(find "$FOLDER" -type f | shuf -n $NUM_FILES_SET1))
FILES_SET2=($(find "$FOLDER" -type f | shuf -n $NUM_FILES_SET2))

for FILE in "${FILES_SET1[@]}"; do
    ./utils/push_check/scripts/test_execution.sh "$BIN_PATH" "$FILE"
    sleep 2
done

for FILE in "${FILES_SET2[@]}"; do
    ./utils/push_check/scripts/test_planning.sh "$BIN_PATH" "$FILE"
    sleep 2
done
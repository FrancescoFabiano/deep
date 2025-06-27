#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Path to the folder containing files
FOLDER="utils/push_check/experiments"

# Number of random files to pick for each script set
NUM_FILES_SET1=8
NUM_FILES_SET2=5

# Get random files for set 1
FILES_SET1=($(find "$FOLDER" -type f | shuf -n $NUM_FILES_SET1))

# Get random files for set 2
FILES_SET2=($(find "$FOLDER" -type f | shuf -n $NUM_FILES_SET2))

# Run scripts for set 1
for FILE in "${FILES_SET1[@]}"; do
    ./utils/push_check/scripts/test_execution.sh "$FILE"
    sleep 2

done

# Run a different script for set 2
for FILE in "${FILES_SET2[@]}"; do
    ./utils/push_check/scripts/test_planning.sh "$FILE"
    sleep 2
done
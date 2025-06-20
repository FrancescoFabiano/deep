#!/bin/bash

# Path to the folder containing files
FOLDER="utils/push_check/experiments"

# Number of random files to pick
NUM_FILES=20

# Get a list of 5 random files
FILES=($(find "$FOLDER" -type f | shuf -n $NUM_FILES))

# Loop through each selected file
for FILE in "${FILES[@]}"; do
    echo "Running scripts on: $FILE"

    ./utils/push_check/scripts/test_execution.sh "$FILE"
    sleep 10
    ./utils/push_check/scripts/test_planning.sh "$FILE"
    sleep 10

done
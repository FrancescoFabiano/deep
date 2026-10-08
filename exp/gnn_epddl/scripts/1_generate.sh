#!/bin/bash
# Step 1: write every generated problem into exp/gnn_epddl/instances (byte-identical to the shipped files;
# the domain files and act_lib.epddl are the benchmark's and are shipped as they are).
set -euo pipefail
source "$(dirname "$0")/env.sh"
gnn generate --out "$INSTANCES"

#!/bin/bash
# Step 4: the training runs of the experiments (50 epochs, validation and checkpoint every 10 epochs).
# Per-domain models: seeds 0, 1, 2 (selection picks over all three); general and subset models: seed 0.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$WORK"
T=(--split "$SPLIT" --data "$DATA" --instances "$INSTANCES" --deep "$DEEP" --ckpt "$CKPT" --log "$LOG"
   --epochs 50 --eval-every 10 --device "$DEVICE")
for seed in 0 1 2; do
  for fam in active-muddy-child blocks-world cloud-scheduling consecutive-numbers gossip search-and-rescue selective-communication; do
    gnn train --family "$fam" --seed "$seed" --threads 2 --vjobs 2 "${T[@]}"
  done
done
gnn train --all --seed 0 --threads 4 --vjobs 8 "${T[@]}"                                                      # final-all
gnn train --families search-and-rescue,cloud-scheduling,blocks-world --seed 0 --threads 4 --vjobs 6 "${T[@]}"  # final-sub
# Transfer models (one group of families held out), seed 0:
gnn train --families active-muddy-child,consecutive-numbers,blocks-world,search-and-rescue,cloud-scheduling --name tr-nospread --seed 0 --threads 4 --vjobs 4 "${T[@]}"
gnn train --families gossip,selective-communication,blocks-world,search-and-rescue,cloud-scheduling --name tr-nopuzzle --seed 0 --threads 4 --vjobs 4 "${T[@]}"
gnn train --families gossip,selective-communication,active-muddy-child,consecutive-numbers --name tr-nophys --seed 0 --threads 4 --vjobs 4 "${T[@]}"
# Sub-family models (an ablation; SUBFAMILY=1 to train them): 30 epochs, evaluation every 5, seeds 0-2.
if [ "${SUBFAMILY:-0}" = 1 ]; then
  S=(--split "$SPLIT" --data "$DATA" --instances "$INSTANCES" --deep "$DEEP" --ckpt "$CKPT" --log "$LOG" --epochs 30 --eval-every 5 --device "$DEVICE")
  for seed in 0 1 2; do
    for sf in "blocks-world classical" "blocks-world epistemic" "blocks-world clumsy" "gossip all" "gossip all-deceived" "gossip imp-deceived" "gossip single"; do
      gnn train --subfamily $sf --seed "$seed" --threads 2 --vjobs 2 "${S[@]}"
    done
  done
fi

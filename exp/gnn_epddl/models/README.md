# Exported models

The trained rankers of the experiments, each in two exports with the same
weights and the same scores (up to floating-point rounding):

- `cpu/`: the faster export for the CPU, which sums each node's incoming
  messages with a running sum over edges sorted by target;
- `gpu/`: the standard export (scatter-add), fast on CUDA.

Every `.onnx` file has a sidecar `<file>.onnx.vocab` that deep reads: the
node-type vocabulary and the model options `#anon` (predicate names hidden),
`#drop_holds`, `#ops` (goal operators typed) and `#scale`.

## Layout

```
cpu|gpu/per-domain/<domain name>.onnx   one model per domain (the name after "domain" in the domain file)
cpu|gpu/general.onnx                    one model for all seven domains
cpu|gpu/subset.onnx                     trained on search-and-rescue, cloud-scheduling, blocks-world only
cpu|gpu/no-spreading.onnx               trained without selective-communication and gossip
cpu|gpu/no-puzzle.onnx                  trained without active-muddy-child and consecutive-numbers
cpu|gpu/no-physical.onnx                trained without blocks-world, search-and-rescue and cloud-scheduling
cpu|gpu/sub-family/<family>-<sub>.onnx  one model per sub-family of blocks-world and gossip (an ablation)
cpu|gpu/untrained.onnx                  the network as training initialises it, never trained (a control)
```

`MODELS.csv` gives, for every model, the training families, the seeds trained,
the training schedule, the selected seed and epoch, the validation result
(problems solved and the sum of log10 nodes over the validation searches), and
the md5 of both files. Selection rule: most validation problems solved, then
the smallest sum of log10 nodes, ties to the later epoch, over every
validation point of every seed.

## Use

A folder selects the per-domain model of the problem's domain, falling back to
`general.onnx`; a file uses that model for every domain:

```
deep <domain> <problem> --act_lib <lib> -b -c --fast-comparison -s HFS -u GNN --onnx_threads 1 \
     --ranker_model exp/gnn_epddl/models/cpu [--ranker_model_gpu exp/gnn_epddl/models/gpu]
deep ... -s HFS -u GNN --ranker_model exp/gnn_epddl/models/cpu/general.onnx
deep ... -s RL -u RL_H --RL_fringe_size 8 --ranker_model exp/gnn_epddl/models/cpu
```

`scripts/6_export.sh` writes both exports of a selected checkpoint;
`scripts/7_run.sh` runs the experiments' configurations with these files.

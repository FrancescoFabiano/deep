"""Generate RL datasets by delegating to the gnn_exp generator.

Default (no flag): builds TRAINING data from each domain's Training/ folder into
_models/<domain>/training_data/ (unchanged behaviour).

--test-data: builds held-out TEST data from each domain's Test/ folder into
_models/<domain>/test_data/, mirroring the training pass EXACTLY (same
reconstructed tree + `Distance From Goal`/d* construction; only the input folder
and output dataset name differ). Domains without a Test/ folder are skipped
silently by the generator (it only walks domains that contain the input folder).
Test data is for held-out measurement only -- never gradients/selection.

Every other flag passes through to the gnn_exp generator, which REQUIRES both
ceilings explicitly (--dataset-max-creation, the WRITE cap, and
--dataset-max-generation, the VISIT cap -- the binding one). They have no defaults
on purpose: a default is a second opinion about how data was generated and drifts
from what callers actually pass.

--dataset-generation BFS DFS S_DFS HFS (any subset, or `all`) picks the behaviour
policies: the generator runs once per strategy and writes each tree to
_models/<domain>/<dataset>/<STRAT>/<instance>/, and rl_handler trains one
reconstructed tree per (instance, strategy), replaying that search's own expansion
order as the behaviour (lib/rl_handler/src/offline/strategies.py, policies.py).

Examples (run from project root):
  python3 scripts/rl_exp/create_all_training_data.py exp/rl_exp/batch0 \\
      --deep_exe cmake-build-release-nn/bin/deep \\
      --dataset-max-creation 50000 --dataset-max-generation 100000
  python3 scripts/rl_exp/create_all_training_data.py exp/rl_exp/batch0 \\
      --deep_exe cmake-build-release-nn/bin/deep \\
      --dataset-max-creation 5000 --dataset-max-generation 100000 --test-data
"""

import subprocess
import sys


def main():
    args = list(sys.argv[1:])
    if "--test-data" in args:
        args = [a for a in args if a != "--test-data"]
        # Convenience alias for the held-out test pass: read Test/, write
        # test_data/. These levers already exist in the gnn_exp generator, so
        # the test pass reuses the identical tree + d* construction.
        conflict = any(
            a == "--training-folder" or a.startswith("--training-folder=")
            or a == "--dataset-name" or a.startswith("--dataset-name=")
            for a in args
        )
        if conflict:
            print("[ERROR] --test-data cannot be combined with explicit "
                  "--training-folder/--dataset-name.")
            sys.exit(2)
        args += ["--training-folder", "Test", "--dataset-name", "test_data"]
    cmd = ["python3", "scripts/gnn_exp/create_all_training_data.py", *args]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

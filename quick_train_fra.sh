source .venv/bin/activate
python3 scripts/gnn_exp/create_all_training_data.py exp/rl_exp/batch0_separated --deep_exe cmake-build-release-nn/bin/deep --dataset-max-creation 30000
python3 scripts/gnn_exp/create_all_training_data.py exp/rl_exp/batch0_separated --deep_exe cmake-build-release-nn/bin/deep --dataset-name "test_data" --dataset-max-creation 1000 --training-folder "Test"
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_separated

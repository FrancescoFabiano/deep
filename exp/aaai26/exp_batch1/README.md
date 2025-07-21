# Experiment Batch 1: Standard Benchmarks

This folder contains the first batch of experiments for the project.

- This batch tests GNN-based heuristics against breadth-first search on standard benchmarks.
- For each domain, a dedicated model is trained (with the problem instance in the `Training` subfolder) and then used to solve all instances within that domain.
- All models of interest are located in the `models` subfolder.
- Scripts to run multiple experiments and aggregate results are provided in the `scripts` subfolder.
# Experiment Batch 2: Granular Domain Testing

This folder contains the second batch of experiments for the project.

- Each domain is split into subdomains to test whether GNN models trained with more or less domain-specific training data perform better.
- No extra instances are added; the goal is to verify how much training data is needed for effective GNN-based heuristics.
- For each sub-domain, a dedicated model is trained (with the problem instance in the `Training` subfolder) and then used to solve all instances within that domain.
- All models of interest are located in the `models` subfolder.
- Scripts to run multiple experiments and aggregate results are provided in the `scripts` subfolder.
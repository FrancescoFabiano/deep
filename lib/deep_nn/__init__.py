"""Shared data path of the learned heuristics (GNN distance estimator, RL fringe ranker).

Both models are deployed through the SAME planner consumer, FringeEvalRL, so
everything between the generator's output and the ONNX file is common:

    dot.py       planner DOT -> StateGraph (node ids, edges, designated worlds)
    cache.py     all states of one instance parsed once (+ derived goal graph)
    pack.py      FringeEvalRL::fringe_to_tensor_minimal, in Python
    contract.py  the ONNX input/output contract: names, order, dtypes, export, checks
    features.py  the one model block every network shares (designated-world marker)
    tree.py, env.py, policies.py, dataset.py, strategies.py, planner_config.py
                 the generation table -> search tree -> planner-shaped beams

What differs per model -- the network and its loss -- lives in lib/gnn_handler
and lib/rl_handler.  Nothing here imports from either.
"""

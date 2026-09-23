# Portfolio Configurations

The portfolio parser reads one configuration per line, with comma-separated
`key=value` pairs:

```text
search=Astar,heuristics=SUBGOALS,bisimulation=true,bisimulation_interval=1,check_visited=true,fast_world_comparison=true,fast_state_comparison=true
```

Blank lines and lines starting with `#` are ignored.

Included examples:

- `config-P5.ut`: the first five built-in portfolio configurations.
- `config-ALL.ut`: the full built-in portfolio order.

The current built-in portfolio order is:

1. `BFS`, interval `2`
2. `Astar + SUBGOALS`, interval `2`
3. `DFS`, interval `2`
4. `BFS`, interval `1`
5. `Astar + SUBGOALS`, interval `1`
6. `DFS`, interval `1`
7. `BFS`, interval `5`
8. `Astar + SUBGOALS`, interval `5`
9. `DFS`, interval `5`

All built-in entries enable:

- `bisimulation=true`
- `check_visited=true`
- `fast_world_comparison=true`
- `fast_state_comparison=true`

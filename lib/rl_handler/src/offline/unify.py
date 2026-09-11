"""The UNIFIED state graph: one graph per problem, merged across behaviour policies.

WHY (2026-09-09)
----------------
Each generation strategy (BFS / DFS / S_DFS / HFS, see `strategies.py`) writes its
own tree of the SAME problem, and the DOT files it mints are numbered per run, so
one planner state reached by two strategies lives on disk twice under two names
(and, because the C++ mints a file per VISIT, sometimes twice inside one tree).
Training on the per-strategy trees therefore scores one state against several
labels -- one delta per tree, each a discovery distance inside that tree alone --
and a state that the BFS tree proves 4 steps from a goal can sit at 6 in the DFS
tree and at 1e6 (censored, never expanded) in the HFS tree.

Measured on CC_2_2_3__pl_4 (all four strategies, batch1_cc_strat):
    63,690 DOT files -> 58,722 unique states; 695 in >= 2 strategies; 3 in all 4;
    36 of the 695 shared states carry DISAGREEING deltas across trees.

The unified graph collapses that. States are identified by their CONTENT; the
per-strategy trees become one DAG whose edges are the union of every recorded
parent->child edge, and every label is recomputed on that DAG:

    delta(v)   = shortest distance to a goal in the UNION   (<= min over trees)
    h*(v)      = min over trees                              (the generator's memo)
    is_goal(v) = OR over trees                               (conflicts are counted)
    expanded   = OR over trees                               (any search expanded it)
    censored   = recomputed on the union (an unexpanded leaf in one tree that
                 another tree expanded is NOT censored any more)
    depth(v)   = BFS depth from the root in the union

The behaviour policies survive as per-strategy TRACES: `traces[s][v]` is the
expansion rank of v in strategy s (None where s never expanded v), so the
`trace:<s>` behaviour (`policies.py`) replays s's own search on the unified graph.

WHAT IS THE IDENTITY OF A STATE
-------------------------------
The raw bytes of the DOT file: the belief edges over world ids (hash of the
fluent set plus repetition, the planner's own world identity) with agent labels. In separated mode the C++ does NOT write the
pointed world (HelperPrint.cpp, the `doublecircle` line is commented out), so two
planner states that differ only in the pointed world collapse to one here. That is
model-equivalence -- the network cannot tell them apart either -- but it is not
planner-equivalence, and it is why `n_goal_conflicts` exists: a goal is a property
of the pointed world. In merged mode the epsilon->pointed edge is in the file and
the identity is exact.

MEMORY / TIME
-------------
The fingerprint pass reads each file once and hashes its raw bytes (16 bytes per
state kept, nothing else; the writer's edge order is deterministic, verified). No graph tensors are built here; only the
REPRESENTATIVE DOT of each unique state is later parsed into the encoder cache, so
the cache shrinks to the unique count. Fingerprints are cached per tree
(`<cache_dir>/<tree name>.fp.json`, keyed on the CSV path and state count).

THE GRAPH IS A DAG WITH POSSIBLE CYCLES, NOT A TREE
---------------------------------------------------
Two visits of one state in one tree, or one state in two trees, become one node
with several parents; a state re-reached below itself becomes a cycle. Every
consumer that walks `children` must carry a visited set (`compute_delta`,
`compute_censored`, `FringeEnv`, `bfs_open_max` already do;
`initial_forced_chain_len` was fixed for this).
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .strategies import NAME_SEP, STRATEGIES
from .tree import (
    INF_DELTA,
    UNREACHABLE_DISTANCE,
    TreeInstance,
    compute_censored,
    compute_delta,
)

# The `strategy` attribute of a unified instance. Deliberately NOT in
# strategies.STRATEGIES: it is not a generator, and dir_name() must keep refusing it.
UNIFIED_STRATEGY = "unified"

FINGERPRINT_VERSION = 2   # 2: raw-bytes digest (was sorted edge lines)


def unified_tree_name(instance: str) -> str:
    """`CC_2_2_3__pl_4` -> `CC_2_2_3__pl_4@unified`. `config_of` / `expected_optimal`
    keep parsing it (they split on `__pl_`; `split_tree_name` splits on `@`)."""
    return f"{instance}{NAME_SEP}{UNIFIED_STRATEGY}"


# ----------------------------------------------------------------------------
# fingerprints
# ----------------------------------------------------------------------------

def fingerprint_dot_bytes(data: bytes) -> bytes:
    """16-byte digest of the RAW BYTES of a planner DOT (2026-09-10, per the user:
    the binary content IS the identity).

    The C++ writer (HelperPrint::print_dataset_format) iterates ordered containers,
    so two writes of the same state are byte-identical: measured on batch1_cc_strat
    CC, 231,345 files give 223,640 unique digests both by raw bytes and by the
    order-independent sorted-edge variant -- no difference. Raw bytes is the
    stricter of the two (any formatting difference keeps two files apart, which
    can only UNDER-merge, never wrongly merge), and it needs no parsing.

    Whatever the writer puts in the file is part of the identity: in merged mode
    that includes the epsilon->pointed edge and the goal subgraph; in separated
    mode the file holds the belief edges only (no pointed world -- see the module
    docstring).
    """
    return hashlib.blake2b(data, digest_size=16).digest()


def fingerprint_edges_sorted(data: bytes) -> bytes:
    """Order-independent variant (sorted edge lines): a diagnostic to detect a
    writer whose edge order is NOT deterministic (then raw != sorted counts)."""
    edges = sorted(ln.strip() for ln in data.splitlines() if b"->" in ln)
    h = hashlib.blake2b(digest_size=16)
    for e in edges:
        h.update(e)
        h.update(b"\n")
    return h.digest()


def fingerprint_dot(path: str | Path) -> bytes:
    with open(path, "rb") as fh:
        return fingerprint_dot_bytes(fh.read())


def fingerprints_for_tree(
    tree: TreeInstance,
    repo_root: str | Path,
    cache_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> List[bytes]:
    """One digest per state id of `tree`, streamed file by file; cached per tree."""
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{tree.name}.fp.json"
        if cache_file.exists():
            try:
                doc = json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                doc = {}
            if (doc.get("version") == FINGERPRINT_VERSION
                    and doc.get("csv_path") == str(Path(tree.csv_path).resolve())
                    and doc.get("n_states") == tree.n_states
                    and len(doc.get("hex", [])) == tree.n_states):
                return [bytes.fromhex(h) for h in doc["hex"]]
    import time
    t0 = time.time()
    out = [fingerprint_dot(p) for p in tree.state_paths_abs(repo_root)]
    if verbose:
        dt = time.time() - t0
        print(f"[unify] fingerprinted {tree.name}: {len(out)} DOT files in {dt:.1f}s")
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "version": FINGERPRINT_VERSION,
            "csv_path": str(Path(tree.csv_path).resolve()),   # absolute: cwd-independent key
            "n_states": tree.n_states,
            "hex": [d.hex() for d in out],
        }))
    return out


# ----------------------------------------------------------------------------
# the unified instance
# ----------------------------------------------------------------------------

@dataclass
class UnifiedInstance(TreeInstance):
    """A TreeInstance whose states are unique by content and whose structure is the
    union of several per-strategy trees of one problem.

    Extra fields (all keyed by CANONICAL state id):
      traces      strategy -> expansion rank per state (None = s never expanded it)
      n_trees_of  how many member trees contain the state (1 = private to one pi_b)
      members     strategy -> member tree name;  csv_paths: every member table
      manifest    the merge report (counts, conflicts, improvements) -- goes in the
                  run sidecar so a merge that silently did nothing is visible
    """
    traces: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    n_trees_of: List[int] = field(default_factory=list)
    members: Dict[str, str] = field(default_factory=dict)
    csv_paths: List[str] = field(default_factory=list)
    manifest: Dict[str, object] = field(default_factory=dict)

    @property
    def has_trace(self) -> bool:            # type: ignore[override]
        return bool(self.traces)

    @property
    def n_expanded(self) -> int:            # type: ignore[override]
        if self.expanded is None:
            return 0
        return int(sum(1 for e in self.expanded if e))

    def trace_strategies(self) -> List[str]:
        return [s for s in STRATEGIES if s in self.traces]

    def initial_forced_chain_len(self) -> int:   # cycle-safe override
        v, n, seen = self.root_id, 0, set()
        while len(self.children[v]) == 1 and v not in seen:
            seen.add(v)
            n += 1
            v = self.children[v][0]
        return n

    def stats(self) -> Dict[str, object]:
        out = super().stats()
        out["unified"] = {k: v for k, v in self.manifest.items() if k != "per_tree"}
        out["per_tree"] = self.manifest.get("per_tree")
        return out


def unify_trees(
    trees: Sequence[TreeInstance],
    fingerprints: Sequence[Sequence[bytes]],
    name: Optional[str] = None,
    depth_bound: Optional[int] = None,
) -> UnifiedInstance:
    """Merge the per-strategy trees of ONE problem into a UnifiedInstance.

    `fingerprints[i][v]` is the content digest of state v of `trees[i]`
    (`fingerprints_for_tree`). All trees must share the problem (`.instance`) and
    their roots must fingerprint identically -- a root mismatch means the tables
    are not of the same problem or were generated in different modes, and refusing
    is the only safe answer.
    """
    if not trees:
        raise ValueError("unify_trees: no trees given")
    if len(fingerprints) != len(trees):
        raise ValueError("unify_trees: one fingerprint list per tree")
    instance = trees[0].instance
    for t in trees:
        if t.instance != instance:
            raise ValueError(
                f"unify_trees: trees of different problems ({instance!r} vs "
                f"{t.instance!r}); unify one problem at a time.")
    strategies_seen = [t.strategy for t in trees]
    if len(set(strategies_seen)) != len(strategies_seen):
        raise ValueError(
            f"unify_trees: {instance}: two trees for one strategy "
            f"({strategies_seen}); one tree per (instance, strategy).")
    for t, fp in zip(trees, fingerprints):
        if len(fp) != t.n_states:
            raise ValueError(f"unify_trees: {t.name}: {len(fp)} fingerprints for "
                             f"{t.n_states} states")

    canon: Dict[bytes, int] = {}
    rep_path: List[str] = []
    depth_min: List[int] = []
    h_star: List[float] = []
    is_goal: List[bool] = []
    goal_votes: List[List[bool]] = []          # per canonical id: goal verdicts seen
    n_trees_of: List[int] = []
    children: List[List[int]] = []
    child_sets: List[set] = []
    expanded_any: List[bool] = []
    any_expanded_info = any(t.expanded is not None for t in trees)
    per_tree_delta: List[Dict[int, float]] = []   # canonical id -> delta in that tree
    traces: Dict[str, List[Optional[int]]] = {}
    n_self_loops = 0
    n_within_tree_dups = 0
    roots: List[int] = []
    maps: List[List[int]] = []

    for t, fp in zip(trees, fingerprints):
        m: List[int] = [0] * t.n_states
        seen_here: set = set()
        for v in range(t.n_states):
            cid = canon.get(fp[v])
            if cid is None:
                cid = len(rep_path)
                canon[fp[v]] = cid
                rep_path.append(t.state_paths[v])
                depth_min.append(t.depth[v])
                h_star.append(t.h_star[v])
                is_goal.append(bool(t.is_goal[v]))
                goal_votes.append([bool(t.is_goal[v])])
                n_trees_of.append(0)
                children.append([])
                child_sets.append(set())
                expanded_any.append(False)
            else:
                depth_min[cid] = min(depth_min[cid], t.depth[v])
                h_star[cid] = min(h_star[cid], t.h_star[v])
                is_goal[cid] = is_goal[cid] or bool(t.is_goal[v])
                goal_votes[cid].append(bool(t.is_goal[v]))
            m[v] = cid
            if cid in seen_here:
                n_within_tree_dups += 1
            else:
                seen_here.add(cid)
                n_trees_of[cid] += 1
            if t.expanded is not None and t.expanded[v]:
                expanded_any[cid] = True
        maps.append(m)
        roots.append(m[t.root_id])
        # union of edges, first-seen order, no self loops, no duplicates
        for v in range(t.n_states):
            pv = m[v]
            for c in t.children[v]:
                pc = m[c]
                if pc == pv:
                    n_self_loops += 1
                    continue
                if pc not in child_sets[pv]:
                    child_sets[pv].add(pc)
                    children[pv].append(pc)
        d: Dict[int, float] = {}
        for v in range(t.n_states):
            cid = m[v]
            d[cid] = min(d.get(cid, INF_DELTA), t.delta[v])
        per_tree_delta.append(d)
        if t.expansion_rank is not None and t.strategy:
            tr: List[Optional[int]] = [None] * len(rep_path)
            traces[t.strategy] = tr
    # traces are sized once every state is known; fill them now
    for t, m in zip(trees, maps):
        if t.expansion_rank is None or not t.strategy:
            continue
        tr = traces[t.strategy]
        if len(tr) < len(rep_path):
            tr.extend([None] * (len(rep_path) - len(tr)))
        for v, r in enumerate(t.expansion_rank):
            if r is None:
                continue
            cid = m[v]
            tr[cid] = r if tr[cid] is None else min(tr[cid], r)

    if len(set(roots)) != 1:
        raise ValueError(
            f"unify_trees: {instance}: the member trees do not share a root state "
            f"(fingerprints differ across {[t.name for t in trees]}). Same problem, "
            f"same generation mode? Refusing to merge.")
    root_id = roots[0]
    n = len(rep_path)

    # depth: BFS from the root over the union (unreachable states keep the min
    # recorded depth so bfs/dfs rankings and censoring still have a number).
    depth = list(depth_min)
    dist = [-1] * n
    dist[root_id] = 0
    q: deque[int] = deque([root_id])
    while q:
        v = q.popleft()
        for c in children[v]:
            if dist[c] == -1:
                dist[c] = dist[v] + 1
                q.append(c)
    for v in range(n):
        if dist[v] >= 0:
            depth[v] = dist[v]
    n_reachable = sum(1 for d in dist if d >= 0)

    delta = compute_delta(children, is_goal)
    expanded = expanded_any if any_expanded_info else None
    censored = compute_censored(children, is_goal, delta, h_star, depth,
                                depth_bound=depth_bound, expanded=expanded)

    # ---- the merge report ----
    n_goal_conflicts = sum(1 for votes in goal_votes if len(set(votes)) > 1)
    shared = [cid for cid in range(n) if n_trees_of[cid] >= 2]
    n_delta_disagree = 0
    n_delta_improved = 0
    for cid in shared:
        vals = {d[cid] for d in per_tree_delta if cid in d}
        if len(vals) > 1:
            n_delta_disagree += 1
    for cid in range(n):
        best_tree = min((d[cid] for d in per_tree_delta if cid in d), default=INF_DELTA)
        if delta[cid] < best_tree:
            n_delta_improved += 1
    n_hstar_contradicted = sum(
        1 for cid in range(n)
        if delta[cid] == INF_DELTA and h_star[cid] < UNREACHABLE_DISTANCE)
    hist: Dict[str, int] = {}
    for k in n_trees_of:
        hist[str(k)] = hist.get(str(k), 0) + 1
    per_tree = {
        t.name: {
            "strategy": t.strategy,
            "n_states": t.n_states,
            "n_unique_in_tree": len(set(m)),
            "delta_root": (None if t.delta_root == INF_DELTA else t.delta_root),
        }
        for t, m in zip(trees, maps)
    }
    manifest = {
        "instance": instance,
        "strategies": [t.strategy for t in trees],
        "n_files": sum(t.n_states for t in trees),
        "n_unique_states": n,
        "n_reachable": n_reachable,
        "n_shared_states": len(shared),
        "n_in_all_trees": sum(1 for k in n_trees_of if k == len(trees)),
        "n_trees_of_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "n_within_tree_duplicates": n_within_tree_dups,
        "n_self_loops_dropped": n_self_loops,
        "n_goal_conflicts": n_goal_conflicts,
        "n_shared_delta_disagree": n_delta_disagree,
        "n_delta_improved_by_union": n_delta_improved,
        "n_hstar_contradicted": n_hstar_contradicted,
        "delta_root_unified": (None if delta[root_id] == INF_DELTA else delta[root_id]),
        "delta_root_per_tree": {t.name: per_tree[t.name]["delta_root"] for t in trees},
        "per_tree": per_tree,
    }

    return UnifiedInstance(
        name=name or unified_tree_name(instance),
        csv_path=str(trees[0].csv_path),      # goal_tree.dot is looked up beside it
        state_paths=rep_path,
        depth=depth,
        h_star=h_star,
        is_goal=is_goal,
        children=children,
        root_id=root_id,
        n_orphan_states=n - n_reachable,
        delta=delta,
        goal_path=trees[0].goal_path,
        censored=censored,
        n_orphan_edges=sum(t.n_orphan_edges for t in trees),
        n_edge_rows=sum(t.n_edge_rows for t in trees),
        instance=instance,
        strategy=UNIFIED_STRATEGY,
        expansion_rank=None,
        expanded=expanded,
        traces=traces,
        n_trees_of=n_trees_of,
        members={t.strategy: t.name for t in trees if t.strategy},
        csv_paths=[str(t.csv_path) for t in trees],
        manifest=manifest,
    )


def group_by_instance(trees: Iterable[TreeInstance]) -> Dict[str, List[TreeInstance]]:
    out: Dict[str, List[TreeInstance]] = {}
    for t in trees:
        out.setdefault(t.instance, []).append(t)
    return out


def unify_instances(
    trees: Sequence[TreeInstance],
    repo_root: str | Path,
    cache_dir: Optional[str | Path] = None,
    depth_bound: Optional[int] = None,
    verbose: bool = True,
) -> List[UnifiedInstance]:
    """Group per-strategy trees by problem and merge each group. A problem with a
    single tree is still wrapped (its within-tree duplicates collapse), so every
    instance downstream has one representation."""
    out: List[UnifiedInstance] = []
    for inst, group in sorted(group_by_instance(trees).items()):
        fps = [fingerprints_for_tree(t, repo_root, cache_dir, verbose=verbose)
               for t in group]
        u = unify_trees(group, fps, depth_bound=depth_bound)
        if verbose:
            m = u.manifest
            print(f"[unify] {inst}: {m['n_files']} files -> {m['n_unique_states']} unique "
                  f"states ({m['n_reachable']} reachable); shared>=2: "
                  f"{m['n_shared_states']}, in all {len(group)}: {m['n_in_all_trees']}; "
                  f"within-tree dups {m['n_within_tree_duplicates']}; goal conflicts "
                  f"{m['n_goal_conflicts']}; delta improved by union "
                  f"{m['n_delta_improved_by_union']}; delta_root "
                  f"{m['delta_root_per_tree']} -> {m['delta_root_unified']}")
        out.append(u)
    return out


def report_unified(insts: Sequence[UnifiedInstance]) -> str:
    lines = ["  instance                  files  unique  shared  all  goalconf  dImproved  delta_root"]
    for u in insts:
        m = u.manifest
        lines.append(f"  {u.instance:24} {m['n_files']:>6} {m['n_unique_states']:>7} "
                     f"{m['n_shared_states']:>7} {m['n_in_all_trees']:>4} "
                     f"{m['n_goal_conflicts']:>9} {m['n_delta_improved_by_union']:>10} "
                     f"{m['delta_root_unified']}")
    return "\n".join(lines)

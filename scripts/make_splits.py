"""Write exp/trials/{basic,intermediate,hard,all}/split.csv from the baseline planner's results.

    python scripts/make_splits.py [exp/trials/deep_solutions.csv]

Rule: a problem the baseline solved (`status == correct`) is train, every other one
(memout, timeout, no-plan) is test. The tier trials keep the benchmark's domain and
problem names; `all` prefixes both with the tier (`basic-gossip`, `basic-gos-03-all`),
because gossip and blocks-world exist in every tier with colliding problem names and
the trainers key trees by problem name.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRIALS = REPO / "exp" / "trials"
TIERS = {"1-basic": "basic", "2-intermediate": "intermediate", "3-hard": "hard"}


def main(solutions: Path) -> None:
    rows = list(csv.DictReader(solutions.open()))
    per_trial: dict[str, list[tuple[str, str, str]]] = {t: [] for t in [*TIERS.values(), "all"]}
    for r in rows:
        _, tier_dir, domain, _, problem = r["problem"].split("/")
        tier, problem = TIERS[tier_dir], problem.removesuffix(".epddl")
        split = "train" if r["status"] == "correct" else "test"
        per_trial[tier].append((domain, problem, split))
        per_trial["all"].append((f"{tier}-{domain}", f"{tier}-{problem}", split))
    for trial, entries in per_trial.items():
        out = TRIALS / trial / "split.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["domain", "problem", "split"])
            w.writerows(sorted(entries))
        n_train = sum(s == "train" for *_, s in entries)
        print(f"{out.relative_to(REPO)}: {n_train} train / {len(entries) - n_train} test")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else TRIALS / "deep_solutions.csv")

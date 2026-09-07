"""Generation table -> (balanced, split) rows -> indices into a GraphStore.

The distance estimator is a per-STATE regressor: one state graph in (goal
inlined in merged mode, or the instance's goal graph beside it in separated
mode), one score out. No tree, no fringe. So a sample is just
    (index of the state graph, index of the goal graph or -1, depth, target).
The graphs themselves live once in a GraphStore (src/graph_store.py), parsed
once per instance and cached; batches are assembled by index. `samples.pt`
holds the dataframes, the index tensors and the cache manifest -- no graph
objects.

Layout read (flat, one level; the GNN never uses the strategy layout):
    <training_data>/<instance>/<instance>[_<TOKEN>]_depth_<D>.csv
                              /RawFiles/hash_{merged,separated}/NNNNNN.dot
                              /goal_tree.dot                (separated)
The CSV name token (BFS/S_DFS/...) and the zero-padded distances the newer
generator writes are both handled; a strategy directory level is refused with
an explicit message (see _build_df).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.graph_store import GraphStore, dot_paths_in_csv
from src.utils import KEYWORD_BITMASK

# Directory names the multi-strategy generator writes; the GNN reads the flat
# S_DFS layout only, so meeting these one level up is a wrong --folder-raw-data.
_STRATEGY_DIRS = {"BFS", "DFS", "S_DFS", "HFS"}
_DEPTH_RE = re.compile(r"_depth_(\d+)\.csv$")


def generation_depth_of(csv_path: str | Path) -> Optional[int]:
    """`<inst>_S_DFS_depth_25.csv` -> 25; None if the name carries no depth."""
    m = _DEPTH_RE.search(Path(csv_path).name)
    return int(m.group(1)) if m else None


class GraphDataPipeline:

    def __init__(
        self,
        folder_data: str,
        list_subset_train: List,
        dataset_type: str,
        kind_of_data: str,
        unreachable_state_value: int,
        max_percentage_per_class: float = 0.2,
        test_size: float = 0.2,
        use_goal: bool = True,
        use_depth: bool = True,
        random_state: int = 42,
        remove_unreachable_goal_states: bool = True,
    ):
        self.folder_data = Path(folder_data)
        self.list_subset_train = list_subset_train
        self.dataset_type = dataset_type
        self.data_kind = kind_of_data
        self.test_size = test_size
        self.use_goal = use_goal
        self.use_depth = use_depth

        # Separated state DOTs are goal-free: the goal lives in a per-instance
        # goal_tree.dot (the CSV `Goal` column) and MUST be fed as a separate
        # graph, so use_goal is mandatory.  In merged mode the goal is inlined
        # into each state DOT, so the separate goal graph is never loaded (it
        # would double-count, and the merged goal_tree.dot is not even written).
        if kind_of_data == "separated" and not use_goal:
            raise ValueError(
                "kind_of_data='separated' requires use_goal=True: separated "
                "state DOTs are goal-free, so the goal_tree.dot from the CSV "
                "`Goal` column must be fed as a separate goal graph."
            )
        self.unreachable_state_value = unreachable_state_value
        self.max_percentage_per_class = max_percentage_per_class
        self.random_state = random_state
        self.remove_unreachable_goal_states = remove_unreachable_goal_states

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self._instance_dirs: List[Path] = []
        self._csv_paths: List[Path] = []
        # Filled by build_store(): one GraphStore over every instance, and the
        # per-split index tensors (see module docstring).
        self.store: Optional[GraphStore] = None
        self.train_index: Dict[str, torch.Tensor] = {}
        self.test_index: Dict[str, torch.Tensor] = {}
        self.cache_files: List[str] = []
        self.generation_depths: Dict[str, Optional[int]] = {}

        self._build_df()

    def _get_all_items(self, folder: Path) -> List[Path]:
        return list(folder.iterdir())

    def _read_csv(self, csv_path: Path) -> pd.DataFrame:
        # The generation table is
        #   File Path, Depth, Distance From Goal, Goal,
        #   File Path Predecessor, Action
        # (the last two columns were added later).  Parse by HEADER so the
        # `Goal` column (the goal_tree.dot path in separated mode) is extracted
        # exactly — a fixed maxsplit would otherwise fold Predecessor+Action
        # into Goal.  Field values are DOT-file paths with no embedded commas,
        # so a full split is safe.
        COL_NAMES_CSV = [
            "File Path",
            "Depth",
            "Distance From Goal",
            "Goal",
        ]

        with csv_path.open(newline="") as f:
            header = next(f).rstrip("\n").split(",")
            col_idx = {name: header.index(name) for name in COL_NAMES_CSV}
            records = []
            for raw in f:
                parts = raw.rstrip("\n").split(",")
                records.append([parts[col_idx[name]] for name in COL_NAMES_CSV])
        df = pd.DataFrame(records, columns=COL_NAMES_CSV)
        df["Depth"] = pd.to_numeric(df["Depth"], errors="coerce")
        df["Distance From Goal"] = pd.to_numeric(
            df["Distance From Goal"], errors="coerce"
        )
        return df

    """def _read_csv(self, csv_path: Path) -> pd.DataFrame:
        COL_NAMES_CSV = [
            "Path Hash",
            "File Path",
            "Path Mapped",
            "Path Mapped Merged",
            "Depth",
            "Distance From Goal",
            "Goal",
        ]

        records = []
        with csv_path.open(newline="") as f:
            next(f)  # skip header
            for raw in f:
                parts = raw.rstrip("\n").split(",", len(COL_NAMES_CSV) - 1)
                records.append(parts)
        df = pd.DataFrame(records, columns=COL_NAMES_CSV)
        df["Depth"] = pd.to_numeric(df["Depth"], errors="coerce")
        df["Distance From Goal"] = pd.to_numeric(
            df["Distance From Goal"], errors="coerce"
        )
        return df"""

    def _my_train_test_split(self, df: pd.DataFrame, stratify_col="Distance From Goal"):
        # 1) pull out singleton labels
        vc = df[stratify_col].value_counts()
        singletons = vc[vc == 1].index
        is_single = df[stratify_col].isin(singletons)
        df_single = df[is_single]
        df_main = df[~is_single]

        # 2) prepare for stratified split
        X_main = df_main.drop(columns=[stratify_col])
        y_main = df_main[stratify_col]
        n_classes = y_main.nunique()
        n_samples = len(y_main)

        # 3) compute test_size fraction and enforce minimum
        desired_ratio = self.test_size / (1 + self.test_size)
        min_frac = n_classes / n_samples
        ts_main = max(desired_ratio, min_frac)

        # 4) do the split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_main,
            y_main,
            test_size=ts_main,
            random_state=self.random_state,
            stratify=y_main,
        )

        # 5) rebuild DataFrames and re‑attach the singletons to train
        train = pd.concat(
            [X_tr.assign(**{stratify_col: y_tr}), df_single], axis=0
        ).sample(frac=1, random_state=self.random_state)

        test = X_te.assign(**{stratify_col: y_te})
        return train, test

    def _balance_dataset(
        self, df: pd.DataFrame, feature: str = "Distance From Goal"
    ) -> pd.DataFrame:
        """
        Undersample each class in 'Distance From Goal' so that no class exceeds
        self.max_percentage_per_class * len(df) samples.
        """
        if self.remove_unreachable_goal_states:
            filtered_df = df.loc[
                df["Distance From Goal"] != self.unreachable_state_value
            ].copy()
        else:
            filtered_df = df.copy()

        original_len_df = len(filtered_df)
        # Compute the maximum allowed samples per class
        max_samples_per_class = int(self.max_percentage_per_class * original_len_df)

        balanced_splits = []
        # Group by target value
        for value, group in filtered_df.groupby(feature):
            count = len(group)
            if count > max_samples_per_class:
                # Randomly sample max_samples_per_class from this class
                sampled = group.sample(
                    n=max_samples_per_class, random_state=self.random_state
                )
                balanced_splits.append(sampled)
            else:
                # Keep the entire group if it's below the threshold
                balanced_splits.append(group)

        # Concatenate and shuffle the resulting DataFrame
        balanced_df = pd.concat(balanced_splits)
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state)
        balanced_df = balanced_df.reset_index(drop=True)
        return balanced_df

    def _build_df(self):
        train_frames, test_frames = [], []

        if not self.folder_data.is_dir():
            raise FileNotFoundError(f"training_data folder not found: {self.folder_data}")
        subdirs = sorted(p for p in self._get_all_items(self.folder_data) if p.is_dir())
        strat = [p.name for p in subdirs if p.name in _STRATEGY_DIRS]
        if strat:
            raise ValueError(
                f"{self.folder_data} holds the per-strategy layout ({strat}); the GNN "
                f"distance estimator reads the flat layout <training_data>/<instance>/ "
                f"(S_DFS, the generator's default). Point --folder-raw-data at a batch "
                f"generated WITHOUT --dataset-generation, or at one strategy folder."
            )

        for prob_dir in subdirs:
            if len(self.list_subset_train) > 0:
                if os.path.basename(prob_dir) not in self.list_subset_train:
                    continue
            csvs = sorted(p for p in self._get_all_items(prob_dir) if p.suffix == ".csv")
            if not csvs:
                continue
            if len(csvs) > 1:
                raise ValueError(
                    f"{prob_dir} holds {len(csvs)} generation tables ({[c.name for c in csvs]}); "
                    f"one table per instance folder -- remove the extra one."
                )
            csv = csvs[0]
            df = self._read_csv(csv)
            df["_instance"] = prob_dir.name
            df = self._balance_dataset(df)
            train_df, test_df = self._my_train_test_split(df)
            train_frames.append(train_df)
            test_frames.append(test_df)
            self._instance_dirs.append(Path(prob_dir))
            self._csv_paths.append(csv)
            self.generation_depths[prob_dir.name] = generation_depth_of(csv)

        if len(train_frames) == 0 or len(test_frames) == 0:
            raise ValueError(
                f"train samples = {len(train_frames)}, test samples = {len(test_frames)} "
                f"(no instance folder with a .csv under {self.folder_data}"
                f"{' matching ' + str(self.list_subset_train) if self.list_subset_train else ''})"
            )
        self.train_df = pd.concat(train_frames, ignore_index=True)
        self.test_df = pd.concat(test_frames, ignore_index=True)

    # ---- graphs: parse once per instance, cache, index -----------------------

    def build_store(self, cache_dir: Optional[str | Path] = None,
                    workers: Optional[int] = None, verbose: bool = True) -> GraphStore:
        """Parse every DOT the tables reference (all rows, not just the kept
        ones, so the cache is reusable across seeds/balancing) and map each
        kept row to its graph index.

        Cache: <cache_dir>/<instance>.<DATASET_TYPE>.pt, one per instance. A
        cache whose path list differs from the table's is rebuilt.
        """
        bitmask = self.dataset_type == KEYWORD_BITMASK
        load_separate_goal = self.data_kind == "separated" and self.use_goal
        # Merged tables still carry a `Goal` column, but the file is never
        # written in merged mode (the goal is inlined in every state DOT).
        columns = ("File Path", "Goal") if load_separate_goal else ("File Path",)
        stores: List[GraphStore] = []
        self.cache_files = []
        for inst_dir, csv in zip(self._instance_dirs, self._csv_paths):
            paths = dot_paths_in_csv(csv, columns=columns)
            cache_file = None
            if cache_dir is not None:
                cache_file = Path(cache_dir) / f"{inst_dir.name}.{self.dataset_type}.pt"
                self.cache_files.append(str(cache_file))
            stores.append(GraphStore.from_paths(paths, bitmask=bitmask, cache_file=cache_file,
                                                workers=workers, verbose=verbose))
        self.store = GraphStore.concat(stores)
        self.train_index = self._index_split(self.train_df, load_separate_goal)
        self.test_index = self._index_split(self.test_df, load_separate_goal)
        return self.store

    def _index_split(self, df: pd.DataFrame, with_goal: bool) -> Dict[str, torch.Tensor]:
        idx = self.store.index
        try:
            state = torch.tensor([idx[p.strip()] for p in df["File Path"]], dtype=torch.int64)
        except KeyError as e:
            raise KeyError(f"state DOT {e} referenced by the table is not in the graph store") from e
        out = {
            "state": state,
            "target": torch.tensor(df["Distance From Goal"].to_numpy(), dtype=torch.float32),
            "depth": torch.tensor(df["Depth"].to_numpy(), dtype=torch.float32),
        }
        if with_goal:
            out["goal"] = torch.tensor([idx[p.strip()] for p in df["Goal"]], dtype=torch.int64)
        return out

    def instance_names(self) -> List[str]:
        return [p.name for p in self._instance_dirs]

    def save(self, out_dir: str, extra_params: Optional[Dict[str, Any]] = None):
        """samples.pt: dataframes, index tensors and the cache manifest. The
        graphs are NOT in here -- they are in the per-instance caches."""
        if self.store is None:
            raise ValueError("call build_store() before save()")
        out = Path(out_dir)
        payload = {
            "format": 2,
            "params": {
                "folder_data": str(self.folder_data),
                "ordering": self.dataset_type,
                "data_kind": self.data_kind,
                "max_percentage_per_class": self.max_percentage_per_class,
                "use_goal": self.use_goal,
                "use_depth": self.use_depth,
                "instances": self.instance_names(),
                "generation_depths": dict(self.generation_depths),
                "cache_files": list(self.cache_files),
                **(extra_params or {}),
            },
            "train_df": self.train_df,
            "test_df": self.test_df,
            "train_index": self.train_index,
            "test_index": self.test_index,
            "store_paths": self.store.paths,
        }
        torch.save(payload, out)
        return out

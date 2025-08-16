from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import preprocess_sample

COL_NAMES_CSV = [
    "Path Hash",
    "Path Hash Merged",
    "Path Mapped",
    "Path Mapped Merged",
    "Depth",
    "Distance From Goal",
    "Goal",
]


class GraphDataPipeline:

    def __init__(
        self,
        folder_data: str,
        list_subset_train: List,
        kind_of_ordering: str,
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
        self.ordering = kind_of_ordering
        self.data_kind = kind_of_data
        self.test_size = test_size
        self.use_goal = use_goal
        self.use_depth = use_depth
        self.unreachable_state_value = unreachable_state_value
        self.max_percentage_per_class = max_percentage_per_class
        self.random_state = random_state
        self.remove_unreachable_goal_states = remove_unreachable_goal_states

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.train_samples: List[Dict[str, Any]] = []
        self.test_samples: List[Dict[str, Any]] = []

        self._build_df()
        self._load_samples()

    def _get_all_items(self, folder: Path) -> List[Path]:
        return list(folder.iterdir())

    def _read_csv(self, csv_path: Path) -> pd.DataFrame:
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

        # determine which path column to keep
        key = f"Path {'Hash' if self.ordering == 'hash' else 'Mapped'}"
        key += " Merged" if self.data_kind == "merged" else ""
        df = df[["Depth", "Distance From Goal", key, "Goal"]]
        df = df.rename(columns={key: "Path State"})
        return df

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
        for prob_dir in self._get_all_items(self.folder_data):
            if len(self.list_subset_train) > 0:
                if os.path.basename(prob_dir) not in self.list_subset_train:
                    continue
            csv = next(
                (p for p in self._get_all_items(prob_dir) if p.suffix == ".csv"), None
            )
            if not csv:
                continue
            df = self._read_csv(csv)
            df = self._balance_dataset(df)
            train_df, test_df = self._my_train_test_split(df)
            train_frames.append(train_df)
            test_frames.append(test_df)

        if len(train_frames) == 0 or len(test_frames) == 0:
            raise ValueError(
                f"train samples = {len(train_frames)}, test samples = {len(test_frames)}"
            )
        self.train_df = pd.concat(train_frames, ignore_index=True)
        # self.train_df = self._balance_dataset(self.train_df)
        self.test_df = pd.concat(test_frames, ignore_index=True)
        # self.test_df = self._balance_dataset(self.test_df)

    def _load_samples(self):

        s = [self.train_samples, self.test_samples]
        t = [self.train_df, self.test_df]

        for i, df in enumerate(t):
            if df is None:
                raise ValueError("Call build_df() first.")

            desc = "Building train samples..." if i == 0 else "Building test samples..."

            for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
                sample = preprocess_sample(
                    row["Path State"],
                    int(row["Depth"]) if self.use_depth else None,
                    int(row["Distance From Goal"]),
                    row["Goal"] if self.use_goal else None,
                )
                s[i].append(sample)

    def save(self, out_dir: str, extra_params: Optional[Dict[str, Any]] = None):
        out = Path(out_dir)
        payload = {
            "params": {
                "folder_data": str(self.folder_data),
                "ordering": self.ordering,
                "data_kind": self.data_kind,
                "max_percentage_per_class": self.max_percentage_per_class,
                "use_goal": self.use_goal,
                **(extra_params or {}),
            },
            "train_df": self.train_df,
            "test_df": self.test_df,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
        }
        torch.save(payload, out)
        return out

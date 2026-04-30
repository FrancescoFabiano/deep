from __future__ import annotations

import json
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from src.models.frontier_policy import FrontierPolicyNetwork
from torch import nn
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence

FAILURE_REWARD_VALUE = -1.0
FAILURE_EPS = 1e-9
REGIME_ALL = "all"
REGIME_NO_FAILURE = "no_failure"
REGIME_WITH_FAILURE = "with_failure"
REGIME_ORDER = [REGIME_ALL, REGIME_NO_FAILURE, REGIME_WITH_FAILURE]


class OnnxFrontierPolicyWrapper(nn.Module):
    def __init__(self, core: FrontierPolicyNetwork):
        super().__init__()
        self.core = core

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membership: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.core(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            membership=membership,
            candidate_batch=None,
            mask=mask,
        )


class OnnxFrontierPolicySeparatedWrapper(nn.Module):
    def __init__(self, core: FrontierPolicyNetwork):
        super().__init__()
        self.core = core

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membership: torch.Tensor,
        goal_node_features: torch.Tensor,
        goal_edge_index: torch.Tensor,
        goal_edge_attr: torch.Tensor,
        goal_batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.core(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            membership=membership,
            candidate_batch=None,
            mask=mask,
            goal_node_features=goal_node_features,
            goal_edge_index=goal_edge_index,
            goal_edge_attr=goal_edge_attr,
            goal_batch=goal_batch,
        )


class RLFrontierTrainer:
    def __init__(
        self,
        model: FrontierPolicyNetwork,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str | torch.device | None = None,
        reward_formulation: str = "negative_distance",
        kind_of_data: str = "merged",
        max_grad_norm: float = 0.0,
        m_failed_state: Optional[float] = None,
        **_: object,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.reward_formulation = reward_formulation
        self.kind_of_data = kind_of_data
        self.max_grad_norm = float(max_grad_norm)
        self.m_failed_state = (
            float(m_failed_state) if m_failed_state is not None else float("inf")
        )
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be >= 0.")

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target_device = self._model_device()
        self.device = target_device
        out = {}
        for k, v in batch.items():
            out[k] = v.to(target_device) if isinstance(v, torch.Tensor) else v
        return out

    def _build_model_kwargs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, torch.Tensor] = {
            "node_features": batch["node_features"],
            "edge_index": batch["edge_index"],
            "edge_attr": batch["edge_attr"],
            "membership": batch["membership"],
            "candidate_batch": batch["candidate_batch"],
        }
        if "pool_node_index" in batch and "pool_membership" in batch:
            kwargs["pool_node_index"] = batch["pool_node_index"]
            kwargs["pool_membership"] = batch["pool_membership"]
        if "goal_node_features" in batch:
            kwargs["goal_node_features"] = batch["goal_node_features"]
            kwargs["goal_edge_index"] = batch["goal_edge_index"]
            kwargs["goal_edge_attr"] = batch["goal_edge_attr"]
            kwargs["goal_batch"] = batch["goal_batch"]
        target_device = self._model_device()
        for key, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.device != target_device:
                kwargs[key] = value.to(target_device)
        return kwargs

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _iqm_iqr_stats(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "q1": 0.0,
                "q3": 0.0,
                "iqm": 0.0,
                "iqr": 0.0,
                "iqr_std": 0.0,
            }
        t = torch.tensor(values, dtype=torch.float32)
        mean = float(t.mean().item())
        std = float(t.std(unbiased=False).item())
        q1 = float(torch.quantile(t, 0.25).item())
        q3 = float(torch.quantile(t, 0.75).item())
        trimmed = t[(t >= q1) & (t <= q3)]
        if trimmed.numel() == 0:
            trimmed = t
        iqm = float(trimmed.mean().item())
        iqr = float(q3 - q1)
        iqr_std = float(trimmed.std(unbiased=False).item()) if trimmed.numel() > 0 else 0.0
        return {
            "mean": mean,
            "std": std,
            "q1": q1,
            "q3": q3,
            "iqm": iqm,
            "iqr": iqr,
            "iqr_std": iqr_std,
        }

    @staticmethod
    def _regime_mask(frontier_has_failure: Sequence[int], regime: str) -> List[bool]:
        if regime == REGIME_ALL:
            return [True] * len(frontier_has_failure)
        if regime == REGIME_NO_FAILURE:
            return [int(x) == 0 for x in frontier_has_failure]
        if regime == REGIME_WITH_FAILURE:
            return [int(x) == 1 for x in frontier_has_failure]
        raise ValueError(f"Unsupported regime: {regime}")

    @classmethod
    def _build_regime_metrics(
        cls,
        rewards: Sequence[float],
        accuracies: Sequence[int],
        frontier_has_failure: Sequence[int],
        frontier_sizes: Sequence[int],
        abs_reward_gaps: Sequence[float],
    ) -> Dict[str, Dict[str, object]]:
        def _size_curve(
            sizes: Sequence[int],
            values: Sequence[float],
        ) -> Dict[str, List[float]]:
            by_size: Dict[int, List[float]] = {}
            for s, v in zip(sizes, values):
                by_size.setdefault(int(s), []).append(float(v))
            ordered_sizes = sorted(by_size.keys())
            iqm = []
            iqr_std = []
            for s in ordered_sizes:
                stats = cls._iqm_iqr_stats(by_size[s])
                iqm.append(float(stats["iqm"]))
                iqr_std.append(float(stats["iqr_std"]))
            return {
                "sizes": [int(s) for s in ordered_sizes],
                "iqm": iqm,
                "iqr_std": iqr_std,
            }

        regimes: Dict[str, Dict[str, object]] = {}
        for regime in REGIME_ORDER:
            mask = cls._regime_mask(frontier_has_failure, regime)
            regime_rewards = [float(r) for r, m in zip(rewards, mask) if m]
            regime_accuracies = [int(a) for a, m in zip(accuracies, mask) if m]
            regime_sizes = [int(s) for s, m in zip(frontier_sizes, mask) if m]
            regime_abs_reward_gaps = [float(g) for g, m in zip(abs_reward_gaps, mask) if m]
            regimes[regime] = {
                "n_frontiers": len(regime_rewards),
                "rewards": regime_rewards,
                "accuracies": regime_accuracies,
                "abs_reward_gaps": regime_abs_reward_gaps,
                "frontier_sizes": regime_sizes,
                "reward_stats": cls._iqm_iqr_stats(regime_rewards),
                "accuracy_stats": cls._iqm_iqr_stats(regime_accuracies),
                "abs_reward_gap_stats": cls._iqm_iqr_stats(regime_abs_reward_gaps),
                "reward_by_size": _size_curve(regime_sizes, regime_rewards),
                "accuracy_by_size": _size_curve(regime_sizes, regime_accuracies),
                "abs_reward_gap_by_size": _size_curve(regime_sizes, regime_abs_reward_gaps),
            }
        return regimes

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _plot_iqm_with_iqr_std_band(
        out_path: Path,
        x: Sequence[int],
        iqm: Sequence[float],
        iqr_std: Sequence[float],
        xlabel: str,
        ylabel: str,
        y_lim: Optional[tuple[float, float]] = None,
        title: str = "",
    ) -> None:
        plt.figure(figsize=(8, 4), dpi=300)
        if iqm:
            lo = [m - s for m, s in zip(iqm, iqr_std)]
            hi = [m + s for m, s in zip(iqm, iqr_std)]
            plt.plot(x, iqm, label="IQM")
            plt.fill_between(x, lo, hi, alpha=0.25, label="IQM ± IQR-STD")
            plt.legend(loc="best")
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if y_lim is not None:
            plt.ylim(y_lim[0], y_lim[1])
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _plot_line_with_std_band(
        out_path: Path,
        x: Sequence[int],
        means: Sequence[float],
        stds: Sequence[float],
        xlabel: str,
        ylabel: str,
        title: str = "",
        y_lim: Optional[tuple[float, float]] = None,
    ) -> None:
        plt.figure(figsize=(8, 4), dpi=300)
        if means:
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            plt.plot(x, means)
            plt.fill_between(x, lo, hi, alpha=0.25)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if y_lim is not None:
            plt.ylim(y_lim[0], y_lim[1])
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _plot_frontier_size_curve_with_band(
        out_path: Path,
        sizes: Sequence[int],
        iqm: Sequence[float],
        iqr_std: Sequence[float],
        ylabel: str,
        title: str,
        y_lim: Optional[tuple[float, float]] = None,
    ) -> None:
        plt.figure(figsize=(8, 4), dpi=300)
        if sizes and iqm and iqr_std and len(sizes) == len(iqm) == len(iqr_std):
            lo = [m - s for m, s in zip(iqm, iqr_std)]
            hi = [m + s for m, s in zip(iqm, iqr_std)]
            plt.plot(sizes, iqm, marker="o", label="IQM")
            plt.fill_between(sizes, lo, hi, alpha=0.25, label="IQM ± IQR-STD")
            plt.legend(loc="best")
            plt.xlabel("Frontier Size")
            plt.ylabel(ylabel)
            if y_lim is not None:
                plt.ylim(y_lim[0], y_lim[1])
        else:
            plt.text(0.5, 0.5, "No frontiers in this regime", ha="center", va="center")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def _save_eval_step_regime_plots(
        self,
        eval_dir: Path,
        eval_name: str,
        epoch: int,
        eval_step: int,
        metrics: Dict[str, object],
    ) -> None:
        step_dir = eval_dir / f"step_{eval_step:04d}_epoch_{epoch:04d}"
        self._ensure_dir(step_dir)
        regimes = metrics["regimes"]

        for regime in REGIME_ORDER:
            regime_metrics = regimes[regime]
            regime_title = f"{eval_name.upper()} | {regime} | epoch {epoch}"
            reward_curve = regime_metrics["reward_by_size"]
            acc_curve = regime_metrics["accuracy_by_size"]
            self._plot_frontier_size_curve_with_band(
                out_path=step_dir / f"{regime}_reward_by_frontier_size_iqm_iqrstd.png",
                sizes=reward_curve["sizes"],
                iqm=reward_curve["iqm"],
                iqr_std=reward_curve["iqr_std"],
                ylabel="Reward",
                y_lim=(-1.0, 0.05),
                title=f"{regime_title} | Reward by Frontier Size (IQM ± IQR-STD)",
            )
            self._plot_frontier_size_curve_with_band(
                out_path=step_dir / f"{regime}_accuracy_by_frontier_size_iqm_iqrstd.png",
                sizes=acc_curve["sizes"],
                iqm=acc_curve["iqm"],
                iqr_std=acc_curve["iqr_std"],
                ylabel="Accuracy",
                y_lim=(-0.05, 1.05),
                title=f"{regime_title} | Accuracy by Frontier Size (IQM ± IQR-STD)",
            )
            abs_gap_curve = regime_metrics["abs_reward_gap_by_size"]
            self._plot_frontier_size_curve_with_band(
                out_path=step_dir / f"{regime}_abs_reward_gap_by_frontier_size_iqm_iqrstd.png",
                sizes=abs_gap_curve["sizes"],
                iqm=abs_gap_curve["iqm"],
                iqr_std=abs_gap_curve["iqr_std"],
                ylabel="|best reward - taken reward|",
                y_lim=(0.0, 1.05),
                title=f"{regime_title} | Absolute Reward Gap by Frontier Size (IQM ± IQR-STD)",
            )

        with (step_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

    def _save_global_regime_history_plots(
        self,
        eval_dir: Path,
        eval_name: str,
        regime_history: Dict[str, Dict[str, List[float]]],
    ) -> None:
        for regime in REGIME_ORDER:
            hist = regime_history[regime]
            epochs = [int(e) for e in hist["epochs"]]
            self._plot_line_with_std_band(
                out_path=eval_dir / f"{regime}_global_accuracy_over_epochs.png",
                x=epochs,
                means=hist["accuracy_mean"],
                stds=hist["accuracy_std"],
                xlabel="Epoch",
                ylabel="Global Accuracy",
                title=f"{eval_name.upper()} | {regime} | Global Accuracy (mean ± std)",
                y_lim=(-0.05, 1.05),
            )
            self._plot_line_with_std_band(
                out_path=eval_dir / f"{regime}_global_reward_over_epochs.png",
                x=epochs,
                means=hist["reward_mean"],
                stds=hist["reward_std"],
                xlabel="Epoch",
                ylabel="Global Reward",
                title=f"{eval_name.upper()} | {regime} | Global Reward (mean ± std)",
                y_lim=(-1.0, 0.05),
            )
            self._plot_line_with_std_band(
                out_path=eval_dir / f"{regime}_global_abs_reward_gap_over_epochs.png",
                x=epochs,
                means=hist["abs_reward_gap_mean"],
                stds=hist["abs_reward_gap_std"],
                xlabel="Epoch",
                ylabel="|best reward - taken reward|",
                title=f"{eval_name.upper()} | {regime} | Absolute Reward Gap (mean ± std)",
                y_lim=(0.0, 1.05),
            )

    @staticmethod
    def _append_regime_history(
        regime_history: Dict[str, Dict[str, List[float]]],
        epoch: int,
        regimes: Dict[str, Dict[str, object]],
    ) -> None:
        for regime in REGIME_ORDER:
            stats_acc = regimes[regime]["accuracy_stats"]
            stats_reward = regimes[regime]["reward_stats"]
            stats_abs_gap = regimes[regime]["abs_reward_gap_stats"]
            regime_history[regime]["epochs"].append(int(epoch))
            regime_history[regime]["accuracy_mean"].append(float(stats_acc["mean"]))
            regime_history[regime]["accuracy_std"].append(float(stats_acc["std"]))
            regime_history[regime]["reward_mean"].append(float(stats_reward["mean"]))
            regime_history[regime]["reward_std"].append(float(stats_reward["std"]))
            regime_history[regime]["abs_reward_gap_mean"].append(float(stats_abs_gap["mean"]))
            regime_history[regime]["abs_reward_gap_std"].append(float(stats_abs_gap["std"]))

    @staticmethod
    def _empty_regime_history() -> Dict[str, Dict[str, List[float]]]:
        return {
            regime: {
                "epochs": [],
                "accuracy_mean": [],
                "accuracy_std": [],
                "reward_mean": [],
                "reward_std": [],
                "abs_reward_gap_mean": [],
                "abs_reward_gap_std": [],
            }
            for regime in REGIME_ORDER
        }

    @staticmethod
    def _collect_frontier_decisions(
        logits: torch.Tensor,
        rewards: torch.Tensor,
        frontier_ptr: torch.Tensor,
    ) -> Dict[str, List[float]]:
        chosen_rewards: List[float] = []
        accuracies: List[int] = []
        frontier_has_failure: List[int] = []
        frontier_sizes: List[int] = []
        abs_reward_gaps: List[float] = []

        for i in range(frontier_ptr.numel() - 1):
            s = int(frontier_ptr[i].item())
            e = int(frontier_ptr[i + 1].item())
            if e <= s:
                continue
            frontier_logits = logits[s:e]
            frontier_rewards = rewards[s:e]
            chosen_local = int(torch.argmax(frontier_logits).item())
            chosen_reward = float(frontier_rewards[chosen_local].item())
            best_reward = float(frontier_rewards.max().item())
            abs_gap = abs(best_reward - chosen_reward)
            is_correct = int(abs_gap <= FAILURE_EPS)
            has_failure = int(
                bool(
                    (frontier_rewards <= (FAILURE_REWARD_VALUE + FAILURE_EPS))
                    .any()
                    .item()
                )
            )

            chosen_rewards.append(chosen_reward)
            accuracies.append(is_correct)
            frontier_has_failure.append(has_failure)
            frontier_sizes.append(int(e - s))
            abs_reward_gaps.append(float(abs_gap))

        return {
            "chosen_rewards": chosen_rewards,
            "accuracies": accuracies,
            "frontier_has_failure": frontier_has_failure,
            "frontier_sizes": frontier_sizes,
            "abs_reward_gaps": abs_reward_gaps,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = self.model(**self._build_model_kwargs(batch))
        frontier_ptr = batch["frontier_ptr"]
        rewards = batch.get("reward_target", batch["rewards"])

        reward_terms = []
        for i in range(frontier_ptr.numel() - 1):
            s = int(frontier_ptr[i].item())
            e = int(frontier_ptr[i + 1].item())
            if e <= s:
                continue
            probs = torch.softmax(logits[s:e], dim=0)
            reward_terms.append((probs * rewards[s:e]).sum())

        if reward_terms:
            expected_reward = torch.stack(reward_terms).mean()
        else:
            expected_reward = logits.new_tensor(0.0)
        total_loss = -expected_reward

        return {
            "total_loss": total_loss,
            "reward_loss": total_loss,
            "ranking_loss": total_loss.new_tensor(0.0),
            "failure_avoidance_loss": total_loss.new_tensor(0.0),
            "entropy_bonus": total_loss.new_tensor(0.0),
            "logit_l2": total_loss.new_tensor(0.0),
            "expected_reward": expected_reward,
            "logits": logits,
        }

    def evaluate(self, loader, verbose: bool = False) -> Dict[str, object]:
        self.model.eval()
        chosen_rewards: List[float] = []
        accuracies: List[int] = []
        frontier_has_failure: List[int] = []
        frontier_sizes: List[int] = []
        abs_reward_gaps: List[float] = []

        with torch.no_grad():
            for raw_batch in loader:
                batch = self._move_to_device(raw_batch)
                logits = self.model(**self._build_model_kwargs(batch))
                frontier_ptr = batch["frontier_ptr"]
                rewards = batch.get("reward_target", batch["rewards"])
                batch_decisions = self._collect_frontier_decisions(
                    logits=logits,
                    rewards=rewards,
                    frontier_ptr=frontier_ptr,
                )
                chosen_rewards.extend(batch_decisions["chosen_rewards"])
                accuracies.extend(batch_decisions["accuracies"])
                frontier_has_failure.extend(batch_decisions["frontier_has_failure"])
                frontier_sizes.extend(batch_decisions["frontier_sizes"])
                abs_reward_gaps.extend(batch_decisions["abs_reward_gaps"])

        regimes = self._build_regime_metrics(
            rewards=chosen_rewards,
            accuracies=accuracies,
            frontier_has_failure=frontier_has_failure,
            frontier_sizes=frontier_sizes,
            abs_reward_gaps=abs_reward_gaps,
        )
        all_reward_stats = regimes[REGIME_ALL]["reward_stats"]
        all_accuracy_stats = regimes[REGIME_ALL]["accuracy_stats"]
        all_abs_gap_stats = regimes[REGIME_ALL]["abs_reward_gap_stats"]

        metrics: Dict[str, object] = {
            "n_frontiers": len(chosen_rewards),
            "chosen_rewards": chosen_rewards,
            "accuracies": accuracies,
            "frontier_has_failure": frontier_has_failure,
            "frontier_sizes": frontier_sizes,
            "abs_reward_gaps": abs_reward_gaps,
            "regimes": regimes,
            "mean_reward": float(all_reward_stats["mean"]),
            "std_reward": float(all_reward_stats["std"]),
            "iqm_reward": float(all_reward_stats["iqm"]),
            "iqr_reward": float(all_reward_stats["iqr"]),
            "iqr_std_reward": float(all_reward_stats["iqr_std"]),
            "mean_accuracy": float(all_accuracy_stats["mean"]),
            "std_accuracy": float(all_accuracy_stats["std"]),
            "mean_abs_reward_gap": float(all_abs_gap_stats["mean"]),
            "std_abs_reward_gap": float(all_abs_gap_stats["std"]),
            "iqm_abs_reward_gap": float(all_abs_gap_stats["iqm"]),
            "iqr_abs_reward_gap": float(all_abs_gap_stats["iqr"]),
            "iqr_std_abs_reward_gap": float(all_abs_gap_stats["iqr_std"]),
            "oracle_accuracy": float(all_accuracy_stats["mean"]),
        }

        # Compatibility aliases for older callers.
        n_frontiers = len(chosen_rewards)
        n_failure_frontiers = int(sum(frontier_has_failure))
        n_mixed_frontiers = n_failure_frontiers
        metrics["n_mixed_frontiers"] = n_mixed_frontiers
        metrics["n_failure_frontiers"] = n_failure_frontiers
        metrics["failure_frontier_rate"] = (
            float(n_failure_frontiers / n_frontiers) if n_frontiers > 0 else 0.0
        )
        metrics["n_failure_choices_on_mixed"] = 0
        metrics["failure_choice_rate_on_mixed"] = 0.0
        metrics["normalized_regret_mean"] = 0.0
        metrics["normalized_regret_std"] = 0.0
        metrics["normalized_regrets"] = [0.0 for _ in chosen_rewards]

        if verbose:
            print("\nEvaluation summary")
            print(f"frontiers        : {metrics['n_frontiers']}")
            print(
                f"mean reward      : {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}"
            )
            print(
                f"iqm reward       : {metrics['iqm_reward']:.4f} ± {metrics['iqr_std_reward']:.4f} (IQR-STD band)"
            )
            print(
                f"mean accuracy    : {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}"
            )
            print(
                f"abs reward gap   : {metrics['mean_abs_reward_gap']:.4f} ± {metrics['std_abs_reward_gap']:.4f}"
            )

        return metrics

    def train(
        self,
        train_loader,
        eval_loader,
        n_epochs: int,
        checkpoint_dir: str,
        model_name: str = "frontier_policy",
        eval_every: int = 1,
        early_stopping_patience_evals: int = 0,
        eval2_loader=None,
        eval_loader_name: str = "eval",
        eval2_loader_name: str = "eval2",
    ) -> Dict[str, object]:
        checkpoint = Path(checkpoint_dir)
        self._ensure_dir(checkpoint)

        metrics_root = checkpoint / "metrics"
        train_metrics_dir = metrics_root / "train"
        eval_metrics_dir = metrics_root / str(eval_loader_name)
        eval2_metrics_dir = metrics_root / str(eval2_loader_name)
        self._ensure_dir(train_metrics_dir)
        self._ensure_dir(eval_metrics_dir)
        self._ensure_dir(eval2_metrics_dir)

        history: Dict[str, object] = {
            "train_epochs": [],
            "train_iqm_reward": [],
            "train_iqr_std_reward": [],
            "train_mean_reward": [],
            "train_std_reward": [],
            "train_mean_loss": [],
            "eval_epochs": [],
            "eval_iqm_reward": [],
            "eval_iqr_std_reward": [],
            "eval_iqm_abs_reward_gap": [],
            "eval_iqr_std_abs_reward_gap": [],
            "eval2_epochs": [],
            "eval2_iqm_reward": [],
            "eval2_iqr_std_reward": [],
            "eval2_iqm_abs_reward_gap": [],
            "eval2_iqr_std_abs_reward_gap": [],
        }

        eval_regime_history = self._empty_regime_history()
        eval2_regime_history = self._empty_regime_history()

        best_eval_reward = -float("inf")
        best_eval_epoch = -1
        best_eval_metrics: Dict[str, object] = {}
        no_improve_eval_steps = 0
        eval_step = 0

        epoch_pbar = tqdm(range(n_epochs), desc="training", leave=True)
        for epoch in epoch_pbar:
            self.model.train()
            epoch_batch_rewards: List[float] = []
            epoch_batch_losses: List[float] = []

            for raw_batch in train_loader:
                batch = self._move_to_device(raw_batch)
                self.optimizer.zero_grad()
                loss_terms = self.compute_loss(batch)
                loss = loss_terms["total_loss"]
                loss.backward()
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_batch_rewards.append(float(loss_terms["expected_reward"].detach().item()))
                epoch_batch_losses.append(float(loss_terms["total_loss"].detach().item()))

            train_stats = self._iqm_iqr_stats(epoch_batch_rewards)
            history["train_epochs"].append(epoch + 1)
            history["train_iqm_reward"].append(float(train_stats["iqm"]))
            history["train_iqr_std_reward"].append(float(train_stats["iqr_std"]))
            history["train_mean_reward"].append(float(train_stats["mean"]))
            history["train_std_reward"].append(float(train_stats["std"]))
            history["train_mean_loss"].append(self._safe_mean(epoch_batch_losses))

            self._plot_iqm_with_iqr_std_band(
                out_path=train_metrics_dir / "reward_iqm_iqr_std_over_epochs.png",
                x=history["train_epochs"],
                iqm=history["train_iqm_reward"],
                iqr_std=history["train_iqr_std_reward"],
                xlabel="Epoch",
                ylabel="Train Reward",
                y_lim=(-1.0, 0.0),
                title="Train Reward IQM ± IQR-STD",
            )

            if (epoch + 1) % eval_every == 0:
                eval_step += 1
                eval_metrics = self.evaluate(eval_loader, verbose=False)
                history["eval_epochs"].append(epoch + 1)
                history["eval_iqm_reward"].append(float(eval_metrics["iqm_reward"]))
                history["eval_iqr_std_reward"].append(float(eval_metrics["iqr_std_reward"]))
                history["eval_iqm_abs_reward_gap"].append(
                    float(eval_metrics["iqm_abs_reward_gap"])
                )
                history["eval_iqr_std_abs_reward_gap"].append(
                    float(eval_metrics["iqr_std_abs_reward_gap"])
                )
                self._append_regime_history(
                    regime_history=eval_regime_history,
                    epoch=epoch + 1,
                    regimes=eval_metrics["regimes"],
                )
                self._save_eval_step_regime_plots(
                    eval_dir=eval_metrics_dir,
                    eval_name=str(eval_loader_name),
                    epoch=epoch + 1,
                    eval_step=eval_step,
                    metrics=eval_metrics,
                )
                self._plot_iqm_with_iqr_std_band(
                    out_path=eval_metrics_dir / "reward_iqm_iqr_std_over_eval_steps.png",
                    x=history["eval_epochs"],
                    iqm=history["eval_iqm_reward"],
                    iqr_std=history["eval_iqr_std_reward"],
                    xlabel="Epoch (eval checkpoints)",
                    ylabel=f"{str(eval_loader_name)} Reward",
                    y_lim=(-1.0, 0.0),
                    title=f"{str(eval_loader_name)} Reward IQM ± IQR-STD",
                )
                self._plot_iqm_with_iqr_std_band(
                    out_path=eval_metrics_dir / "abs_reward_gap_iqm_iqr_std_over_eval_steps.png",
                    x=history["eval_epochs"],
                    iqm=history["eval_iqm_abs_reward_gap"],
                    iqr_std=history["eval_iqr_std_abs_reward_gap"],
                    xlabel="Epoch (eval checkpoints)",
                    ylabel="|best reward - taken reward|",
                    y_lim=(0.0, 1.05),
                    title=f"{str(eval_loader_name)} Absolute Reward Gap IQM ± IQR-STD",
                )
                self._save_global_regime_history_plots(
                    eval_dir=eval_metrics_dir,
                    eval_name=str(eval_loader_name),
                    regime_history=eval_regime_history,
                )

                if eval2_loader is not None:
                    eval2_metrics = self.evaluate(eval2_loader, verbose=False)
                    history["eval2_epochs"].append(epoch + 1)
                    history["eval2_iqm_reward"].append(float(eval2_metrics["iqm_reward"]))
                    history["eval2_iqr_std_reward"].append(float(eval2_metrics["iqr_std_reward"]))
                    history["eval2_iqm_abs_reward_gap"].append(
                        float(eval2_metrics["iqm_abs_reward_gap"])
                    )
                    history["eval2_iqr_std_abs_reward_gap"].append(
                        float(eval2_metrics["iqr_std_abs_reward_gap"])
                    )
                    self._append_regime_history(
                        regime_history=eval2_regime_history,
                        epoch=epoch + 1,
                        regimes=eval2_metrics["regimes"],
                    )
                    self._save_eval_step_regime_plots(
                        eval_dir=eval2_metrics_dir,
                        eval_name=str(eval2_loader_name),
                        epoch=epoch + 1,
                        eval_step=eval_step,
                        metrics=eval2_metrics,
                    )
                    self._plot_iqm_with_iqr_std_band(
                        out_path=eval2_metrics_dir / "reward_iqm_iqr_std_over_eval_steps.png",
                        x=history["eval2_epochs"],
                        iqm=history["eval2_iqm_reward"],
                        iqr_std=history["eval2_iqr_std_reward"],
                        xlabel="Epoch (eval checkpoints)",
                        ylabel=f"{str(eval2_loader_name)} Reward",
                        y_lim=(-1.0, 0.0),
                        title=f"{str(eval2_loader_name)} Reward IQM ± IQR-STD",
                    )
                    self._plot_iqm_with_iqr_std_band(
                        out_path=eval2_metrics_dir / "abs_reward_gap_iqm_iqr_std_over_eval_steps.png",
                        x=history["eval2_epochs"],
                        iqm=history["eval2_iqm_abs_reward_gap"],
                        iqr_std=history["eval2_iqr_std_abs_reward_gap"],
                        xlabel="Epoch (eval checkpoints)",
                        ylabel="|best reward - taken reward|",
                        y_lim=(0.0, 1.05),
                        title=f"{str(eval2_loader_name)} Absolute Reward Gap IQM ± IQR-STD",
                    )
                    self._save_global_regime_history_plots(
                        eval_dir=eval2_metrics_dir,
                        eval_name=str(eval2_loader_name),
                        regime_history=eval2_regime_history,
                    )

                eval_reward = float(eval_metrics["mean_reward"])
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_eval_epoch = epoch + 1
                    best_eval_metrics = dict(eval_metrics)
                    no_improve_eval_steps = 0
                    self.save_model(
                        checkpoint / "best.pt",
                        metrics={
                            "best_eval_epoch": best_eval_epoch,
                            "best_eval_reward": best_eval_reward,
                        },
                    )
                    # Keep a conventional named checkpoint for backward compatibility.
                    self.save_model(
                        checkpoint / f"{model_name}.pt",
                        metrics={
                            "best_eval_epoch": best_eval_epoch,
                            "best_eval_reward": best_eval_reward,
                        },
                    )
                else:
                    no_improve_eval_steps += 1

                epoch_pbar.set_postfix(
                    train_iqm=f"{float(train_stats['iqm']):.4f}",
                    eval_reward=f"{eval_reward:.4f}",
                    eval_acc=f"{float(eval_metrics['mean_accuracy']):.4f}",
                )

                if (
                    early_stopping_patience_evals > 0
                    and no_improve_eval_steps >= early_stopping_patience_evals
                ):
                    print(
                        "Early stopping: "
                        f"no eval-reward improvement for {no_improve_eval_steps} checkpoints."
                    )
                    break
            else:
                epoch_pbar.set_postfix(
                    train_iqm=f"{float(train_stats['iqm']):.4f}",
                    train_loss=f"{self._safe_mean(epoch_batch_losses):.4f}",
                )

        if best_eval_epoch < 0:
            # If no eval checkpoint was reached, still persist a canonical best model.
            self.save_model(
                checkpoint / "best.pt",
                metrics={
                    "best_eval_epoch": None,
                    "best_eval_reward": None,
                    "note": "No eval checkpoint reached; saved final training weights.",
                },
            )
            self.save_model(
                checkpoint / f"{model_name}.pt",
                metrics={
                    "best_eval_epoch": None,
                    "best_eval_reward": None,
                    "note": "No eval checkpoint reached; saved final training weights.",
                },
            )

        # Save full histories.
        with (train_metrics_dir / "history.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "epochs": history["train_epochs"],
                    "iqm_reward": history["train_iqm_reward"],
                    "iqr_std_reward": history["train_iqr_std_reward"],
                    "mean_reward": history["train_mean_reward"],
                    "std_reward": history["train_std_reward"],
                    "mean_loss": history["train_mean_loss"],
                },
                fh,
                indent=2,
            )
        with (eval_metrics_dir / "history.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "epochs": history["eval_epochs"],
                    "iqm_reward": history["eval_iqm_reward"],
                    "iqr_std_reward": history["eval_iqr_std_reward"],
                    "iqm_abs_reward_gap": history["eval_iqm_abs_reward_gap"],
                    "iqr_std_abs_reward_gap": history["eval_iqr_std_abs_reward_gap"],
                    "regimes": eval_regime_history,
                    "best_eval_epoch": best_eval_epoch,
                    "best_eval_reward": best_eval_reward,
                },
                fh,
                indent=2,
            )
        with (eval2_metrics_dir / "history.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "epochs": history["eval2_epochs"],
                    "iqm_reward": history["eval2_iqm_reward"],
                    "iqr_std_reward": history["eval2_iqr_std_reward"],
                    "iqm_abs_reward_gap": history["eval2_iqm_abs_reward_gap"],
                    "iqr_std_abs_reward_gap": history["eval2_iqr_std_abs_reward_gap"],
                    "regimes": eval2_regime_history,
                },
                fh,
                indent=2,
            )

        with (checkpoint / "history_losses.json").open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        with (checkpoint / "best_model_performance.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "best_model_name": "best.pt",
                    "selection_metric": "mean_reward",
                    "best_eval_epoch": best_eval_epoch if best_eval_epoch >= 0 else None,
                    "best_eval_reward": best_eval_reward if best_eval_epoch >= 0 else None,
                    "best_metrics": best_eval_metrics if best_eval_epoch >= 0 else {},
                },
                fh,
                indent=2,
            )

        return history

    def save_model(self, out_path: str | Path, metrics: Optional[Dict[str, float]] = None):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": {
                "node_input_dim": self.model.encoder.input_proj.in_features,
                "hidden_dim": self.model.encoder.input_proj.out_features,
                "gnn_layers": len(self.model.encoder.layers),
                "conv_type": self.model.encoder.conv_type,
                "pooling_type": self.model.pooling_type,
                "dataset_type": self.model.dataset_type,
                "edge_emb_dim": self.model.edge_emb_dim,
                "num_edge_labels": self.model.num_edge_labels,
                "num_node_labels": self.model.num_node_labels,
                "use_global_context": self.model.use_global_context,
                "mlp_depth": (len(self.model.policy_head) - 1) // 2,
                "use_goal_separate_input": self.model.use_goal_separate_input,
            },
            "metrics": metrics or {},
        }
        torch.save(payload, out_path)

    @staticmethod
    def load_model(path: str | Path, device: Optional[str | torch.device] = None) -> FrontierPolicyNetwork:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = payload["config"]
        cfg.setdefault("num_node_labels", 4096)
        model = FrontierPolicyNetwork(**cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def to_onnx(
        self,
        out_path: str | Path,
        node_input_dim: int,
        onnx_frontier_size: int = 32,
    ) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # ONNX tracing may specialize pooling/scatter dimensions to the traced
        # candidate range; keep this comfortably above common frontier sizes.
        n_candidates = int(onnx_frontier_size)
        if n_candidates <= 0:
            raise ValueError("onnx_frontier_size must be > 0.")
        n_nodes, n_edges = max(8, n_candidates), 12
        dataset_type = str(self.model.dataset_type).upper()
        raw_node_input_dim = int(node_input_dim)
        node_tensor_dtype = (
            torch.float32
            if dataset_type == "BITMASK"
            else torch.int64
        )
        if dataset_type == "HASHED":
            node_features_dummy = torch.zeros((n_nodes,), dtype=torch.int64)
        else:
            node_features_dummy = torch.zeros((n_nodes, raw_node_input_dim), dtype=node_tensor_dtype)
        model_was_training = bool(self.model.training)
        model_device = self._model_device()
        try:
            if self.kind_of_data == "separated" and self.model.use_goal_separate_input:
                wrapper = OnnxFrontierPolicySeparatedWrapper(self.model).eval().cpu()
                n_goal_nodes, n_goal_edges = 6, 8
                if dataset_type == "HASHED":
                    goal_node_features_dummy = torch.zeros((n_goal_nodes,), dtype=torch.int64)
                else:
                    goal_node_features_dummy = torch.zeros(
                        (n_goal_nodes, raw_node_input_dim),
                        dtype=node_tensor_dtype,
                    )
                dummy_inputs = (
                    node_features_dummy,
                    torch.zeros((2, n_edges), dtype=torch.int64),
                    torch.zeros((n_edges,), dtype=torch.int64),
                    torch.arange(n_nodes, dtype=torch.int64) % n_candidates,
                    goal_node_features_dummy,
                    torch.zeros((2, n_goal_edges), dtype=torch.int64),
                    torch.zeros((n_goal_edges,), dtype=torch.int64),
                    torch.zeros((n_goal_nodes,), dtype=torch.int64),
                    torch.ones((n_candidates,), dtype=torch.uint8),
                )
                input_names = [
                    "node_features",
                    "edge_index",
                    "edge_attr",
                    "membership",
                    "goal_node_features",
                    "goal_edge_index",
                    "goal_edge_attr",
                    "goal_batch",
                    "mask",
                ]
                dynamic_axes = {
                    "node_features": {0: "N"},
                    "edge_index": {1: "E"},
                    "edge_attr": {0: "E"},
                    "membership": {0: "N"},
                    "goal_node_features": {0: "GN"},
                    "goal_edge_index": {1: "GE"},
                    "goal_edge_attr": {0: "GE"},
                    "goal_batch": {0: "GN"},
                    "mask": {0: "F"},
                    "logits": {0: "F"},
                }
            else:
                wrapper = OnnxFrontierPolicyWrapper(self.model).eval().cpu()
                dummy_inputs = (
                    node_features_dummy,
                    torch.zeros((2, n_edges), dtype=torch.int64),
                    torch.zeros((n_edges,), dtype=torch.int64),
                    torch.arange(n_nodes, dtype=torch.int64) % n_candidates,
                    torch.ones((n_candidates,), dtype=torch.uint8),
                )
                input_names = [
                    "node_features",
                    "edge_index",
                    "edge_attr",
                    "membership",
                    "mask",
                ]
                dynamic_axes = {
                    "node_features": {0: "N"},
                    "edge_index": {1: "E"},
                    "edge_attr": {0: "E"},
                    "membership": {0: "N"},
                    "mask": {0: "F"},
                    "logits": {0: "F"},
                }

            torch.onnx.export(
                wrapper,
                dummy_inputs,
                out_path.as_posix(),
                opset_version=18,
                dynamo=False,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=False,
            )
        finally:
            self.model.to(model_device)
            self.model.train(model_was_training)
            self.device = model_device

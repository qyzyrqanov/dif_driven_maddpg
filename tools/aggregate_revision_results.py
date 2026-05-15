"""Aggregate revision experiment outputs into tables and figures.

The training/evaluation batch writes artifacts outside the repository under
``~/Desktop/dif_driven_revision_corrected_artifacts``. This module keeps the notebook
thin: it audits run completeness, loads evaluation outputs, parses available
training curves, and writes the CSV/PNG artifacts needed for Phase 3.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ARTIFACT_ROOT = Path.home() / "Desktop" / "dif_driven_revision_corrected_artifacts"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^n(?P<n>[456])_(?P<mode>full|ablation|nocoll)_seed(?P<seed>\d+)$")
EPISODE_RE = re.compile(
    r"Episode\s+(?P<episode>\d+),\s+Mean Score:\s+(?P<score>-?\d+(?:\.\d+)?),\s+Tagged count:\s+(?P<tagged>\d+)"
)
EVAL_POLICY_RE = re.compile(
    r"^(?P<kind>policy|generalization)_n(?P<n>[456])_(?P<mode>full|ablation|nocoll)_"
    r"trainseed(?P<train_seed>\d+)_env(?P<env_size>\d+)_evalseed(?P<eval_seed>\d+)\.csv$"
)
EVAL_HEURISTIC_RE = re.compile(
    r"^heuristic_n(?P<n>[456])_env(?P<env_size>\d+)_evalseed(?P<eval_seed>\d+)\.csv$"
)
SCALES = {
    "full": np.array([1, 1, 0, 10, 10, 10, 1, 1, 1], dtype=float),
    "ablation": np.array([0, 0, 0, 10, 10, 10, 1, 1, 1], dtype=float),
    "nocoll": np.array([1, 1, 0, 10, 0, 0, 1, 1, 1], dtype=float),
}
MODES = ("full", "ablation", "nocoll")
NS = (4, 5, 6)
SEEDS = (9832, 0, 13)


@dataclass(frozen=True)
class Paths:
    artifact_root: Path
    runs_dir: Path
    eval_dir: Path
    out_dir: Path


def get_paths(artifact_root: Path = ARTIFACT_ROOT, out_dir: Path | None = None) -> Paths:
    artifact_root = artifact_root.expanduser()
    runs_dir = artifact_root / "runs"
    return Paths(
        artifact_root=artifact_root,
        runs_dir=runs_dir,
        eval_dir=runs_dir / "eval",
        out_dir=(out_dir or artifact_root / "res").expanduser(),
    )


def _fmt_mean_sd(values: pd.Series, *, scale: float = 1.0, digits: int = 2) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna() * scale
    if clean.empty:
        return ""
    if len(clean) == 1:
        return f"{clean.iloc[0]:.{digits}f}"
    return f"{clean.mean():.{digits}f} +/- {clean.std(ddof=1):.{digits}f}"


def _sem_ci95(values: pd.Series, *, scale: float = 1.0) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float) * scale
    if len(clean) == 0:
        return math.nan, math.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    mean = clean.mean()
    half = stats.t.ppf(0.975, len(clean) - 1) * clean.std(ddof=1) / math.sqrt(len(clean))
    return float(mean - half), float(mean + half)


def audit_training_runs(paths: Paths) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    legacy_valid = {
        (4, "full", 9832),
        (5, "full", 9832),
        (6, "full", 9832),
        (4, "ablation", 9832),
        (5, "ablation", 9832),
    }
    legacy_csv = {
        (4, "full", 9832): REPO_ROOT / "res" / "result4.csv",
        (5, "full", 9832): REPO_ROOT / "res" / "result5.csv",
        (6, "full", 9832): REPO_ROOT / "res" / "result6.csv",
        (4, "ablation", 9832): REPO_ROOT / "res" / "result4_ablation.csv",
        (5, "ablation", 9832): REPO_ROOT / "res" / "result5_ablation.csv",
    }

    for n in NS:
        for mode in MODES:
            for seed in SEEDS:
                run_dir = paths.runs_dir / f"n{n}_{mode}_seed{seed}"
                meta_path = run_dir / "meta.json"
                row: dict[str, object] = {
                    "n": n,
                    "mode": mode,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "meta_path": str(meta_path) if meta_path.exists() else "",
                    "episodes_completed": np.nan,
                    "finished": False,
                    "launcher_status": "",
                    "valid_for_revision": False,
                    "source_note": "missing",
                }
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    episodes = int(meta.get("episodes_completed") or 0)
                    finished = bool(meta.get("finished")) or meta.get("launcher_status") == "finished"
                    row.update(
                        {
                            "episodes_completed": episodes,
                            "finished": finished,
                            "launcher_status": meta.get("launcher_status", ""),
                            "total_train_hours": meta.get("total_train_hours", np.nan),
                            "peak_gpu_gb": meta.get("peak_gpu_gb", np.nan),
                            "valid_for_revision": bool(finished and episodes >= 1000),
                            "source_note": "generated_meta",
                        }
                    )
                if (n, mode, seed) in legacy_valid and legacy_csv[(n, mode, seed)].exists():
                    row.update(
                        {
                            "episodes_completed": 1000,
                            "finished": True,
                            "launcher_status": "legacy",
                            "valid_for_revision": True,
                            "source_note": "legacy_res_csv",
                            "legacy_csv": str(legacy_csv[(n, mode, seed)]),
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["mode", "n", "seed"]).reset_index(drop=True)


def load_eval_episodes(paths: Paths, audit: pd.DataFrame) -> pd.DataFrame:
    valid_lookup = {
        (int(r.n), str(r.mode), int(r.seed)): bool(r.valid_for_revision)
        for r in audit.itertuples(index=False)
    }
    frames: list[pd.DataFrame] = []
    for path in sorted(paths.eval_dir.glob("*.csv")):
        if path.name.endswith("_steps.csv"):
            continue
        policy_match = EVAL_POLICY_RE.match(path.name)
        heuristic_match = EVAL_HEURISTIC_RE.match(path.name)
        if not policy_match and not heuristic_match:
            continue

        df = pd.read_csv(path)
        if policy_match:
            meta = policy_match.groupdict()
            n = int(meta["n"])
            mode = meta["mode"]
            train_seed = int(meta["train_seed"])
            eval_kind = meta["kind"]
            method = mode
            valid_training = valid_lookup.get((n, mode, train_seed), False)
        else:
            meta = heuristic_match.groupdict()
            n = int(meta["n"])
            train_seed = np.nan
            eval_kind = "heuristic"
            method = "heuristic"
            valid_training = True

        df["n"] = n
        df["method"] = method
        df["mode"] = method if method != "heuristic" else ""
        df["train_seed"] = train_seed
        df["eval_seed"] = int(meta["eval_seed"])
        df["env_size"] = float(meta["env_size"])
        df["eval_kind"] = eval_kind
        df["source_csv"] = str(path)
        df["valid_training"] = valid_training
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_eval_runs(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    group_cols = ["eval_kind", "n", "method", "train_seed", "eval_seed", "env_size", "source_csv", "valid_training"]
    rows = []
    for key, group in eval_df.groupby(group_cols, dropna=False):
        record = dict(zip(group_cols, key, strict=False))
        record.update(
            {
                "episodes": len(group),
                "success_rate": group["success"].mean(),
                "mean_completion_time": group.loc[group["success"].astype(bool), "completion_time"].mean(),
                "mean_episode_return": group["episode_return"].mean(),
                "collision_active_step_rate": group["collision_step_rate"].mean(),
                "path_length_total": group["path_length_total"].mean(),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["eval_kind", "n", "method", "train_seed", "eval_seed"]).reset_index(drop=True)


def make_summary_table(seed_summary: pd.DataFrame, *, eval_kind: str, methods: tuple[str, ...], valid_only: bool = True) -> pd.DataFrame:
    df = seed_summary[seed_summary["eval_kind"].eq(eval_kind) & seed_summary["method"].isin(methods)].copy()
    if valid_only:
        df = df[df["valid_training"]]
    rows = []
    for (n, method), group in df.groupby(["n", "method"], dropna=False):
        ci_lo, ci_hi = _sem_ci95(group["success_rate"], scale=100.0)
        rows.append(
            {
                "n": int(n),
                "method": method,
                "seeds": int(group["train_seed"].nunique(dropna=True) or group["eval_seed"].nunique()),
                "success_rate_pct_mean_sd": _fmt_mean_sd(group["success_rate"], scale=100.0),
                "success_rate_pct_ci95": "" if math.isnan(ci_lo) else f"{ci_lo:.2f} to {ci_hi:.2f}",
                "completion_time_steps_mean_sd": _fmt_mean_sd(group["mean_completion_time"]),
                "episode_return_mean_sd": _fmt_mean_sd(group["mean_episode_return"]),
                "collision_active_steps_pct_mean_sd": _fmt_mean_sd(group["collision_active_step_rate"], scale=100.0),
                "path_length_mean_sd": _fmt_mean_sd(group["path_length_total"]),
                "valid_training_only": bool(valid_only),
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "method"]).reset_index(drop=True)


def make_generalization_table(seed_summary: pd.DataFrame, *, valid_only: bool = True) -> pd.DataFrame:
    env20 = seed_summary[
        seed_summary["eval_kind"].eq("policy")
        & seed_summary["method"].eq("full")
        & seed_summary["env_size"].eq(20.0)
    ].copy()
    env25 = seed_summary[
        seed_summary["eval_kind"].eq("generalization")
        & seed_summary["method"].eq("full")
        & seed_summary["env_size"].eq(25.0)
    ].copy()
    df = pd.concat([env20, env25], ignore_index=True)
    if valid_only:
        df = df[df["valid_training"]]

    rows = []
    for (n, env_size), group in df.groupby(["n", "env_size"]):
        rows.append(
            {
                "n": int(n),
                "training_field": "20 x 20",
                "evaluation_field": f"{int(env_size)} x {int(env_size)}",
                "seeds": int(group["train_seed"].nunique()),
                "success_rate_pct_mean_sd": _fmt_mean_sd(group["success_rate"], scale=100.0),
                "completion_time_steps_mean_sd": _fmt_mean_sd(group["mean_completion_time"]),
                "collision_active_steps_pct_mean_sd": _fmt_mean_sd(group["collision_active_step_rate"], scale=100.0),
                "path_length_mean_sd": _fmt_mean_sd(group["path_length_total"]),
                "valid_training_only": bool(valid_only),
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "evaluation_field"]).reset_index(drop=True)


def compute_stats(seed_summary: pd.DataFrame, *, valid_only: bool = True) -> pd.DataFrame:
    df = seed_summary[seed_summary["eval_kind"].eq("policy")].copy()
    if valid_only:
        df = df[df["valid_training"]]
    rows = []
    for n in NS:
        full = df[(df["n"].eq(n)) & (df["method"].eq("full"))]["success_rate"].dropna()
        for method in ("ablation", "nocoll"):
            other = df[(df["n"].eq(n)) & (df["method"].eq(method))]["success_rate"].dropna()
            if len(full) > 0 and len(other) > 0:
                try:
                    stat, p = stats.mannwhitneyu(full, other, alternative="two-sided")
                except ValueError:
                    stat, p = math.nan, math.nan
            else:
                stat, p = math.nan, math.nan
            rows.append(
                {
                    "test": "mannwhitneyu_success_rate",
                    "n": n,
                    "comparison": f"full_vs_{method}",
                    "full_seed_count": len(full),
                    "other_seed_count": len(other),
                    "statistic": stat,
                    "p_value": p,
                    "valid_training_only": bool(valid_only),
                }
            )
    for method in ("full", "ablation", "nocoll"):
        groups = [
            df[(df["n"].eq(n)) & (df["method"].eq(method))]["success_rate"].dropna()
            for n in NS
        ]
        usable = [g for g in groups if len(g) > 0]
        if len(usable) >= 2:
            try:
                stat, p = stats.kruskal(*usable)
            except ValueError:
                stat, p = math.nan, math.nan
        else:
            stat, p = math.nan, math.nan
        rows.append(
            {
                "test": "kruskal_success_rate_across_n",
                "n": np.nan,
                "comparison": method,
                "full_seed_count": np.nan,
                "other_seed_count": np.nan,
                "statistic": stat,
                "p_value": p,
                "valid_training_only": bool(valid_only),
            }
        )
    return pd.DataFrame(rows)


def parse_episode_log(path: Path, *, n: int, mode: str, seed: int) -> pd.DataFrame:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = EPISODE_RE.search(line)
        if not match:
            continue
        tagged = int(match.group("tagged"))
        rows.append(
            {
                "n": n,
                "mode": mode,
                "seed": seed,
                "episode_id": int(match.group("episode")),
                "mean_score": float(match.group("score")),
                "tagged_count": tagged,
                "success": tagged >= n,
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def parse_legacy_step_csv(path: Path, *, n: int, mode: str, seed: int, chunksize: int = 100_000) -> pd.DataFrame:
    scale = SCALES[mode]
    comp_cols = [f"agent{agent}_comp{k}" for agent in range(n) for k in range(1, 10)]
    usecols = ["episode_id", "timestep", "done_count", *comp_cols]
    totals: dict[int, dict[str, float]] = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        comps = chunk[comp_cols].to_numpy(dtype=float).reshape(len(chunk), n, 9)
        chunk_reward = (comps * scale).sum(axis=(1, 2))
        temp = pd.DataFrame(
            {
                "episode_id": chunk["episode_id"].to_numpy(),
                "reward": chunk_reward,
                "done_count": chunk["done_count"].to_numpy(),
                "timestep": chunk["timestep"].to_numpy(),
            }
        )
        grouped = temp.groupby("episode_id").agg(
            episode_return=("reward", "sum"),
            max_done_count=("done_count", "max"),
            max_timestep=("timestep", "max"),
        )
        for episode_id, record in grouped.iterrows():
            slot = totals.setdefault(
                int(episode_id),
                {"episode_return": 0.0, "max_done_count": 0.0, "max_timestep": 0.0},
            )
            slot["episode_return"] += float(record["episode_return"])
            slot["max_done_count"] = max(slot["max_done_count"], float(record["max_done_count"]))
            slot["max_timestep"] = max(slot["max_timestep"], float(record["max_timestep"]))

    rows = []
    for episode_id, record in sorted(totals.items()):
        rows.append(
            {
                "n": n,
                "mode": mode,
                "seed": seed,
                "episode_id": episode_id,
                "mean_score": record["episode_return"] / n,
                "tagged_count": int(record["max_done_count"]),
                "success": record["max_done_count"] >= n,
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_training_curves(paths: Paths) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in sorted(paths.runs_dir.glob("n*_seed*")):
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        log_path = run_dir / "episode_log.txt"
        if not log_path.exists():
            continue
        frames.append(
            parse_episode_log(
                log_path,
                n=int(match.group("n")),
                mode=match.group("mode"),
                seed=int(match.group("seed")),
            )
        )

    legacy = [
        (REPO_ROOT / "res" / "result4.csv", 4, "full", 9832),
        (REPO_ROOT / "res" / "result5.csv", 5, "full", 9832),
        (REPO_ROOT / "res" / "result6.csv", 6, "full", 9832),
        (REPO_ROOT / "res" / "result4_ablation.csv", 4, "ablation", 9832),
        (REPO_ROOT / "res" / "result5_ablation.csv", 5, "ablation", 9832),
    ]
    for path, n, mode, seed in legacy:
        if path.exists():
            frames.append(parse_legacy_step_csv(path, n=n, mode=mode, seed=seed))

    if not frames:
        return pd.DataFrame()
    curves = pd.concat(frames, ignore_index=True)
    return curves.sort_values(["mode", "n", "seed", "episode_id"]).reset_index(drop=True)


def plot_learning_curves(curves: pd.DataFrame, out_dir: Path, *, valid_audit: pd.DataFrame) -> list[Path]:
    out_paths: list[Path] = []
    if curves.empty:
        return out_paths
    valid = {
        (int(r.n), str(r.mode), int(r.seed))
        for r in valid_audit[valid_audit["valid_for_revision"]].itertuples(index=False)
    }
    curves = curves[
        curves.apply(lambda r: (int(r["n"]), str(r["mode"]), int(r["seed"])) in valid, axis=1)
    ].copy()
    if curves.empty:
        return out_paths
    curves["score_roll"] = curves.groupby(["mode", "n", "seed"])["mean_score"].transform(
        lambda s: s.rolling(50, min_periods=1).mean()
    )
    curves["success_roll"] = curves.groupby(["mode", "n", "seed"])["success"].transform(
        lambda s: s.rolling(50, min_periods=1).mean() * 100.0
    )

    for metric, ylabel, filename in [
        ("score_roll", "Mean episode score, rolling 50", "revision_learning_curve_full_valid.png"),
        ("success_roll", "Success rate (%), rolling 50", "revision_rolling_success_full_valid.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        full = curves[curves["mode"].eq("full")]
        for n, group in full.groupby("n"):
            pivot = group.pivot_table(index="episode_id", columns="seed", values=metric, aggfunc="mean").sort_index()
            mean = pivot.mean(axis=1)
            sd = pivot.std(axis=1)
            ax.plot(mean.index, mean.values, label=f"n={n} (seeds={pivot.shape[1]})")
            if pivot.shape[1] > 1:
                ax.fill_between(mean.index, (mean - sd).values, (mean + sd).values, alpha=0.18)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title("Full method training curves: valid completed runs only")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_bar_tables(seed_summary: pd.DataFrame, out_dir: Path, *, valid_only: bool = True) -> list[Path]:
    out_paths: list[Path] = []
    df = seed_summary.copy()
    if valid_only:
        df = df[df["valid_training"]]
    policy = df[df["eval_kind"].eq("policy") & df["method"].isin(["full", "ablation", "nocoll"])]
    heuristic = df[df["eval_kind"].eq("heuristic")]
    baseline = pd.concat([policy, heuristic], ignore_index=True)
    if not baseline.empty:
        means = baseline.groupby(["n", "method"])["success_rate"].mean().mul(100).unstack("method")
        errors = baseline.groupby(["n", "method"])["success_rate"].std(ddof=1).mul(100).unstack("method")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        means.plot(kind="bar", yerr=errors, ax=ax, capsize=3)
        ax.set_ylabel("Success rate (%)")
        ax.set_xlabel("Number of agents")
        ax.set_title("Baseline comparison: valid completed training only")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        out_path = out_dir / "revision_baseline_success_valid.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)

    gen = df[
        (
            df["eval_kind"].eq("policy")
            & df["method"].eq("full")
            & df["env_size"].eq(20.0)
        )
        | (
            df["eval_kind"].eq("generalization")
            & df["method"].eq("full")
            & df["env_size"].eq(25.0)
        )
    ]
    if not gen.empty:
        means = gen.groupby(["n", "env_size"])["success_rate"].mean().mul(100).unstack("env_size")
        errors = gen.groupby(["n", "env_size"])["success_rate"].std(ddof=1).mul(100).unstack("env_size")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        means.plot(kind="bar", yerr=errors, ax=ax, capsize=3)
        ax.set_ylabel("Success rate (%)")
        ax.set_xlabel("Number of agents")
        ax.set_title("Full-method generalization: valid completed training only")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="Eval field")
        fig.tight_layout()
        out_path = out_dir / "revision_generalization_success_valid.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def run_all(artifact_root: Path = ARTIFACT_ROOT, out_dir: Path | None = None) -> dict[str, Path | int]:
    paths = get_paths(artifact_root, out_dir)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_training_runs(paths)
    eval_episodes = load_eval_episodes(paths, audit)
    seed_summary = summarize_eval_runs(eval_episodes)
    curves = load_training_curves(paths)

    outputs: dict[str, Path | int] = {}
    csvs = {
        "revision_training_audit.csv": audit,
        "revision_eval_episodes.csv": eval_episodes,
        "revision_eval_seed_summary.csv": seed_summary,
        "revision_multiseed_summary_valid.csv": make_summary_table(seed_summary, eval_kind="policy", methods=("full",), valid_only=True),
        "revision_multiseed_summary_all_evals.csv": make_summary_table(seed_summary, eval_kind="policy", methods=("full",), valid_only=False),
        "revision_baseline_comparison_valid.csv": make_summary_table(seed_summary, eval_kind="policy", methods=("full", "ablation", "nocoll"), valid_only=True),
        "revision_baseline_comparison_all_evals.csv": make_summary_table(seed_summary, eval_kind="policy", methods=("full", "ablation", "nocoll"), valid_only=False),
        "revision_generalization_valid.csv": make_generalization_table(seed_summary, valid_only=True),
        "revision_generalization_all_evals.csv": make_generalization_table(seed_summary, valid_only=False),
        "revision_stats_valid.csv": compute_stats(seed_summary, valid_only=True),
        "revision_stats_all_evals.csv": compute_stats(seed_summary, valid_only=False),
        "revision_training_curves.csv": curves,
    }
    for name, df in csvs.items():
        path = paths.out_dir / name
        df.to_csv(path, index=False)
        outputs[name] = path

    for path in plot_learning_curves(curves, paths.out_dir, valid_audit=audit):
        outputs[path.name] = path
    for path in plot_bar_tables(seed_summary, paths.out_dir, valid_only=True):
        outputs[path.name] = path

    outputs["valid_training_runs"] = int(audit["valid_for_revision"].sum())
    outputs["eval_episode_rows"] = int(len(eval_episodes))
    outputs["training_curve_rows"] = int(len(curves))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate revision experiment results.")
    parser.add_argument("--artifact_root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()
    outputs = run_all(args.artifact_root, args.out_dir)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

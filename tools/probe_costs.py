"""Collect compute and reproducibility facts for revision reporting."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ARTIFACT_ROOT = Path.home() / "Desktop" / "dif_driven_revision_artifacts"

from config import actor_lr, batch_size, critic_lr, device, gamma, tau  # noqa: E402
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import (  # noqa: E402
    DiffDriveParallelEnvDone,
)
from models.simpleactor import SimpleActor  # noqa: E402
from models.simplecritic import SharedCritic  # noqa: E402


try:
    import psutil
except ImportError:  # pragma: no cover - depends on local environment
    psutil = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write hardware/model/training compute summary tables."
    )
    parser.add_argument("--runs_dir", type=Path, default=DEFAULT_ARTIFACT_ROOT / "runs")
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "res" / "compute_repro_table.csv",
    )
    parser.add_argument(
        "--raw_meta_csv",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "res" / "compute_repro_meta_raw.csv",
    )
    parser.add_argument("--benchmark_n", type=int, default=6)
    parser.add_argument("--benchmark_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def add_row(
    rows: list[dict[str, Any]],
    *,
    category: str,
    metric: str,
    value: Any,
    unit: str = "",
    n: int | str = "",
    mode: str = "",
    seed: int | str = "",
    source: str = "probe",
) -> None:
    rows.append(
        {
            "category": category,
            "metric": metric,
            "n": n,
            "mode": mode,
            "seed": seed,
            "value": value,
            "unit": unit,
            "source": source,
        }
    )


def hardware_rows(rows: list[dict[str, Any]]) -> None:
    add_row(rows, category="hardware", metric="hostname", value=socket.gethostname())
    add_row(rows, category="hardware", metric="platform", value=platform.platform())
    add_row(rows, category="hardware", metric="processor", value=platform.processor())
    add_row(rows, category="software", metric="python_version", value=platform.python_version())
    add_row(rows, category="software", metric="torch_version", value=torch.__version__)
    add_row(rows, category="software", metric="cuda_available", value=torch.cuda.is_available())
    add_row(rows, category="software", metric="cuda_version", value=torch.version.cuda)

    ram_total_gb = None
    if psutil is not None:
        ram_total_gb = psutil.virtual_memory().total / 1e9
    elif hasattr(os, "sysconf"):
        try:
            ram_total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except (OSError, ValueError):
            ram_total_gb = None
    add_row(rows, category="hardware", metric="ram_total_gb", value=ram_total_gb, unit="GB")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        add_row(rows, category="hardware", metric="gpu_name", value=torch.cuda.get_device_name(0))
        add_row(
            rows,
            category="hardware",
            metric="gpu_total_memory_gb",
            value=props.total_memory / 1e9,
            unit="GB",
        )
        add_row(rows, category="hardware", metric="gpu_multiprocessors", value=props.multi_processor_count)
    else:
        add_row(rows, category="hardware", metric="gpu_name", value="none")


def model_and_benchmark_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark_n: int,
    benchmark_steps: int,
    warmup_steps: int,
) -> None:
    env = DiffDriveParallelEnvDone(
        num_agents=benchmark_n,
        num_obstacles=0,
        v_ang_max=torch.pi / 2,
    )
    _, obs = env.reset_tensor()
    actor = SimpleActor(env.obs_dim, env.action_dim, device=device)
    critic = SharedCritic(input_dim=env.obs_dim + env.action_dim, output_dim=1, device=device)

    add_row(rows, category="model", metric="obs_dim", n=benchmark_n, value=env.obs_dim)
    add_row(rows, category="model", metric="action_dim", n=benchmark_n, value=env.action_dim)
    add_row(rows, category="model", metric="critic_input_dim", n=benchmark_n, value=env.obs_dim + env.action_dim)
    add_row(rows, category="model", metric="actor_parameters", n=benchmark_n, value=count_parameters(actor))
    add_row(rows, category="model", metric="critic_parameters", n=benchmark_n, value=count_parameters(critic))
    add_row(rows, category="training_config", metric="gamma", value=gamma)
    add_row(rows, category="training_config", metric="tau", value=tau)
    add_row(rows, category="training_config", metric="actor_lr", value=actor_lr)
    add_row(rows, category="training_config", metric="critic_lr", value=critic_lr)
    add_row(rows, category="training_config", metric="batch_size_config_default", value=batch_size)
    add_row(rows, category="training_config", metric="revision_batch_size", value=128)
    add_row(rows, category="training_config", metric="revision_replay_buffer_size", value=50000)

    actor.eval()
    with torch.no_grad():
        for _ in range(max(warmup_steps, 0)):
            actor.choose_action(obs, use_noise=False, eval_mode=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(max(benchmark_steps, 1)):
            actor.choose_action(obs, use_noise=False, eval_mode=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    steps = max(benchmark_steps, 1)
    add_row(rows, category="inference", metric="benchmark_n", value=benchmark_n)
    add_row(rows, category="inference", metric="benchmark_steps", value=steps)
    add_row(rows, category="inference", metric="actor_ms_per_env_step", value=elapsed * 1000.0 / steps, unit="ms")
    add_row(
        rows,
        category="inference",
        metric="actor_ms_per_agent_step",
        value=elapsed * 1000.0 / (steps * benchmark_n),
        unit="ms",
    )


def load_meta_files(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []

    metas = []
    for path in sorted(runs_dir.rglob("meta.json")):
        try:
            with path.open("r") as f:
                meta = json.load(f)
        except json.JSONDecodeError as exc:
            meta = {"meta_read_error": str(exc)}
        meta["_path"] = str(path)
        metas.append(meta)
    return metas


def write_raw_meta_csv(path: Path, metas: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "n",
        "num_agents",
        "mode",
        "seed",
        "finished",
        "episodes_completed",
        "total_steps",
        "total_train_seconds",
        "total_train_hours",
        "command_wall_seconds",
        "peak_gpu_bytes",
        "peak_gpu_gb",
        "gpu_name",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for meta in metas:
            writer.writerow(
                {
                    "path": meta.get("_path"),
                    "n": meta.get("n"),
                    "num_agents": meta.get("num_agents"),
                    "mode": meta.get("mode"),
                    "seed": meta.get("seed"),
                    "finished": meta.get("finished"),
                    "episodes_completed": meta.get("episodes_completed"),
                    "total_steps": meta.get("total_steps"),
                    "total_train_seconds": meta.get("total_train_seconds"),
                    "total_train_hours": meta.get("total_train_hours"),
                    "command_wall_seconds": meta.get("command_wall_seconds"),
                    "peak_gpu_bytes": meta.get("peak_gpu_bytes"),
                    "peak_gpu_gb": meta.get("peak_gpu_gb"),
                    "gpu_name": meta.get("gpu_name"),
                }
            )


def training_meta_rows(rows: list[dict[str, Any]], metas: list[dict[str, Any]]) -> None:
    add_row(rows, category="training_meta", metric="meta_files_found", value=len(metas), source="runs")
    if not metas:
        return

    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for meta in metas:
        n = meta.get("n", meta.get("num_agents", ""))
        mode = meta.get("mode", "")
        grouped.setdefault((n, mode), []).append(meta)

        add_row(
            rows,
            category="training_meta_raw",
            metric="total_train_hours",
            n=n,
            mode=mode,
            seed=meta.get("seed", ""),
            value=meta.get("total_train_hours"),
            unit="h",
            source=meta.get("_path", "runs"),
        )
        add_row(
            rows,
            category="training_meta_raw",
            metric="peak_gpu_gb",
            n=n,
            mode=mode,
            seed=meta.get("seed", ""),
            value=meta.get("peak_gpu_gb"),
            unit="GB",
            source=meta.get("_path", "runs"),
        )

    for (n, mode), group in sorted(grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        hours = [
            float(meta["total_train_hours"])
            for meta in group
            if meta.get("total_train_hours") is not None
        ]
        peak_gb = [
            float(meta["peak_gpu_gb"])
            for meta in group
            if meta.get("peak_gpu_gb") is not None
        ]
        episodes = [
            float(meta["episodes_completed"])
            for meta in group
            if meta.get("episodes_completed") is not None
        ]
        add_summary_stats(rows, "training_meta_summary", "total_train_hours", hours, "h", n, mode)
        add_summary_stats(rows, "training_meta_summary", "peak_gpu_gb", peak_gb, "GB", n, mode)
        add_summary_stats(rows, "training_meta_summary", "episodes_completed", episodes, "episodes", n, mode)


def add_summary_stats(
    rows: list[dict[str, Any]],
    category: str,
    metric: str,
    values: list[float],
    unit: str,
    n: Any,
    mode: Any,
) -> None:
    if not values:
        return
    add_row(rows, category=category, metric=f"{metric}_count", n=n, mode=mode, value=len(values), source="runs")
    add_row(rows, category=category, metric=f"{metric}_mean", n=n, mode=mode, value=statistics.mean(values), unit=unit, source="runs")
    if len(values) > 1:
        add_row(rows, category=category, metric=f"{metric}_sd", n=n, mode=mode, value=statistics.stdev(values), unit=unit, source="runs")
    else:
        add_row(rows, category=category, metric=f"{metric}_sd", n=n, mode=mode, value=0.0, unit=unit, source="runs")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "metric", "n", "mode", "seed", "value", "unit", "source"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def print_markdown(rows: list[dict[str, Any]], out_csv: Path, raw_meta_csv: Path) -> None:
    print(f"Wrote summary CSV: {out_csv}")
    print(f"Wrote raw meta CSV: {raw_meta_csv}")
    print()
    print("| category | metric | n | mode | value | unit |")
    print("|---|---|---:|---|---:|---|")
    for row in rows:
        if row["category"] in {"training_meta_raw"}:
            continue
        print(
            "| {category} | {metric} | {n} | {mode} | {value} | {unit} |".format(
                category=row["category"],
                metric=row["metric"],
                n=row["n"],
                mode=row["mode"],
                value=fmt(row["value"]),
                unit=row["unit"],
            )
        )


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []

    hardware_rows(rows)
    model_and_benchmark_rows(
        rows,
        benchmark_n=args.benchmark_n,
        benchmark_steps=args.benchmark_steps,
        warmup_steps=args.warmup_steps,
    )
    metas = load_meta_files(args.runs_dir)
    training_meta_rows(rows, metas)

    write_summary_csv(args.out_csv, rows)
    write_raw_meta_csv(args.raw_meta_csv, metas)
    print_markdown(rows, args.out_csv, args.raw_meta_csv)


if __name__ == "__main__":
    main()

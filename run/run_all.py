"""Resumable orchestration for revision training, evaluation, and cost probes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_ARTIFACT_ROOT = Path.home() / "Desktop" / "dif_driven_revision_artifacts"

TRAIN_NS = [4, 5, 6]
TRAIN_SEEDS = [9832, 0, 13]
NEW_TRAIN_RUNS = [
    *[(n, "full", seed) for n in TRAIN_NS for seed in [0, 13]],
    *[(n, "ablation", seed) for n in TRAIN_NS for seed in [0, 13]],
    (6, "ablation", 9832),
    *[(n, "nocoll", seed) for n in TRAIN_NS for seed in TRAIN_SEEDS],
]
POLICY_EVAL_MODES = ["full", "ablation", "nocoll"]
HEURISTIC_EVAL_SEEDS = [42, 100, 200]
POLICY_EVAL_SEED = 42


@dataclass(frozen=True)
class CommandTask:
    name: str
    cmd: list[str]
    done_path: Path | None = None
    required_path: Path | None = None
    result_copy: tuple[Path, Path] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all revision jobs sequentially with resume checks."
    )
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Base directory for generated runs, evals, logs, and derived CSVs.",
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=None,
        help="Override generated run directory. Defaults to ARTIFACT_ROOT/runs.",
    )
    parser.add_argument(
        "--res_dir",
        type=Path,
        default=None,
        help="Override derived result directory. Defaults to ARTIFACT_ROOT/res.",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--heuristic_kp", type=float, default=2.0)
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Override orchestrator log path. Defaults to ARTIFACT_ROOT/run_all.log.",
    )
    parser.add_argument(
        "--copy_large_csvs_to_res",
        action="store_true",
        help=(
            "Also copy per-run/eval CSVs into res_dir. Disabled by default to avoid "
            "large duplicated files being indexed by IDEs."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_probe", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run tasks even if their done_path already exists.",
    )
    parser.add_argument(
        "--strict_missing_eval_actor",
        action="store_true",
        help="Fail if an eval actor checkpoint is missing instead of logging a skip.",
    )
    return parser.parse_args()


def env_for_subprocess() -> dict[str, str]:
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not prior else f"{REPO_ROOT}{os.pathsep}{prior}"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    return env


def log(message: str, log_file: Path) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as f:
        f.write(line + "\n")


def run_dir(runs_dir: Path, n: int, mode: str, seed: int) -> Path:
    return runs_dir / f"n{n}_{mode}_seed{seed}"


def result_filename(n: int, mode: str) -> str:
    suffix = {"full": "", "ablation": "_ablation", "nocoll": "_nocoll"}[mode]
    return f"result{n}{suffix}.csv"


def result_copy_path(res_dir: Path, n: int, mode: str, seed: int) -> Path:
    return res_dir / "revision_runs" / f"n{n}_{mode}_seed{seed}_{result_filename(n, mode)}"


def train_tasks(args: argparse.Namespace) -> list[CommandTask]:
    tasks = []
    for n, mode, seed in NEW_TRAIN_RUNS:
        out_dir = run_dir(args.runs_dir, n, mode, seed)
        result_csv = out_dir / result_filename(n, mode)
        tasks.append(
            CommandTask(
                name=f"train n={n} mode={mode} seed={seed}",
                cmd=[
                    PYTHON,
                    str(REPO_ROOT / "run" / "train_seeded.py"),
                    "--n",
                    str(n),
                    "--mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--episodes",
                    str(args.episodes),
                    "--out_dir",
                    str(out_dir),
                ],
                done_path=result_csv,
                result_copy=(
                    (result_csv, result_copy_path(args.res_dir, n, mode, seed))
                    if args.copy_large_csvs_to_res
                    else None
                ),
            )
        )
    return tasks


def actor_path(runs_dir: Path, n: int, mode: str, seed: int) -> Path:
    return run_dir(runs_dir, n, mode, seed) / "shared_actor.pth"


def policy_eval_task(
    *,
    args: argparse.Namespace,
    n: int,
    mode: str,
    train_seed: int,
    env_size: float,
    tag: str,
) -> CommandTask:
    actor_ckpt = actor_path(args.runs_dir, n, mode, train_seed)
    out_dir = args.runs_dir / "eval"
    env_tag = f"env{int(env_size)}" if float(env_size).is_integer() else f"env{env_size}"
    out_csv = out_dir / f"{tag}_n{n}_{mode}_trainseed{train_seed}_{env_tag}_evalseed{POLICY_EVAL_SEED}.csv"
    summary_json = out_csv.with_suffix(".json")
    return CommandTask(
        name=f"eval policy {tag} n={n} mode={mode} train_seed={train_seed} env_size={env_size}",
        cmd=[
            PYTHON,
            str(REPO_ROOT / "run" / "eval_policy.py"),
            "--actor_ckpt",
            str(actor_ckpt),
            "--n",
            str(n),
            "--mode",
            mode,
            "--env_size",
            str(env_size),
            "--episodes",
            str(args.eval_episodes),
            "--seed",
            str(POLICY_EVAL_SEED),
            "--max_steps",
            str(args.max_steps),
            "--out_csv",
            str(out_csv),
            "--summary_json",
            str(summary_json),
        ],
        done_path=summary_json,
        required_path=actor_ckpt,
        result_copy=(
            (out_csv, args.res_dir / "revision_evals" / out_csv.name)
            if args.copy_large_csvs_to_res
            else None
        ),
    )


def heuristic_eval_tasks(args: argparse.Namespace) -> list[CommandTask]:
    tasks = []
    out_dir = args.runs_dir / "eval"
    for n in TRAIN_NS:
        for eval_seed in HEURISTIC_EVAL_SEEDS:
            out_csv = out_dir / f"heuristic_n{n}_env20_evalseed{eval_seed}.csv"
            summary_json = out_csv.with_suffix(".json")
            tasks.append(
                CommandTask(
                    name=f"eval heuristic n={n} seed={eval_seed}",
                    cmd=[
                        PYTHON,
                        str(REPO_ROOT / "run" / "eval_hungarian_p.py"),
                        "--n",
                        str(n),
                        "--env_size",
                        "20",
                        "--episodes",
                        str(args.eval_episodes),
                        "--seed",
                        str(eval_seed),
                        "--Kp",
                        str(args.heuristic_kp),
                        "--mode",
                        "full",
                        "--max_steps",
                        str(args.max_steps),
                        "--out_csv",
                        str(out_csv),
                        "--summary_json",
                        str(summary_json),
                    ],
                    done_path=summary_json,
                    result_copy=(
                        (out_csv, args.res_dir / "revision_evals" / out_csv.name)
                        if args.copy_large_csvs_to_res
                        else None
                    ),
                )
            )
    return tasks


def eval_tasks(args: argparse.Namespace) -> list[CommandTask]:
    tasks = []
    for mode in POLICY_EVAL_MODES:
        for n in TRAIN_NS:
            for train_seed in TRAIN_SEEDS:
                tasks.append(
                    policy_eval_task(
                        args=args,
                        n=n,
                        mode=mode,
                        train_seed=train_seed,
                        env_size=20,
                        tag="policy",
                    )
                )

    for n in TRAIN_NS:
        for train_seed in TRAIN_SEEDS:
            tasks.append(
                policy_eval_task(
                    args=args,
                    n=n,
                    mode="full",
                    train_seed=train_seed,
                    env_size=25,
                    tag="generalization",
                )
            )

    tasks.extend(heuristic_eval_tasks(args))
    return tasks


def probe_task(args: argparse.Namespace) -> CommandTask:
    out_csv = args.res_dir / "compute_repro_table.csv"
    raw_csv = args.res_dir / "compute_repro_meta_raw.csv"
    return CommandTask(
        name="probe costs",
        cmd=[
            PYTHON,
            str(REPO_ROOT / "tools" / "probe_costs.py"),
            "--runs_dir",
            str(args.runs_dir),
            "--out_csv",
            str(out_csv),
            "--raw_meta_csv",
            str(raw_csv),
        ],
        done_path=out_csv,
    )


def copy_result(copy_pair: tuple[Path, Path] | None, log_file: Path) -> None:
    if copy_pair is None:
        return
    source, dest = copy_pair
    if not source.exists():
        log(f"copy skipped, source missing: {source}", log_file)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    log(f"copied {source} -> {dest}", log_file)


def run_task(task: CommandTask, args: argparse.Namespace) -> str:
    log_file = args.log_file
    if task.done_path is not None and task.done_path.exists() and not args.rerun:
        log(f"SKIP completed: {task.name} ({task.done_path})", log_file)
        copy_result(task.result_copy, log_file)
        return "skipped_done"

    if task.required_path is not None and not task.required_path.exists():
        message = f"SKIP missing requirement: {task.name} requires {task.required_path}"
        if args.strict_missing_eval_actor:
            raise FileNotFoundError(message)
        log(message, log_file)
        return "skipped_missing"

    log(f"START {task.name}", log_file)
    log("CMD " + " ".join(task.cmd), log_file)
    if args.dry_run:
        log(f"DRY-RUN finish: {task.name}", log_file)
        return "dry_run"

    start = time.time()
    with log_file.open("a") as f:
        f.write(f"\n===== {task.name} =====\n")
        f.flush()
        proc = subprocess.run(
            task.cmd,
            cwd=REPO_ROOT,
            env=env_for_subprocess(),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed = time.time() - start
    if proc.returncode != 0:
        log(f"FAIL {task.name} rc={proc.returncode} elapsed_s={elapsed:.1f}", log_file)
        raise subprocess.CalledProcessError(proc.returncode, task.cmd)

    log(f"DONE {task.name} elapsed_s={elapsed:.1f}", log_file)
    copy_result(task.result_copy, log_file)
    return "ran"


def write_summary(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"counts": counts, "finished_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=2)


def main() -> None:
    args = parse_args()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    if args.runs_dir is None:
        args.runs_dir = args.artifact_root / "runs"
    if args.res_dir is None:
        args.res_dir = args.artifact_root / "res"
    if args.log_file is None:
        args.log_file = args.artifact_root / "run_all.log"

    args.runs_dir = args.runs_dir.resolve()
    args.res_dir = args.res_dir.resolve()
    args.log_file = args.log_file.resolve()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.res_dir.mkdir(parents=True, exist_ok=True)

    if args.train_only and args.eval_only:
        raise ValueError("--train_only and --eval_only cannot be used together")

    tasks: list[CommandTask] = []
    if not args.eval_only:
        tasks.extend(train_tasks(args))
    if not args.train_only:
        tasks.extend(eval_tasks(args))
    if not args.skip_probe and not args.train_only:
        tasks.append(probe_task(args))

    log(f"Prepared {len(tasks)} tasks", args.log_file)
    counts = {"ran": 0, "skipped_done": 0, "skipped_missing": 0, "dry_run": 0}
    for index, task in enumerate(tasks, start=1):
        log(f"Task {index}/{len(tasks)}", args.log_file)
        status = run_task(task, args)
        counts[status] = counts.get(status, 0) + 1

    summary_path = args.runs_dir / "run_all_summary.json"
    write_summary(summary_path, counts)
    log(f"All requested tasks handled. Summary: {counts}", args.log_file)
    log(f"Summary JSON: {summary_path}", args.log_file)


if __name__ == "__main__":
    main()

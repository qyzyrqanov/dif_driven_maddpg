"""Resumable orchestration for revision training, evaluation, and cost probes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.offload_artifacts import (  # noqa: E402
    ensure_target_root,
    offload_run_dir,
)

PYTHON = sys.executable
DEFAULT_ARTIFACT_ROOT = Path.home() / "Desktop" / "dif_driven_revision_corrected_artifacts"
DEFAULT_OFFLOAD_ROOT = Path("/media/abz/Z7S/experiments_revision_corrected")

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
LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class CommandTask:
    name: str
    cmd: list[str]
    kind: str
    index: int = 0
    done_path: Path | None = None
    required_paths: tuple[Path, ...] = ()
    result_copy: tuple[Path, Path] | None = None
    lock_path: Path | None = None
    cleanup_dir: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run revision jobs with resume checks and bounded parallelism."
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
        "--v_ang_max",
        choices=["pi9", "pi2"],
        default="pi9",
        help="Angular velocity cap for training/evaluation. Default pi9 is corrected.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help=(
            "Maximum number of independent subprocess tasks to run at once. "
            "Use 1 while actively using the laptop; use 5 when the machine is free."
        ),
    )
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
        "--offload_root",
        type=Path,
        default=DEFAULT_OFFLOAD_ROOT,
        help=(
            "External artifact mirror. After each completed training run, run_all "
            "copies that run directory here, verifies file sizes, then prunes local "
            "heavy files. train_seeded also mirrors each saved episode checkpoint "
            "here. If unavailable, it only logs a warning."
        ),
    )
    parser.add_argument(
        "--disable_offload",
        action="store_true",
        help="Disable automatic post-training offload/cleanup.",
    )
    parser.add_argument(
        "--keep_local_result_csv",
        action="store_true",
        help=(
            "Keep completed training result CSVs in the local artifact root after "
            "offload. Default removes them to save space; run_all can still detect "
            "the offloaded CSV."
        ),
    )
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
    parser.add_argument(
        "--check_completeness",
        action="store_true",
        help=(
            "Write a train/eval completeness report and exit without launching "
            "subprocess tasks."
        ),
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
    with LOG_LOCK:
        print(line, flush=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(line + "\n")


def task_log_path(task: CommandTask, args: argparse.Namespace) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in task.name).strip("_")
    return args.artifact_root / "logs" / f"{safe}.log"


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode(errors="replace")


def pid_matches_task(pid: int, task: CommandTask) -> bool:
    cmdline = pid_cmdline(pid)
    if not cmdline:
        return False
    script = str(REPO_ROOT / "run" / "train_seeded.py")
    if task.kind == "train" and script not in cmdline:
        return False
    for token in task.cmd[1:]:
        if token not in cmdline:
            return False
    return True


def active_task_lock(task: CommandTask) -> bool:
    if task.lock_path is None or not task.lock_path.exists():
        return False
    try:
        prior = json.loads(task.lock_path.read_text())
        raw_pid = prior.get("pid")
        if raw_pid is None:
            raw_pid = prior.get("owner_pid")
        prior_pid = int(raw_pid)
    except Exception:
        return False
    return pid_is_running(prior_pid) and pid_matches_task(prior_pid, task)


def acquire_task_lock(task: CommandTask, args: argparse.Namespace) -> bool:
    if task.lock_path is None or args.dry_run:
        return True

    lock_path = task.lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        try:
            prior = json.loads(lock_path.read_text())
            raw_pid = prior.get("pid")
            if raw_pid is None:
                raw_pid = prior.get("owner_pid")
            prior_pid = int(raw_pid)
        except Exception:
            prior_pid = -1

        if pid_is_running(prior_pid) and pid_matches_task(prior_pid, task):
            log(
                f"SKIP running: {task.name} locked by pid={prior_pid} ({lock_path})",
                args.log_file,
            )
            return False

        log(f"Removing stale lock for {task.name}: {lock_path}", args.log_file)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log(f"SKIP running: {task.name} lock appeared ({lock_path})", args.log_file)
        return False

    with os.fdopen(fd, "w") as f:
        json.dump(
            {
                "pid": None,
                "owner_pid": os.getpid(),
                "task": task.name,
                "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "cmd": task.cmd,
            },
            f,
            indent=2,
        )
    return True


def update_task_lock_pid(task: CommandTask, args: argparse.Namespace, pid: int) -> None:
    if task.lock_path is None or args.dry_run:
        return
    payload = {
        "pid": int(pid),
        "owner_pid": os.getpid(),
        "task": task.name,
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cmd": task.cmd,
    }
    task.lock_path.write_text(json.dumps(payload, indent=2))


def release_task_lock(task: CommandTask, args: argparse.Namespace) -> None:
    if task.lock_path is None or args.dry_run:
        return
    try:
        task.lock_path.unlink()
    except FileNotFoundError:
        pass


def prepare_incomplete_train_dir(task: CommandTask, args: argparse.Namespace) -> str | None:
    if task.kind != "train" or task.cleanup_dir is None:
        return None
    if not task.cleanup_dir.exists():
        return None
    if training_meta_is_finished(task.cleanup_dir, args):
        return None
    if active_task_lock(task):
        return "running"

    has_artifacts = any(task.cleanup_dir.iterdir())
    if not has_artifacts:
        return None

    log(
        f"RESUME incomplete training dir: {task.cleanup_dir}",
        args.log_file,
    )
    return None


def run_dir(runs_dir: Path, n: int, mode: str, seed: int) -> Path:
    return runs_dir / f"n{n}_{mode}_seed{seed}"


def result_filename(n: int, mode: str) -> str:
    suffix = {"full": "", "ablation": "_ablation", "nocoll": "_nocoll"}[mode]
    return f"result{n}{suffix}.csv"


def result_copy_path(res_dir: Path, n: int, mode: str, seed: int) -> Path:
    return res_dir / "revision_runs" / f"n{n}_{mode}_seed{seed}_{result_filename(n, mode)}"


def is_generated_train_run(n: int, mode: str, seed: int) -> bool:
    return (n, mode, seed) in set(NEW_TRAIN_RUNS)


def train_done_path(runs_dir: Path, n: int, mode: str, seed: int) -> Path:
    return run_dir(runs_dir, n, mode, seed) / result_filename(n, mode)


def with_task_indices(tasks: list[CommandTask]) -> list[CommandTask]:
    return [replace(task, index=index) for index, task in enumerate(tasks, start=1)]


def train_tasks(args: argparse.Namespace) -> list[CommandTask]:
    tasks = []
    for n, mode, seed in NEW_TRAIN_RUNS:
        out_dir = run_dir(args.runs_dir, n, mode, seed)
        result_csv = out_dir / result_filename(n, mode)
        tasks.append(
            CommandTask(
                name=f"train n={n} mode={mode} seed={seed}",
                kind="train",
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
                    "--v_ang_max",
                    args.v_ang_max,
                    "--artifact_root",
                    str(args.artifact_root),
                    "--offload_root",
                    str(args.offload_root),
                    *(
                        ["--disable_episode_offload"]
                        if args.disable_offload
                        else []
                    ),
                ],
                done_path=result_csv,
                lock_path=out_dir / ".run_all.lock",
                cleanup_dir=out_dir,
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
    required_paths = [actor_ckpt]
    if is_generated_train_run(n, mode, train_seed):
        required_paths.append(train_done_path(args.runs_dir, n, mode, train_seed))

    out_dir = args.runs_dir / "eval"
    env_tag = f"env{int(env_size)}" if float(env_size).is_integer() else f"env{env_size}"
    out_csv = out_dir / f"{tag}_n{n}_{mode}_trainseed{train_seed}_{env_tag}_evalseed{POLICY_EVAL_SEED}.csv"
    summary_json = out_csv.with_suffix(".json")
    return CommandTask(
        name=f"eval policy {tag} n={n} mode={mode} train_seed={train_seed} env_size={env_size}",
        kind="eval",
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
            "--v_ang_max",
            args.v_ang_max,
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
        required_paths=tuple(required_paths),
        lock_path=summary_json.with_suffix(".lock"),
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
                    kind="heuristic",
                    cmd=[
                        PYTHON,
                        str(REPO_ROOT / "run" / "eval_hungarian_p.py"),
                        "--n",
                        str(n),
                        "--env_size",
                        "20",
                        "--v_ang_max",
                        args.v_ang_max,
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
                    lock_path=summary_json.with_suffix(".lock"),
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
        kind="probe",
        cmd=[
            PYTHON,
            str(REPO_ROOT / "tools" / "probe_costs.py"),
            "--runs_dir",
            str(args.runs_dir),
            "--out_csv",
            str(out_csv),
            "--raw_meta_csv",
            str(raw_csv),
            "--v_ang_max",
            args.v_ang_max,
        ],
        done_path=out_csv,
        lock_path=out_csv.with_suffix(".lock"),
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


def offloaded_equivalent(path: Path, args: argparse.Namespace) -> Path | None:
    if args.disable_offload:
        return None
    try:
        rel = path.resolve().relative_to(args.artifact_root)
    except ValueError:
        return None
    return args.offload_root / rel


def path_exists_local_or_offloaded(path: Path, args: argparse.Namespace) -> bool:
    if path.exists():
        return True
    offloaded = offloaded_equivalent(path, args)
    return bool(offloaded is not None and offloaded.exists())


def first_existing_local_or_offloaded(path: Path, args: argparse.Namespace) -> Path | None:
    if path.exists():
        return path
    offloaded = offloaded_equivalent(path, args)
    if offloaded is not None and offloaded.exists():
        return offloaded
    return None


def is_training_result_path(path: Path) -> bool:
    return path.parent.name.startswith("n") and path.name.startswith("result") and path.suffix == ".csv"


def training_meta_is_finished(run_path: Path, args: argparse.Namespace) -> bool:
    candidates = [run_path / "meta.json"]
    offloaded = offloaded_equivalent(run_path / "meta.json", args)
    if offloaded is not None:
        candidates.append(offloaded)

    for meta_path in candidates:
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        episodes_completed = int(meta.get("episodes_completed") or 0)
        requested = int(meta.get("episodes_requested") or meta.get("n_games_target") or args.episodes)
        finished = bool(meta.get("finished")) or meta.get("launcher_status") == "finished"
        if finished and episodes_completed >= requested:
            return True
    return False


def read_training_meta(run_path: Path, args: argparse.Namespace) -> dict:
    candidates = [run_path / "meta.json"]
    offloaded = offloaded_equivalent(run_path / "meta.json", args)
    if offloaded is not None:
        candidates.append(offloaded)
    for meta_path in candidates:
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        meta["_meta_path"] = str(meta_path)
        return meta
    return {}


def csv_data_rows(path: Path) -> int | None:
    try:
        with path.open(newline="") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except FileNotFoundError:
        return None


def task_arg_value(task: CommandTask, flag: str) -> str | None:
    try:
        index = task.cmd.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(task.cmd):
        return None
    return task.cmd[index + 1]


def eval_task_is_complete(task: CommandTask, args: argparse.Namespace) -> bool:
    if task.done_path is None:
        return False
    summary_path = first_existing_local_or_offloaded(task.done_path, args)
    if summary_path is None:
        return False

    expected_episodes = int(task_arg_value(task, "--episodes") or args.eval_episodes)
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        summary = {}
    if int(summary.get("episodes") or 0) < expected_episodes:
        return False

    out_csv_arg = task_arg_value(task, "--out_csv")
    if out_csv_arg:
        out_csv = first_existing_local_or_offloaded(Path(out_csv_arg), args)
        if out_csv is None:
            return False
        rows = csv_data_rows(out_csv)
        if rows is None or rows < expected_episodes:
            return False

    if task.required_paths:
        actor = task.required_paths[0]
        if actor.name == "shared_actor.pth" and actor.exists() and summary_path.exists():
            return summary_path.stat().st_mtime >= actor.stat().st_mtime
    return True


def requirement_exists(path: Path, args: argparse.Namespace) -> bool:
    if is_training_result_path(path):
        return training_meta_is_finished(path.parent, args)
    if path.exists():
        return True
    # Evaluation opens actor_ckpt from the local path in the subprocess command.
    # Only scheduler-only completion markers, such as train result CSVs, can be
    # satisfied from the external mirror.
    if path.name == "shared_actor.pth":
        return False
    return path_exists_local_or_offloaded(path, args)


def task_is_done(task: CommandTask, args: argparse.Namespace) -> bool:
    if task.kind == "train" and task.cleanup_dir is not None:
        return (
            task.done_path is not None
            and training_meta_is_finished(task.cleanup_dir, args)
            and not args.rerun
        )
    if task.kind == "eval" and task.done_path is not None and not args.rerun:
        return eval_task_is_complete(task, args)
    return (
        task.done_path is not None
        and path_exists_local_or_offloaded(task.done_path, args)
        and not args.rerun
    )


def missing_requirements_for_task(task: CommandTask, args: argparse.Namespace) -> list[Path]:
    return [
        path
        for path in task.required_paths
        if not requirement_exists(path, args)
    ]


def pick_ready_task_index(tasks: list[CommandTask], args: argparse.Namespace) -> int | None:
    ready = [
        index
        for index, candidate in enumerate(tasks)
        if task_is_done(candidate, args) or not missing_requirements_for_task(candidate, args)
    ]
    if not ready:
        return None

    # Finished/skipped tasks are cheap to account for, then ready evals should
    # start before opening more training jobs so evaluation can overlap with
    # remaining training as soon as a corresponding actor is final.
    for index in ready:
        if task_is_done(tasks[index], args):
            return index
    for index in ready:
        if tasks[index].kind == "eval":
            return index
    return ready[0]


def run_task(task: CommandTask, args: argparse.Namespace) -> str:
    log_file = args.log_file
    if task_is_done(task, args):
        log(f"SKIP completed task#{task.index}: {task.name} ({task.done_path})", log_file)
        copy_result(task.result_copy, log_file)
        offload_finished_task(task, args)
        return "skipped_done"

    missing = missing_requirements_for_task(task, args)
    if missing:
        message = f"SKIP missing requirement task#{task.index}: {task.name} requires {missing[0]}"
        if args.strict_missing_eval_actor:
            raise FileNotFoundError(message)
        log(message, log_file)
        return "skipped_missing"

    prepared = prepare_incomplete_train_dir(task, args)
    if prepared == "running":
        return "skipped_running"

    if not acquire_task_lock(task, args):
        return "skipped_running"

    log(f"START task#{task.index}: {task.name}", log_file)
    log("CMD " + " ".join(task.cmd), log_file)
    if args.dry_run:
        log(f"DRY-RUN finish: {task.name}", log_file)
        return "dry_run"

    start = time.time()
    per_task_log = task_log_path(task, args)
    per_task_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with per_task_log.open("a") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {task.name} =====\n")
            f.write("CMD " + " ".join(task.cmd) + "\n")
            f.flush()
            proc = subprocess.Popen(
                task.cmd,
                cwd=REPO_ROOT,
                env=env_for_subprocess(),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            update_task_lock_pid(task, args, proc.pid)
            returncode = proc.wait()

        elapsed = time.time() - start
        if returncode != 0:
            log(
                f"FAIL {task.name} rc={returncode} elapsed_s={elapsed:.1f} log={per_task_log}",
                log_file,
            )
            raise subprocess.CalledProcessError(returncode, task.cmd)

        log(f"DONE {task.name} elapsed_s={elapsed:.1f} log={per_task_log}", log_file)
        copy_result(task.result_copy, log_file)
        offload_finished_task(task, args)
        return "ran"
    finally:
        release_task_lock(task, args)


def offload_finished_task(task: CommandTask, args: argparse.Namespace) -> None:
    if args.disable_offload or task.kind != "train" or task.cleanup_dir is None:
        return
    target_root = ensure_target_root(args.offload_root, dry_run=args.dry_run)
    if target_root is None:
        log(f"OFFLOAD unavailable, target missing: {args.offload_root}", args.log_file)
        return
    try:
        result = offload_run_dir(
            task.cleanup_dir,
            source_root=args.artifact_root,
            target_root=target_root,
            keep_local_result_csv=args.keep_local_result_csv,
            dry_run=args.dry_run,
            progress=False,
        )
    except Exception as exc:
        log(f"OFFLOAD failed for {task.cleanup_dir}: {exc!r}", args.log_file)
        return
    detail = (
        f"status={result.status} copied={result.copied_files} "
        f"removed={result.removed_files} freed_bytes={result.freed_bytes}"
    )
    if result.message:
        detail += f" message={result.message}"
    log(f"OFFLOAD {task.name}: {detail}", args.log_file)


def run_tasks_dynamic(
    tasks: list[CommandTask],
    args: argparse.Namespace,
    counts: dict[str, int],
    group_name: str,
) -> None:
    if not tasks:
        return

    parallel = max(1, int(args.parallel))
    pending_tasks = list(tasks)
    completed = 0
    log(f"Group {group_name}: {len(tasks)} tasks, parallel={parallel}", args.log_file)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {}

        while pending_tasks or futures:
            while pending_tasks and len(futures) < parallel:
                ready_index = pick_ready_task_index(pending_tasks, args)
                if ready_index is None:
                    break

                task = pending_tasks.pop(ready_index)
                log(
                    f"{group_name} submit task#{task.index}: {task.name}",
                    args.log_file,
                )
                futures[executor.submit(run_task, task, args)] = task

            if futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    status = future.result()
                    counts[status] = counts.get(status, 0) + 1
                    completed += 1
                    log(
                        f"{group_name} handled {completed}/{len(tasks)} task#{task.index}: {task.name} [{status}]",
                        args.log_file,
                    )
                continue

            # No internal work is running and every remaining task is waiting on
            # requirements this invocation will not create. Skip those tasks now
            # instead of spinning forever.
            blocked = pending_tasks
            pending_tasks = []
            for task in blocked:
                missing = missing_requirements_for_task(task, args)
                if missing:
                    message = f"SKIP missing requirement task#{task.index}: {task.name} requires {missing[0]}"
                    if args.strict_missing_eval_actor:
                        raise FileNotFoundError(message)
                    log(message, args.log_file)
                    status = "skipped_missing"
                else:
                    status = run_task(task, args)
                counts[status] = counts.get(status, 0) + 1
                completed += 1
                log(
                    f"{group_name} handled {completed}/{len(tasks)} task#{task.index}: {task.name} [{status}]",
                    args.log_file,
                )


def write_summary(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"counts": counts, "finished_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=2)


def task_status(task: CommandTask, args: argparse.Namespace) -> str:
    if task_is_done(task, args):
        return "complete"
    missing = missing_requirements_for_task(task, args)
    if missing:
        return "blocked_missing_requirement"
    return "pending"


def completeness_rows(tasks: list[CommandTask], args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        status = task_status(task, args)
        row: dict[str, object] = {
            "task": task.name,
            "kind": task.kind,
            "status": status,
            "done_path": str(task.done_path or ""),
            "missing_requirement": "",
        }
        missing = missing_requirements_for_task(task, args)
        if missing:
            row["missing_requirement"] = str(missing[0])

        if task.kind == "train" and task.cleanup_dir is not None:
            meta = read_training_meta(task.cleanup_dir, args)
            requested = int(meta.get("episodes_requested") or meta.get("n_games_target") or args.episodes)
            completed = int(meta.get("episodes_completed") or 0)
            row.update(
                {
                    "n": meta.get("n", ""),
                    "mode": meta.get("mode", ""),
                    "seed": meta.get("seed", ""),
                    "episodes_completed": completed,
                    "episodes_expected": requested,
                    "finished": bool(meta.get("finished")),
                    "launcher_status": meta.get("launcher_status", ""),
                    "meta_path": meta.get("_meta_path", ""),
                }
            )
        elif task.kind in {"eval", "heuristic"}:
            expected = int(task_arg_value(task, "--episodes") or args.eval_episodes)
            summary_path = first_existing_local_or_offloaded(task.done_path, args) if task.done_path else None
            out_csv_arg = task_arg_value(task, "--out_csv")
            out_csv = first_existing_local_or_offloaded(Path(out_csv_arg), args) if out_csv_arg else None
            summary = {}
            if summary_path is not None:
                try:
                    summary = json.loads(summary_path.read_text())
                except Exception:
                    summary = {}
            row.update(
                {
                    "n": task_arg_value(task, "--n") or "",
                    "mode": task_arg_value(task, "--mode") or "",
                    "seed": task_arg_value(task, "--seed") or "",
                    "episodes_completed": summary.get("episodes", csv_data_rows(out_csv) if out_csv else 0),
                    "episodes_expected": expected,
                    "summary_path": str(summary_path or ""),
                    "episode_csv": str(out_csv or ""),
                    "episode_csv_rows": csv_data_rows(out_csv) if out_csv else "",
                }
            )
        rows.append(row)
    return rows


def write_completeness_report(tasks: list[CommandTask], args: argparse.Namespace) -> dict[str, object]:
    rows = completeness_rows(tasks, args)
    report_dir = args.runs_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "run_all_completeness.csv"
    json_path = report_dir / "run_all_completeness.json"

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total": len(rows),
        "complete": sum(1 for row in rows if row["status"] == "complete"),
        "pending": sum(1 for row in rows if row["status"] == "pending"),
        "blocked_missing_requirement": sum(
            1 for row in rows if row["status"] == "blocked_missing_requirement"
        ),
        "csv": str(csv_path),
        "incomplete_tasks": [
            row
            for row in rows
            if row["status"] != "complete"
        ],
    }
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log(
        "Completeness: "
        f"complete={summary['complete']}/{summary['total']} "
        f"pending={summary['pending']} "
        f"blocked={summary['blocked_missing_requirement']} "
        f"report={csv_path}",
        args.log_file,
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.parallel < 1:
        raise ValueError("--parallel must be >= 1")

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
    args.offload_root = args.offload_root.expanduser()
    if args.offload_root.exists():
        args.offload_root = args.offload_root.resolve()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.res_dir.mkdir(parents=True, exist_ok=True)

    if args.train_only and args.eval_only:
        raise ValueError("--train_only and --eval_only cannot be used together")

    train_phase_tasks = [] if args.eval_only else train_tasks(args)
    eval_phase_tasks = [] if args.train_only else eval_tasks(args)
    probe_phase_tasks = (
        [] if args.skip_probe or args.train_only else [probe_task(args)]
    )
    all_phase_tasks = with_task_indices(
        train_phase_tasks + eval_phase_tasks + probe_phase_tasks
    )
    train_count = len(train_phase_tasks)
    eval_count = len(eval_phase_tasks)
    train_phase_tasks = all_phase_tasks[:train_count]
    eval_phase_tasks = all_phase_tasks[train_count : train_count + eval_count]
    probe_phase_tasks = all_phase_tasks[train_count + eval_count :]

    total_tasks = len(train_phase_tasks) + len(eval_phase_tasks) + len(probe_phase_tasks)
    log(
        f"Prepared {total_tasks} tasks "
        f"(train={len(train_phase_tasks)}, eval={len(eval_phase_tasks)}, "
        f"probe={len(probe_phase_tasks)}, parallel={args.parallel})",
        args.log_file,
    )
    counts = {
        "ran": 0,
        "skipped_done": 0,
        "skipped_missing": 0,
        "skipped_running": 0,
        "dry_run": 0,
    }

    write_completeness_report(all_phase_tasks, args)
    if args.check_completeness:
        return

    run_tasks_dynamic(train_phase_tasks + eval_phase_tasks, args, counts, "train/eval")
    run_tasks_dynamic(probe_phase_tasks, args, counts, "probe")
    write_completeness_report(all_phase_tasks, args)

    summary_path = args.runs_dir / "run_all_summary.json"
    write_summary(summary_path, counts)
    log(f"All requested tasks handled. Summary: {counts}", args.log_file)
    log(f"Summary JSON: {summary_path}", args.log_file)


if __name__ == "__main__":
    main()

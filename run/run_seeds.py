"""
Configurable multi-seed training runner with flexible output directories.

Allows specifying custom seeds, team sizes, modes, output folder, and parallelism.
Much lighter than run_all.py — focuses only on training, not evals/probes.

Usage Examples:
    # Test new seeds 100-102 with custom output folder
    python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full \
      --out_root /tmp/test_run --parallel 3

    # Quick test (500 episodes, 1 worker)
    python run/run_seeds.py --seeds 9832 --n 5 --mode full --episodes 500 \
      --out_root ./local_artifacts --parallel 1

    # Full batch to external drive
    python run/run_seeds.py --seeds 100 101 --n 4 5 6 --mode full ablation nocoll \
      --out_root /mnt/external_ssd/dif_driven --episodes 1000 --parallel 5

    # Default location (~/Desktop/dif_driven_revision_artifacts)
    python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full ablation
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = sys.executable


@dataclass(frozen=True)
class TrainTask:
    n: int
    mode: str
    seed: int
    out_dir: Path
    episodes: int = 1000
    max_steps: int = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed training with configurable seeds, modes, and parallelism."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[9832, 0, 13],
        help="Seeds to run. Default: [9832, 0, 13]",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        choices=[4, 5, 6],
        default=[4, 5, 6],
        help="Team sizes. Default: [4, 5, 6]",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["full", "ablation", "nocoll"],
        default=["full", "ablation"],
        help="Training modes. Default: [full, ablation]",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path.home() / "Desktop" / "dif_driven_revision_artifacts",
        help=(
            "Base output directory for all training runs. Each run creates a subdirectory "
            "n{N}_{mode}_seed{S}/ with checkpoints, logs, and metrics. "
            "Default: ~/Desktop/dif_driven_revision_artifacts. "
            "Examples: /tmp/quick_test, /mnt/external_ssd, ./local_artifacts, /home/user/my_runs"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Episodes per training run. Default: 1000",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Max steps per episode. Default: 500",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Max parallel workers. Default: 3. Use 1 for laptop, 5 for high-throughput.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print tasks without running them.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run tasks even if output CSVs already exist.",
    )
    return parser.parse_args()


def make_tasks(args: argparse.Namespace) -> list[TrainTask]:
    """Generate all training tasks from args."""
    tasks = []
    for n in args.n:
        for mode in args.mode:
            for seed in args.seeds:
                out_dir = args.out_root / f"n{n}_{mode}_seed{seed}"
                tasks.append(
                    TrainTask(
                        n=n,
                        mode=mode,
                        seed=seed,
                        out_dir=out_dir,
                        episodes=args.episodes,
                        max_steps=args.max_steps,
                    )
                )
    return tasks


def task_already_done(task: TrainTask, args: argparse.Namespace) -> bool:
    """Check if a training task has completed (look for result CSV and meta.json)."""
    if args.rerun:
        return False
    
    csv_path = task.out_dir / f"result{task.n}{'' if task.mode == 'full' else '_' + task.mode}.csv"
    meta_path = task.out_dir / "meta.json"
    
    if not csv_path.exists() or not meta_path.exists():
        return False
    
    # Check if meta.json indicates completion
    try:
        with open(meta_path) as f:
            meta = json.load(f)
            completed_eps = meta.get("episodes_completed", 0)
            return completed_eps >= task.episodes
    except Exception:
        return False


def run_task(task: TrainTask) -> tuple[str, int, bool]:
    """Execute a single training task. Returns (task_name, exit_code, success)."""
    task_name = f"n{task.n}_{task.mode}_seed{task.seed}"
    
    task.out_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        PYTHON,
        str(REPO_ROOT / "run" / "train_seeded.py"),
        "--n", str(task.n),
        "--mode", task.mode,
        "--seed", str(task.seed),
        "--episodes", str(task.episodes),
        "--out_dir", str(task.out_dir),
    ]
    
    log_path = task.out_dir / "train.log"
    
    with open(log_path, "w") as log_file:
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write(f"CWD: {task.out_dir}\n\n")
        log_file.flush()
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=task.out_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=None,
            )
            exit_code = proc.returncode
            success = exit_code == 0
        except subprocess.TimeoutExpired:
            exit_code = 124  # timeout exit code
            success = False
        except Exception as e:
            log_file.write(f"\nException: {e}\n")
            exit_code = 1
            success = False
        
        elapsed = time.time() - t0
        status_msg = f"\n{'='*60}\n"
        status_msg += f"Task {task_name}: {'✓ SUCCESS' if success else '✗ FAILED'}\n"
        status_msg += f"Exit code: {exit_code}, Elapsed: {elapsed/3600:.1f} hours\n"
        status_msg += f"{'='*60}\n"
        log_file.write(status_msg)
    
    return task_name, exit_code, success


def main():
    args = parse_args()
    
    tasks = make_tasks(args)
    
    # Filter out already-done tasks
    pending_tasks = [t for t in tasks if not task_already_done(t, args)]
    skipped_tasks = len(tasks) - len(pending_tasks)
    
    print(f"\n{'='*70}")
    print(f"Multi-Seed Training Configuration")
    print(f"{'='*70}")
    print(f"Seeds: {args.seeds}")
    print(f"Team sizes: {args.n}")
    print(f"Modes: {args.mode}")
    print(f"Episodes per run: {args.episodes}")
    print(f"Output root: {args.out_root}")
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Pending: {len(pending_tasks)}")
    print(f"Already done: {skipped_tasks}")
    print(f"Parallel workers: {args.parallel}")
    print(f"{'='*70}\n")
    
    if args.dry_run:
        print("DRY RUN: Tasks that would be executed:\n")
        for task in pending_tasks:
            print(f"  n={task.n}, mode={task.mode}, seed={task.seed} → {task.out_dir}")
        return
    
    if not pending_tasks:
        print("✓ All tasks already completed. Use --rerun to force re-execution.")
        return
    
    results = {}
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_task, task): task for task in pending_tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            try:
                task_name, exit_code, success = future.result()
                results[task_name] = (exit_code, success)
                status = "✓" if success else "✗"
                print(f"{status} {task_name}: exit_code={exit_code}")
            except Exception as e:
                print(f"✗ Error executing task: {e}")
                results[task.mode] = (1, False)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    
    success_count = sum(1 for _, success in results.values() if success)
    total_count = len(results)
    
    for task_name, (exit_code, success) in sorted(results.items()):
        status = "✓" if success else "✗"
        print(f"{status} {task_name}: exit_code={exit_code}")
    
    print(f"\nCompleted: {success_count}/{total_count}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


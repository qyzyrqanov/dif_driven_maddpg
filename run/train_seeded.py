"""Parameterized canonical trainer for revision multi-seed runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_envs.diff_driven.gym_env.centered_paralelenv.env import (  # noqa: E402
    DiffDriveParallelEnvDone,
)
from rl.maddpg import IDDPGWithoutS  # noqa: E402
from tools.offload_artifacts import ensure_target_root, offload_run_dir  # noqa: E402


SCALES = {
    "full": [1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    "ablation": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    "nocoll": [1.0, 1.0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1.0],
}

CSV_SUFFIX = {
    "full": "",
    "ablation": "_ablation",
    "nocoll": "_nocoll",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one canonical IDDPGWithoutS training configuration."
    )
    parser.add_argument("--n", type=int, choices=[4, 5, 6], required=True)
    parser.add_argument("--mode", choices=sorted(SCALES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--v_ang_max",
        choices=["pi9", "pi6", "pi2"],
        default="pi2",
        help="Angular velocity cap. Default pi2 — confirmed Z7S value (2026-05-18 evidence).",
    )
    parser.add_argument(
        "--use_offline_replay",
        action="store_true",
        help="Use MADDPGBase.main_loop with offline_replay_success (HER-style "
             "goal relabeling) instead of train_loop. Addresses sparse-reward "
             "bootstrap failure for higher N. Disclosed as revision-stage enhancement.",
    )
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=None,
        help="Source root used to preserve relative paths during episode offload.",
    )
    parser.add_argument(
        "--offload_root",
        type=Path,
        default=None,
        help="Mirror this run directory here after each saved episode checkpoint.",
    )
    parser.add_argument(
        "--disable_episode_offload",
        action="store_true",
        help="Disable per-episode artifact mirroring.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Bit-deterministic CUDA. Required for replayability across runs/resumes.
    # CUBLAS_WORKSPACE_CONFIG must be set BEFORE the first CUDA op (we set it
    # here on import-time call before env construction). ~10-30% perf cost.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        print(f"use_deterministic_algorithms not available: {exc!r}", flush=True)


def rename_rewards_csv(num_agents: int, mode: str) -> str | None:
    """Move rewards.csv -> result{N}{suffix}.csv at end of run.

    Resume-safe: if a target already exists from a prior completed run in the
    same out_dir, merge by episode_id (target rows for episodes also present
    in source are dropped in favor of source rows; new source episodes are
    appended). This prevents truncating a full result CSV when a run is
    relaunched and only produces a few additional episodes.
    """
    source = Path("rewards.csv")
    target = Path(f"result{num_agents}{CSV_SUFFIX[mode]}.csv")

    if not source.exists():
        return str(target) if target.exists() else None

    if target.exists():
        import pandas as pd
        old = pd.read_csv(target)
        new = pd.read_csv(source)
        if "episode_id" in old.columns and "episode_id" in new.columns:
            new_episodes = set(new["episode_id"].unique())
            old_keep = old[~old["episode_id"].isin(new_episodes)]
            combined = pd.concat([old_keep, new], ignore_index=True)
            combined = combined.sort_values(["episode_id", "timestep"]
                                            if "timestep" in combined.columns
                                            else ["episode_id"]).reset_index(drop=True)
        else:
            combined = pd.concat([old, new], ignore_index=True)
        combined.to_csv(target, index=False)
        source.unlink()
    else:
        source.rename(target)
    return str(target)


def update_meta(meta_path: Path, extra: dict) -> None:
    if meta_path.exists():
        with meta_path.open("r") as f:
            meta = json.load(f)
    else:
        meta = {}
    meta.update(extra)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def parse_v_ang_max(value: str) -> torch.Tensor:
    if value == "pi9":
        return torch.pi / 9
    if value == "pi6":
        return torch.pi / 6
    if value == "pi2":
        return torch.pi / 2
    raise ValueError(f"Unsupported --v_ang_max: {value}")


def make_episode_offload_callback(
    *,
    out_dir: Path,
    artifact_root: Path | None,
    offload_root: Path | None,
    disabled: bool,
) -> Callable[[int, bool], None] | None:
    if disabled or offload_root is None:
        return None

    source_root = (artifact_root or out_dir.parent.parent).expanduser().resolve()
    target_root = ensure_target_root(offload_root, dry_run=False)
    if target_root is None:
        print(f"Episode offload disabled: target root unavailable: {offload_root}", flush=True)
        return None

    def callback(episodes_completed: int, finished: bool) -> None:
        try:
            result = offload_run_dir(
                out_dir,
                source_root=source_root,
                target_root=target_root,
                keep_local_result_csv=True,
                prune_incomplete_snapshots=False,
                include_running=True,
                dry_run=False,
                progress=False,
            )
            print(
                "Episode offload "
                f"episode={episodes_completed} finished={finished} "
                f"status={result.status} copied={result.copied_files} "
                f"removed={result.removed_files} target={result.target}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"Episode offload failed at episode={episodes_completed}: {exc!r}",
                flush=True,
            )

    return callback


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(args.seed)

    start_time = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    os.chdir(out_dir)

    v_ang_max = parse_v_ang_max(args.v_ang_max)
    env = DiffDriveParallelEnvDone(
        num_agents=args.n,
        num_obstacles=0,
        v_ang_max=v_ang_max,
    )
    maddpg = IDDPGWithoutS(
        env,
        reward_scales=SCALES[args.mode],
        batch_size=128,
        replay_buffer_size=50000,
    )

    meta_extra = {
        "n": args.n,
        "mode": args.mode,
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "v_ang_max": args.v_ang_max,
        "v_ang_max_float": float(v_ang_max),
        "use_offline_replay": bool(args.use_offline_replay),
        "training_loop": "main_loop" if args.use_offline_replay else "train_loop",
        "out_dir": str(out_dir),
        "command_start_iso": start_iso,
    }
    post_episode_callback = make_episode_offload_callback(
        out_dir=out_dir,
        artifact_root=args.artifact_root,
        offload_root=args.offload_root,
        disabled=args.disable_episode_offload,
    )

    try:
        loop_fn = maddpg.main_loop if args.use_offline_replay else maddpg.train_loop
        loop_fn(
            start_training_after=500,
            train_each=100,
            patience=256,
            min_episodes_before_early_stop=10000,
            score_avg_window=256,
            max_steps=500,
            n_games=args.episodes,
            meta_extra=meta_extra,
            post_episode_callback=post_episode_callback,
        )
    except BaseException as exc:
        update_meta(
            Path("meta.json"),
            {
                **meta_extra,
                "command_end_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "command_wall_seconds": time.time() - start_time,
                "launcher_status": "failed",
                "launcher_error": repr(exc),
            },
        )
        raise
    else:
        renamed_csv = rename_rewards_csv(args.n, args.mode)
        update_meta(
            Path("meta.json"),
            {
                **meta_extra,
                "command_end_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "command_wall_seconds": time.time() - start_time,
                "launcher_status": "finished",
                "result_csv": renamed_csv,
            },
        )


if __name__ == "__main__":
    main()

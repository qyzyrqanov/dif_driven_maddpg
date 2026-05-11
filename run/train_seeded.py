"""Parameterized canonical trainer for revision multi-seed runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_envs.diff_driven.gym_env.centered_paralelenv.env import (  # noqa: E402
    DiffDriveParallelEnvDone,
)
from rl.maddpg import IDDPGWithoutS  # noqa: E402


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
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rename_rewards_csv(num_agents: int, mode: str) -> str | None:
    source = Path("rewards.csv")
    if not source.exists():
        return None

    target = Path(f"result{num_agents}{CSV_SUFFIX[mode]}.csv")
    if target.exists():
        target.unlink()
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


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(args.seed)

    start_time = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    os.chdir(out_dir)

    env = DiffDriveParallelEnvDone(
        num_agents=args.n,
        num_obstacles=0,
        v_ang_max=torch.pi / 2,
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
        "out_dir": str(out_dir),
        "command_start_iso": start_iso,
    }

    try:
        maddpg.train_loop(
            start_training_after=500,
            train_each=100,
            patience=256,
            min_episodes_before_early_stop=10000,
            score_avg_window=256,
            max_steps=500,
            n_games=args.episodes,
            meta_extra=meta_extra,
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

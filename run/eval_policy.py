"""Evaluate saved canonical actors and write per-episode/per-step CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import device  # noqa: E402
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import (  # noqa: E402
    DiffDriveParallelEnvDone,
)
from models.simpleactor import SimpleActor  # noqa: E402


SCALES = {
    "full": [1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    "ablation": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    "nocoll": [1.0, 1.0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1.0],
}


PolicyFn = Callable[[torch.Tensor, DiffDriveParallelEnvDone, int, int], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SimpleActor checkpoint."
    )
    parser.add_argument("--actor_ckpt", type=Path, required=True)
    parser.add_argument("--n", type=int, choices=[4, 5, 6], required=True)
    parser.add_argument("--env_size", type=float, default=20.0)
    parser.add_argument("--num_obstacles", type=int, default=0)
    parser.add_argument(
        "--v_ang_max",
        choices=["pi9", "pi6", "pi2"],
        default="pi9",
        help="Angular velocity cap. Default pi9 is the corrected canonical setup.",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=sorted(SCALES), default="full")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--steps_csv", type=Path, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--max_steps", type=int, default=500)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(
    *,
    num_agents: int,
    env_size: float,
    num_obstacles: int = 0,
    v_ang_max: str = "pi9",
) -> DiffDriveParallelEnvDone:
    if v_ang_max == "pi9":
        v_ang = torch.pi / 9
    elif v_ang_max == "pi6":
        v_ang = torch.pi / 6
    elif v_ang_max == "pi2":
        v_ang = torch.pi / 2
    else:
        raise ValueError(f"Unsupported v_ang_max: {v_ang_max}")
    return DiffDriveParallelEnvDone(
        num_agents=num_agents,
        env_size=env_size,
        num_obstacles=num_obstacles,
        v_ang_max=v_ang,
    )


def load_actor(actor_ckpt: Path, env: DiffDriveParallelEnvDone) -> SimpleActor:
    actor = SimpleActor(
        env.obs_dim,
        env.action_dim,
        device=device,
        chckpnt_file=str(actor_ckpt),
    )
    actor.load_checkpoint(filepath=str(actor_ckpt), raise_on_no_file=True)
    actor.eval()
    return actor


def make_actor_policy(actor: SimpleActor) -> PolicyFn:
    def policy(obs: torch.Tensor, env: DiffDriveParallelEnvDone, episode: int, step: int) -> torch.Tensor:
        del env, episode, step
        return actor.choose_action(obs, use_noise=False, eval_mode=True)

    return policy


def step_columns(num_agents: int) -> list[str]:
    cols = ["total_step", "episode_id", "timestep", "done_count"]
    cols += [f"hung_dist_agent{i}" for i in range(num_agents)]
    cols += [f"agent{i}_vel_lin" for i in range(num_agents)]
    cols += [f"agent{i}_vel_ang" for i in range(num_agents)]
    cols += [f"agent{i}_comp{j}" for i in range(num_agents) for j in range(1, 10)]
    return cols


def build_step_row(
    *,
    env: DiffDriveParallelEnvDone,
    reward_scales: torch.Tensor,
    total_step: int,
    episode_id: int,
) -> dict[str, float | int]:
    num_agents = env.num_agents
    comps = env.current_rewards
    weighted = comps * reward_scales.to(device=comps.device, dtype=comps.dtype)

    row: dict[str, float | int] = {
        "total_step": int(total_step),
        "episode_id": int(episode_id),
        "timestep": int(env.timestep),
        "done_count": int(env.dones.sum().item()),
    }

    for i, value in enumerate(env.old_hungarian.detach().float().cpu().tolist()):
        row[f"hung_dist_agent{i}"] = float(value)
    for i, value in enumerate(env.agent_vel_lin.detach().float().cpu().tolist()):
        row[f"agent{i}_vel_lin"] = float(value)
    for i, value in enumerate(env.agent_vel_ang.detach().float().cpu().tolist()):
        row[f"agent{i}_vel_ang"] = float(value)

    flat_components = weighted.reshape(-1).detach().float().cpu().tolist()
    comp_cols = [f"agent{i}_comp{j}" for i in range(num_agents) for j in range(1, 10)]
    for name, value in zip(comp_cols, flat_components):
        row[name] = float(value)

    return row


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def episode_fieldnames(num_agents: int) -> list[str]:
    cols = [
        "episode_id",
        "success",
        "completion_time",
        "steps",
        "episode_return",
        "done_count_final",
        "collision_step_rate",
        "collision_penalty_per_step_raw",
        "collision_penalty_per_step_scaled",
        "path_length_total",
    ]
    cols += [f"agent{i}_path_length" for i in range(num_agents)]
    return cols


def run_episode(
    *,
    env: DiffDriveParallelEnvDone,
    policy: PolicyFn,
    episode_id: int,
    max_steps: int,
    reward_scales: torch.Tensor,
    steps_csv: Path | None,
    total_step_start: int,
) -> tuple[dict[str, float | int | str], int]:
    _, obs = env.reset_tensor()
    done = env.get_dones_tensor().clone()
    total_step = total_step_start
    step_count = 0
    completion_time = ""
    episode_return = 0.0
    collision_steps = 0
    raw_collision_penalty = 0.0
    scaled_collision_penalty = 0.0
    path_lengths = torch.zeros(env.num_agents, dtype=torch.float32, device=env.device)
    step_fieldnames = step_columns(env.num_agents)

    while step_count < max_steps and not done.all():
        prev_pos = env.agent_pos.clone()
        actions = policy(obs, env, episode_id, step_count)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=env.device)
        actions = actions.to(env.device, dtype=torch.float32)

        _, obs, rewards, done = env.step_tensor(actions)
        total_step += 1
        step_count += 1

        path_lengths += torch.norm(env.agent_pos - prev_pos, dim=1)
        weighted_rewards = rewards * reward_scales.to(device=rewards.device, dtype=rewards.dtype)
        episode_return += float(weighted_rewards.sum().item())

        raw_comp5_sum = float(rewards[:, 4].sum().item())
        scaled_comp5_sum = float(weighted_rewards[:, 4].sum().item())
        raw_collision_penalty += raw_comp5_sum
        scaled_collision_penalty += scaled_comp5_sum
        if abs(raw_comp5_sum) > 1e-8:
            collision_steps += 1

        if completion_time == "" and done.all():
            completion_time = int(env.timestep)

        if steps_csv is not None:
            append_csv_row(
                steps_csv,
                step_fieldnames,
                build_step_row(
                    env=env,
                    reward_scales=reward_scales,
                    total_step=total_step,
                    episode_id=episode_id,
                ),
            )

    success = int(done.all().item())
    denom = max(step_count, 1)
    path_lengths_cpu = path_lengths.detach().cpu().tolist()
    row: dict[str, float | int | str] = {
        "episode_id": episode_id,
        "success": success,
        "completion_time": completion_time,
        "steps": step_count,
        "episode_return": episode_return,
        "done_count_final": int(done.sum().item()),
        "collision_step_rate": collision_steps / denom,
        "collision_penalty_per_step_raw": -min(raw_collision_penalty, 0.0) / denom,
        "collision_penalty_per_step_scaled": -min(scaled_collision_penalty, 0.0) / denom,
        "path_length_total": float(sum(path_lengths_cpu)),
    }
    for i, value in enumerate(path_lengths_cpu):
        row[f"agent{i}_path_length"] = float(value)

    return row, total_step


def evaluate_policy(
    *,
    env: DiffDriveParallelEnvDone,
    policy: PolicyFn,
    episodes: int,
    seed: int,
    reward_scales: list[float],
    out_csv: Path,
    steps_csv: Path | None = None,
    max_steps: int = 500,
) -> list[dict]:
    reward_scales_tensor = torch.tensor(reward_scales, dtype=torch.float32, device=env.device)
    fields = episode_fieldnames(env.num_agents)
    rows = []
    total_step = 0

    if out_csv.exists():
        out_csv.unlink()
    if steps_csv is not None and steps_csv.exists():
        steps_csv.unlink()

    for episode_id in range(episodes):
        set_seeds(seed + episode_id)
        row, total_step = run_episode(
            env=env,
            policy=policy,
            episode_id=episode_id,
            max_steps=max_steps,
            reward_scales=reward_scales_tensor,
            steps_csv=steps_csv,
            total_step_start=total_step,
        )
        append_csv_row(out_csv, fields, row)
        rows.append(row)
        print(
            f"Episode {episode_id}: success={row['success']} "
            f"steps={row['steps']} return={row['episode_return']:.2f}"
        )

    return rows


def summarize(rows: list[dict], args: argparse.Namespace, wall_seconds: float) -> dict:
    successes = sum(int(row["success"]) for row in rows)
    completion_times = [
        float(row["completion_time"]) for row in rows if row["completion_time"] != ""
    ]
    returns = [float(row["episode_return"]) for row in rows]
    return {
        "actor_ckpt": str(args.actor_ckpt),
        "n": args.n,
        "env_size": args.env_size,
        "num_obstacles": args.num_obstacles,
        "episodes": args.episodes,
        "seed": args.seed,
        "mode": args.mode,
        "max_steps": args.max_steps,
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "mean_completion_time": float(np.mean(completion_times)) if completion_times else None,
        "mean_episode_return": float(np.mean(returns)) if returns else None,
        "wall_seconds": wall_seconds,
        "out_csv": str(args.out_csv),
        "steps_csv": str(args.steps_csv) if args.steps_csv else None,
    }


def main() -> None:
    args = parse_args()
    args.actor_ckpt = args.actor_ckpt.resolve()
    args.out_csv = args.out_csv.resolve()
    if args.steps_csv is None:
        args.steps_csv = args.out_csv.with_name(f"{args.out_csv.stem}_steps.csv")
    else:
        args.steps_csv = args.steps_csv.resolve()
    if args.summary_json is None:
        args.summary_json = args.out_csv.with_suffix(".json")
    else:
        args.summary_json = args.summary_json.resolve()

    set_seeds(args.seed)
    env = build_env(
        num_agents=args.n,
        env_size=args.env_size,
        num_obstacles=args.num_obstacles,
        v_ang_max=args.v_ang_max,
    )
    actor = load_actor(args.actor_ckpt, env)

    start = time.time()
    rows = evaluate_policy(
        env=env,
        policy=make_actor_policy(actor),
        episodes=args.episodes,
        seed=args.seed,
        reward_scales=SCALES[args.mode],
        out_csv=args.out_csv,
        steps_csv=args.steps_csv,
        max_steps=args.max_steps,
    )
    summary = summarize(rows, args, time.time() - start)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

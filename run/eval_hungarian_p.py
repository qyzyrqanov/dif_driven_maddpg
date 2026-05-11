"""Evaluate a Hungarian-assignment proportional-controller baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.eval_policy import SCALES, build_env, evaluate_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Hungarian assignment plus proportional control."
    )
    parser.add_argument("--n", type=int, choices=[4, 5, 6], required=True)
    parser.add_argument("--env_size", type=float, default=20.0)
    parser.add_argument("--num_obstacles", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--Kp", type=float, default=2.0)
    parser.add_argument("--mode", choices=sorted(SCALES), default="full")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--steps_csv", type=Path, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--max_steps", type=int, default=500)
    return parser.parse_args()


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def hungarian_p_actions(env, kp: float) -> torch.Tensor:
    actions = torch.zeros((env.num_agents, env.action_dim), dtype=torch.float32, device=env.device)
    _, _, _, assigned_landmarks = env.get_hungarian_distances()
    dv_lin_max = float(getattr(env, "dv_lin_max", env.v_lin_max / 4))
    dv_ang_max = float(getattr(env, "dv_ang_max", env.v_ang_max / 4))

    active_agents = (~env.dones).nonzero(as_tuple=True)[0]
    if active_agents.numel() == 0:
        return actions

    for agent_idx in active_agents.tolist():
        landmark_idx = int(assigned_landmarks[agent_idx].item())
        if landmark_idx < 0:
            continue

        target = env.landmarks[landmark_idx]
        delta = target - env.agent_pos[agent_idx]
        distance = torch.linalg.norm(delta)
        desired_heading = torch.atan2(delta[1], delta[0])
        heading_error = wrap_angle(desired_heading - env.agent_dir[agent_idx])

        target_v_lin = torch.clamp(distance, 0.0, float(env.v_lin_max))
        dv_lin = torch.clamp(
            target_v_lin - env.agent_vel_lin[agent_idx],
            -dv_lin_max,
            dv_lin_max,
        )
        target_v_ang = torch.clamp(
            kp * heading_error,
            -float(env.v_ang_max),
            float(env.v_ang_max),
        )
        dv_ang = torch.clamp(
            target_v_ang - env.agent_vel_ang[agent_idx],
            -dv_ang_max,
            dv_ang_max,
        )

        actions[agent_idx, 0] = dv_lin / dv_lin_max
        actions[agent_idx, 1] = dv_ang / dv_ang_max

    return actions.clamp(-1.0, 1.0)


def make_hungarian_policy(kp: float):
    def policy(obs, env, episode: int, step: int) -> torch.Tensor:
        del obs, episode, step
        return hungarian_p_actions(env, kp=kp)

    return policy


def summarize(rows: list[dict], args: argparse.Namespace, wall_seconds: float) -> dict:
    successes = sum(int(row["success"]) for row in rows)
    completion_times = [
        float(row["completion_time"]) for row in rows if row["completion_time"] != ""
    ]
    returns = [float(row["episode_return"]) for row in rows]
    return {
        "policy": "hungarian_p",
        "n": args.n,
        "env_size": args.env_size,
        "num_obstacles": args.num_obstacles,
        "episodes": args.episodes,
        "seed": args.seed,
        "Kp": args.Kp,
        "mode": args.mode,
        "max_steps": args.max_steps,
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "mean_completion_time": (
            sum(completion_times) / len(completion_times) if completion_times else None
        ),
        "mean_episode_return": sum(returns) / len(returns) if returns else None,
        "wall_seconds": wall_seconds,
        "out_csv": str(args.out_csv),
        "steps_csv": str(args.steps_csv) if args.steps_csv else None,
    }


def main() -> None:
    args = parse_args()
    args.out_csv = args.out_csv.resolve()
    if args.steps_csv is None:
        args.steps_csv = args.out_csv.with_name(f"{args.out_csv.stem}_steps.csv")
    else:
        args.steps_csv = args.steps_csv.resolve()
    if args.summary_json is None:
        args.summary_json = args.out_csv.with_suffix(".json")
    else:
        args.summary_json = args.summary_json.resolve()

    env = build_env(
        num_agents=args.n,
        env_size=args.env_size,
        num_obstacles=args.num_obstacles,
    )

    start = time.time()
    rows = evaluate_policy(
        env=env,
        policy=make_hungarian_policy(args.Kp),
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

"""For each N in {4,5,6}, run 7 episodes at v_ang_max=pi/9, agent_radius=0.5,
then compare per-reward-component sums (mean over agents) against Z7S log.

Usage: PYTHONPATH=. .venv310/bin/python tools/compare_components.py
"""
import os, re, sys, math, tempfile, shutil, json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone
from rl.maddpg import IDDPGWithoutS

SEED = 9832
N_EPS = 7
MAX_STEPS = 500
SCALE = [1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
COMP_NAMES = ["progressive", "distance", "base", "reached", "agent_coll",
              "obs_coll", "v_lin", "v_ang/dir", "time"]

EP_RE = re.compile(r"Episode (\d+),")
TENSOR_RE = re.compile(r"reward components:tensor\(\[(.*?)\]\s*,\s*device", re.S)


def parse_components(path):
    """Return list[ndarray(N,9)] per episode (in order)."""
    with open(path) as f:
        text = f.read()
    # Split by 'Episode N,' markers; capture episode idx and following block
    parts = re.split(r"(Episode \d+, Mean Score)", text)
    episodes = {}
    # parts: [pre, 'Episode 0, Mean Score', body, 'Episode 1, Mean Score', body, ...]
    for i in range(1, len(parts) - 1, 2):
        head = parts[i]
        body = parts[i + 1]
        m = re.match(r"Episode (\d+),", head)
        idx = int(m.group(1))
        tm = TENSOR_RE.search(body)
        if not tm:
            continue
        raw = tm.group(1)
        # raw is like "[a, b, ...],\n  [c, d, ...]"
        rows = re.findall(r"\[([^\[\]]+)\]", raw)
        mat = []
        for r in rows:
            vals = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", r)]
            if len(vals) == 9:
                mat.append(vals)
        if mat:
            episodes[idx] = np.array(mat)
    return [episodes[i] for i in range(N_EPS) if i in episodes]


def run(n, workdir):
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    env = DiffDriveParallelEnvDone(
        v_ang_max=math.pi / 2, agent_radius=0.5,
        num_agents=n, num_obstacles=0,
    )
    m = IDDPGWithoutS(env, reward_scales=SCALE, batch_size=128, replay_buffer_size=50000)
    cwd0 = os.getcwd(); os.chdir(workdir)
    try:
        m.train_loop(n_games=N_EPS, start_training_after=500, train_each=100,
                     patience=10**9, min_episodes_before_early_stop=10**9,
                     score_avg_window=256, max_steps=MAX_STEPS)
    finally:
        os.chdir(cwd0)
    return parse_components(os.path.join(workdir, "episode_log.txt"))


def main():
    summary = {}
    for n in (4, 5, 6):
        z7s = parse_components(f"/media/abz/Z7S/experiments/3-6/{n}/episode_log.txt")
        tmp = tempfile.mkdtemp(prefix=f"cmp_n{n}_")
        try:
            mine = run(n, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n========== n={n} ==========")
        # Mean over agents → [9]
        print(f"{'ep':>2} | comp         |  Z7S mean    mine    diff")
        per_ep = []
        for e in range(min(len(z7s), len(mine), N_EPS)):
            z = z7s[e].mean(axis=0)
            m_ = mine[e].mean(axis=0)
            per_ep.append((z, m_))
            for k in range(9):
                d = m_[k] - z[k]
                marker = " <<" if abs(d) > max(1.0, 0.10 * abs(z[k])) else ""
                print(f"{e:>2} | {COMP_NAMES[k]:<12} | {z[k]:>10.2f} {m_[k]:>10.2f} {d:>+8.2f}{marker}")
            print()
        # Summary: mean abs error per component across episodes
        if per_ep:
            zs = np.stack([z for z, _ in per_ep])
            ms = np.stack([m for _, m in per_ep])
            mae = np.abs(ms - zs).mean(axis=0)
            print(f"  MAE per component (ep0..{len(per_ep)-1}):")
            for k in range(9):
                print(f"    {COMP_NAMES[k]:<12}: {mae[k]:>8.2f}  (z_mean_abs={np.abs(zs[:,k]).mean():.2f})")
            summary[n] = {"mae": mae.tolist(), "z_mean_abs": np.abs(zs).mean(axis=0).tolist()}
    with open(os.path.join(PROJECT_ROOT, ".ai", "component_diff_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

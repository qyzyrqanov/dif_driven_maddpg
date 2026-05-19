"""Sweep (v_ang_max, agent_radius) and compare 7-episode trace to Z7S n=4 reference.

Run: PYTHONPATH=. .venv310/bin/python tools/sweep_vang_radius.py
"""
import math, os, re, sys, tempfile, shutil, time, json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone
from rl.maddpg import IDDPGWithoutS

SEED = 9832
N_AGENTS = int(os.environ.get("SWEEP_N", "4"))
N_EPISODES = 7
MAX_STEPS = 500

Z7S_LOG = f"/media/abz/Z7S/experiments/3-6/{N_AGENTS}/episode_log.txt"

SCALE = [1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]

V_ANG_CANDIDATES = [
    ("pi/12", math.pi / 12),
    ("pi/10", math.pi / 10),
    ("pi/9",  math.pi / 9),
    ("pi/6",  math.pi / 6),
    ("pi/4",  math.pi / 4),
    ("pi/3",  math.pi / 3),
]
RADIUS_CANDIDATES = [0.5, 1.0, 2.0]


def parse_log(path):
    """Return list of (mean_score, tagged_count) for first N_EPISODES."""
    rx = re.compile(r"Episode (\d+), Mean Score: (-?[\d.]+), Tagged count: (\d+)")
    out = {}
    with open(path) as f:
        for line in f:
            m = rx.search(line)
            if m:
                out[int(m.group(1))] = (float(m.group(2)), int(m.group(3)))
    return [out[i] for i in range(N_EPISODES) if i in out]


def run_one(v_ang, radius, workdir):
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    env = DiffDriveParallelEnvDone(
        v_ang_max=v_ang,
        agent_radius=radius,
        num_agents=N_AGENTS,
        num_obstacles=0,
    )
    maddpg = IDDPGWithoutS(env, reward_scales=SCALE, batch_size=128, replay_buffer_size=50000)
    cwd0 = os.getcwd()
    os.chdir(workdir)
    try:
        maddpg.train_loop(
            n_games=N_EPISODES,
            start_training_after=500,
            train_each=100,
            patience=10**9,
            min_episodes_before_early_stop=10**9,
            score_avg_window=256,
            max_steps=MAX_STEPS,
        )
    finally:
        os.chdir(cwd0)
    return parse_log(os.path.join(workdir, "episode_log.txt"))


def main():
    ref = parse_log(Z7S_LOG)[:N_EPISODES]
    ref_tags = [t for _, t in ref]
    ref_scores = [s for s, _ in ref]
    print(f"Z7S ref tags: {ref_tags}")
    print(f"Z7S ref scores: {[f'{s:.1f}' for s in ref_scores]}")
    print()

    results = []
    for label, v in V_ANG_CANDIDATES:
        for r in RADIUS_CANDIDATES:
            tmp = tempfile.mkdtemp(prefix=f"sweep_v{label.replace('/', '_')}_r{r}_")
            t0 = time.time()
            try:
                trace = run_one(v, r, tmp)
            except Exception as e:
                print(f"v_ang={label:6s} r={r} FAILED: {e}")
                shutil.rmtree(tmp, ignore_errors=True)
                continue
            dt = time.time() - t0
            tags = [t for _, t in trace]
            scores = [s for s, _ in trace]
            tag_match = sum(int(a == b) for a, b in zip(tags, ref_tags))
            # Mean-error across episodes 1..6 (skip episode 0 which trains less)
            err_window = list(zip(scores[1:], ref_scores[1:]))
            mean_err = sum(abs(a - b) for a, b in err_window) / max(1, len(err_window))
            print(f"v_ang={label:6s} r={r:>4} | tags={tags} match={tag_match}/{len(ref_tags)} "
                  f"mean_err(ep1-6)={mean_err:7.1f}  t={dt:5.1f}s")
            results.append({
                "v_ang": label, "radius": r,
                "tags": tags, "scores": scores,
                "tag_match": tag_match, "mean_err": mean_err,
                "wall_seconds": dt,
            })
            shutil.rmtree(tmp, ignore_errors=True)

    results.sort(key=lambda r: (-r["tag_match"], r["mean_err"]))
    print("\n=== Ranked (best first) ===")
    for r in results:
        print(f"  {r['v_ang']:6s} r={r['radius']:>4}  tags_match={r['tag_match']}/{len(ref_tags)}  err={r['mean_err']:7.1f}")

    out_path = os.path.join(PROJECT_ROOT, ".ai", f"sweep_vang_radius_results_n{N_AGENTS}.json")
    with open(out_path, "w") as f:
        json.dump({"ref_tags": ref_tags, "ref_scores": ref_scores, "results": results}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

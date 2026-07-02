"""Sweep the collision-penalty decay constant c in -exp(-c*d/safe_dist)
for components 4 and 5. Compare to Z7S log for n=4,5,6.
"""
import os, re, sys, math, tempfile, shutil, json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
LEGACY_LOG_ROOT = os.path.expanduser(os.environ.get("LEGACY_LOG_ROOT", "~/dif_driven_archive"))
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone
from rl.maddpg import IDDPGWithoutS

SEED = 9832
N_EPS = 7
MAX_STEPS = 500
SCALE = [1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
C_VALUES = [5.0, 15.0]


def make_env_cls(c):
    class _Env(DiffDriveParallelEnvDone):
        def _compute_rewards_tensor(self):
            N = self._num_agents
            device = self.device
            components = torch.zeros(N, 9, device=device)
            active_agents = ~self.dones
            active_agent_pos = self.agent_pos[active_agents]

            new_hungarian, cov_a, cov_l, assigned_lm = self.get_hungarian_distances()
            d_global = -new_hungarian.sum() / (self.env_size * self.num_agents)
            d_goal = torch.nan_to_num(-new_hungarian / self.env_size, nan=0.0)
            progressive = torch.clamp(torch.nan_to_num(
                (self.old_hungarian - new_hungarian) / self.v_lin_max, nan=0.0), -1, 1)
            components[active_agents, 0] = progressive[active_agents]
            components[active_agents, 1] = d_goal[active_agents]
            components[active_agents, 2] = d_global
            components[cov_a, 3] = 1
            self.covered[cov_l] = True

            if active_agent_pos.shape[0] > 1:
                delta = active_agent_pos[:, None, :] - active_agent_pos[None, :, :]
                dist = torch.norm(delta, dim=2)
                aa = (dist < self.safe_dist) & (~torch.eye(dist.shape[0], device=device, dtype=torch.bool))
                inv = torch.zeros_like(dist)
                inv[aa] = -torch.exp(-dist[aa] * c / self.safe_dist)
                components[active_agents, 4] = inv.sum(dim=1)

            done_pos = self.agent_pos[self.dones]
            nd = int(done_pos.shape[0])
            ob_pos = torch.cat([self.obstacle_pos, done_pos], dim=0)
            ap = active_agent_pos.unsqueeze(1); ob = ob_pos.unsqueeze(0)
            eff = torch.norm(ap - ob, dim=2)
            ob_mask = eff < self.safe_dist
            inv = torch.zeros_like(eff)
            inv[ob_mask] = -torch.exp(-eff[ob_mask] * c / self.safe_dist)
            components[active_agents, 5] = inv.sum(dim=1)

            lin = -(1.0 - (self.agent_vel_lin[active_agents].abs() / self.v_lin_max))
            components[active_agents, 6] = lin

            tgt = torch.full_like(self.agent_pos, float("nan"), device=device)
            if assigned_lm.shape[0] > 0:
                tgt[active_agents] = self.landmarks[assigned_lm[active_agents]]
            to_lm = tgt - self.agent_pos
            to_lm = to_lm / torch.norm(to_lm, dim=1, keepdim=True).clamp(min=1e-8)
            dirv = torch.stack([torch.cos(self.agent_dir), torch.sin(self.agent_dir)], dim=1)
            cos_sim = (dirv * to_lm).sum(dim=1)
            dr = cos_sim - 1.0; dr[~active_agents] = 0.0
            components[:, 7] = dr

            components[active_agents, 8] = -1
            self.current_rewards = components
            self.old_hungarian = new_hungarian
            self.terminate_agents(cov_a)
            return components
    return _Env


EP_RE = re.compile(r"Episode (\d+),")
TENSOR_RE = re.compile(r"reward components:tensor\(\[(.*?)\]\s*,\s*device", re.S)
def parse_components(path):
    with open(path) as f: text = f.read()
    parts = re.split(r"(Episode \d+, Mean Score)", text)
    eps = {}
    for i in range(1, len(parts) - 1, 2):
        idx = int(re.match(r"Episode (\d+),", parts[i]).group(1))
        tm = TENSOR_RE.search(parts[i + 1])
        if not tm: continue
        rows = re.findall(r"\[([^\[\]]+)\]", tm.group(1))
        mat = []
        for r in rows:
            vals = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", r)]
            if len(vals) == 9: mat.append(vals)
        if mat: eps[idx] = np.array(mat)
    return [eps[i] for i in range(N_EPS) if i in eps]


def run(n, env_cls, workdir):
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    env = env_cls(v_ang_max=math.pi/2, agent_radius=0.5, num_agents=n, num_obstacles=0)
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
        z7s = parse_components(f"{LEGACY_LOG_ROOT}/experiments/3-6/{n}/episode_log.txt")[:N_EPS]
        z_stack = np.stack([z7s[e].mean(axis=0) for e in range(N_EPS)])
        z_coll4 = z_stack[:, 4]; z_coll5 = z_stack[:, 5]
        z_score = (z_stack * SCALE).sum(axis=1).mean()
        print(f"\n========== n={n}   Z7S mean comp4={z_coll4.mean():.3f}  comp5={z_coll5.mean():.3f}  score={z_score:.1f}")
        print(f"{'c':>4} | {'comp4 MAE':>10} {'comp5 MAE':>10} {'score':>10} {'score err':>10}")
        rows = {}
        for c in C_VALUES:
            tmp = tempfile.mkdtemp(prefix=f"c{c}_n{n}_")
            try:
                trace = run(n, make_env_cls(c), tmp)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            stack = np.stack([trace[e].mean(axis=0) for e in range(min(N_EPS, len(trace)))])
            mae4 = np.abs(stack[:, 4] - z_coll4).mean()
            mae5 = np.abs(stack[:, 5] - z_coll5).mean()
            sc = (stack * SCALE).sum(axis=1).mean()
            print(f"{c:>4.0f} | {mae4:>10.3f} {mae5:>10.3f} {sc:>10.1f} {abs(sc - z_score):>10.1f}")
            rows[c] = {"mae4": float(mae4), "mae5": float(mae5),
                       "score": float(sc), "score_err": float(abs(sc - z_score))}
        summary[n] = {"z_coll4": float(z_coll4.mean()), "z_coll5": float(z_coll5.mean()),
                      "z_score": float(z_score), "rows": rows}
    with open(os.path.join(PROJECT_ROOT, ".ai", "sweep_coll_c_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

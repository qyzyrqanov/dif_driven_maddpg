"""Compare current vs Z7S-legacy reward formulas against logged Z7S training.

Builds a legacy subclass of DiffDriveParallelEnvDone that restores pre-cec3753
formulas for components 3 (reached), 4 (agent-coll), 5 (obs-coll), 6 (v_lin),
and 7 (v_ang). Components 0,1,2,8 are unchanged across cec3753.

Run: PYTHONPATH=. .venv310/bin/python tools/compare_legacy.py
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


class DiffDriveParallelEnvDoneLegacy(DiffDriveParallelEnvDone):
    """Pre-cec3753 reward formulas. Restores: smooth reached (comp 3),
    1/(d+0.01) collision penalties with edge-to-edge distance (comp 4, 5),
    old lin_penalty (comp 6), and abs-angular penalty (comp 7)."""

    def _compute_rewards_tensor(self) -> torch.Tensor:
        N = self._num_agents
        device = self.device
        components = torch.zeros(N, 9, device=device)
        agent_radius = float(self.agent_radius)

        active_agents = ~self.dones
        active_agent_pos = self.agent_pos[active_agents]

        # current API returns 4 values; we ignore assigned_landmark_indices
        new_hungarian, covered_agent_indices, covered_landmark_indices, _ = \
            self.get_hungarian_distances()

        d_global = -(new_hungarian.sum() / (self.env_size * self.num_agents))
        d_goal = -(new_hungarian / self.env_size)
        d_goal = torch.nan_to_num(d_goal, nan=0.0)
        progressive = torch.nan_to_num(
            (self.old_hungarian - new_hungarian) / self.v_lin_max, nan=0.0)
        # NOTE: legacy did NOT clamp progressive — keep that
        components[active_agents, 0] = progressive[active_agents]
        components[active_agents, 1] = d_goal[active_agents]
        components[active_agents, 2] = d_global

        # LEGACY comp 3: smooth reached
        components[active_agents, 3] = 1.0 / (
            1.0 + 50.0 * torch.exp(new_hungarian[active_agents] - agent_radius))
        self.covered[covered_landmark_indices] = True

        # LEGACY comp 4: agent-agent 1/(edge_dist+0.01)
        if active_agent_pos.shape[0] > 1:
            delta = active_agent_pos[:, None, :] - active_agent_pos[None, :, :]
            dist_matrix = torch.norm(delta, dim=2) - 2 * self.agent_radius
            aa_mask = (dist_matrix < self.safe_dist) & (
                ~torch.eye(dist_matrix.shape[0], device=device, dtype=torch.bool))
            inv = torch.zeros_like(dist_matrix)
            inv[aa_mask] = 1.0 / (dist_matrix[aa_mask] + 0.01)
            components[active_agents, 4] = -inv.sum(dim=1)

        # LEGACY comp 5: agent-(obstacles+done) 1/(edge_dist+0.01)
        done_agent_pos = self.agent_pos[self.dones]
        num_done = int(done_agent_pos.shape[0])
        ob_pos = torch.cat([self.obstacle_pos, done_agent_pos], dim=0)
        ob_rad = torch.cat([
            self.obstacle_radius.to(device),
            torch.full((num_done,), float(self.agent_radius), device=device)
        ], dim=0)
        ap = active_agent_pos.unsqueeze(1)
        ob = ob_pos.unsqueeze(0)
        dist_ap_ob = torch.norm(ap - ob, dim=2)
        effective = dist_ap_ob - self.agent_radius - ob_rad.unsqueeze(0)
        ob_mask = effective < self.safe_dist
        inv = torch.zeros_like(effective)
        inv[ob_mask] = 1.0 / (effective[ob_mask] + 0.01)
        components[active_agents, 5] = -inv.sum(dim=1)

        # LEGACY comp 6: -1 - (vel_lin/v_lin_max)  (no abs)
        lin_penalty = -1.0 - (self.agent_vel_lin[active_agents] / self.v_lin_max)
        components[active_agents, 6] = lin_penalty

        # LEGACY comp 7: -|vel_ang|/v_ang_max  (ang penalty, not directional)
        ang_penalty = -self.agent_vel_ang[active_agents].abs() / self.v_ang_max
        components[active_agents, 7] = ang_penalty

        # comp 8: same
        components[active_agents, 8] = -1

        self.current_rewards = components
        self.old_hungarian = new_hungarian
        self.terminate_agents(covered_agent_indices)
        return components


EP_RE = re.compile(r"Episode (\d+),")
TENSOR_RE = re.compile(r"reward components:tensor\(\[(.*?)\]\s*,\s*device", re.S)


def parse_components(path):
    with open(path) as f:
        text = f.read()
    parts = re.split(r"(Episode \d+, Mean Score)", text)
    episodes = {}
    for i in range(1, len(parts) - 1, 2):
        idx = int(re.match(r"Episode (\d+),", parts[i]).group(1))
        tm = TENSOR_RE.search(parts[i + 1])
        if not tm: continue
        rows = re.findall(r"\[([^\[\]]+)\]", tm.group(1))
        mat = []
        for r in rows:
            vals = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", r)]
            if len(vals) == 9: mat.append(vals)
        if mat: episodes[idx] = np.array(mat)
    return [episodes[i] for i in range(N_EPS) if i in episodes]


def run(n, env_cls, workdir):
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    env = env_cls(v_ang_max=math.pi / 9, agent_radius=0.5,
                  num_agents=n, num_obstacles=0)
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
        z7s = parse_components(f"/media/abz/Z7S/experiments/3-6/{n}/episode_log.txt")[:N_EPS]
        results = {}
        for label, cls in [("current", DiffDriveParallelEnvDone),
                           ("legacy",  DiffDriveParallelEnvDoneLegacy)]:
            tmp = tempfile.mkdtemp(prefix=f"cmp_{label}_n{n}_")
            try:
                results[label] = run(n, cls, tmp)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        K = min(len(z7s), *(len(v) for v in results.values()))
        z_stack = np.stack([z7s[e].mean(axis=0) for e in range(K)])
        cur_stack = np.stack([results["current"][e].mean(axis=0) for e in range(K)])
        leg_stack = np.stack([results["legacy"][e].mean(axis=0)  for e in range(K)])
        cur_mae = np.abs(cur_stack - z_stack).mean(axis=0)
        leg_mae = np.abs(leg_stack - z_stack).mean(axis=0)
        z_abs = np.abs(z_stack).mean(axis=0)
        print(f"\n========== n={n}  (mean over {K} episodes, mean over agents) ==========")
        print(f"{'comp':<12} {'|Z7S|':>9} {'cur MAE':>9} {'leg MAE':>9}   cur/Z   leg/Z   winner")
        for k in range(9):
            c, l, z = cur_mae[k], leg_mae[k], z_abs[k]
            cr = c / z if z > 0.01 else float('nan')
            lr = l / z if z > 0.01 else float('nan')
            win = "legacy" if l < c else ("current" if c < l else "tie")
            print(f"{COMP_NAMES[k]:<12} {z:>9.2f} {c:>9.2f} {l:>9.2f}   {cr:>5.2f}   {lr:>5.2f}   {win}")
        # Total scalar score = sum(SCALE * components) per ep, mean over ep
        z_score = (z_stack * SCALE).sum(axis=1).mean()
        c_score = (cur_stack * SCALE).sum(axis=1).mean()
        l_score = (leg_stack * SCALE).sum(axis=1).mean()
        print(f"  scalar score: Z7S={z_score:.1f}  current={c_score:.1f}  legacy={l_score:.1f}")
        print(f"  score err:  current={abs(c_score-z_score):.1f}  legacy={abs(l_score-z_score):.1f}")
        summary[n] = {
            "cur_mae": cur_mae.tolist(), "leg_mae": leg_mae.tolist(),
            "z_abs": z_abs.tolist(),
            "z_score": float(z_score), "cur_score": float(c_score), "leg_score": float(l_score),
        }
    with open(os.path.join(PROJECT_ROOT, ".ai", "legacy_vs_current_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

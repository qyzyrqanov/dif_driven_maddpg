"""Test multiple collision-penalty formulas against Z7S ep 0 values.
At pi/2 + seed 9832, agent trajectories are byte-matched to Z7S in ep 0.
So differences in collision components isolate the formula choice.

Records every pairwise & agent-obstacle distance per step over a single
random-actor episode, then evaluates each candidate formula on the SAME
trajectory and compares to Z7S logged ep-0 collision sums.
"""
import os, sys, math
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone
from rl.maddpg import IDDPGWithoutS

SEED = 9832
MAX_STEPS = 500
SCALE = [1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]

# Z7S ep0 collision sums (mean over agents)
Z7S = {
    4: {"agent_coll": -0.0061, "obs_coll":  0.0000},   # from the ep0 dump
    5: {"agent_coll": -0.24,   "obs_coll":  0.0000},
    6: {"agent_coll": -0.65,   "obs_coll": -0.54},
}


def record_distances(n):
    """Run 1 episode with seed 9832, random untrained actor, record per-step
    pairwise center-to-center distances and agent-obstacle/done-pos distances."""
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    env = DiffDriveParallelEnvDone(v_ang_max=math.pi/2, agent_radius=0.5,
                                   num_agents=n, num_obstacles=0)
    maddpg = IDDPGWithoutS(env, reward_scales=SCALE, batch_size=128,
                           replay_buffer_size=50000)
    state, obs = env.reset_tensor()
    pair_dists = []     # list of [N_active, N_active] tensors (center-to-center)
    done_dists = []     # list of [N_active, N_done] tensors (center-to-center)
    done_masks = []     # list of [N] bool
    for _ in range(MAX_STEPS):
        active = ~env.dones
        # Pairwise distances among active agents
        pos = env.agent_pos
        delta = pos[active][:, None, :] - pos[active][None, :, :]
        d_pair = torch.norm(delta, dim=2).detach().cpu()
        pair_dists.append(d_pair)
        # Distances from active agents to done agents (as obstacles)
        if env.dones.any():
            done_pos = pos[env.dones]
            d_done = torch.norm(pos[active][:, None, :] - done_pos[None, :, :], dim=2).detach().cpu()
        else:
            d_done = torch.zeros(int(active.sum()), 0)
        done_dists.append(d_done)
        done_masks.append(env.dones.detach().cpu().clone())
        actions = maddpg.actor.choose_action(obs, use_noise=True)
        state, obs, rewards, dones = env.step_tensor(actions)
        if dones.all().item():
            break
    return pair_dists, done_dists, done_masks


def eval_formulas(n, pair_dists, done_dists, R=0.5, SAFE=1.0, SENS=5.0):
    """Compute agent_coll & obs_coll per-episode sums (mean over agents) for
    each candidate formula. Returns dict label -> (agent_coll_mean, obs_coll_mean)."""
    out = {}
    N = pair_dists[0].shape[0]  # initial num active
    eye_initial = torch.eye(N).bool()
    # accumulate per-agent sums over the episode (only over active agents at each step)
    def accumulate(dpair_fn, dobs_fn):
        agent_sum = torch.zeros(n)
        obs_sum = torch.zeros(n)
        for t in range(len(pair_dists)):
            dp = pair_dists[t]
            na = dp.shape[0]
            if na > 1:
                mask = ~torch.eye(na).bool()
                vals = dpair_fn(dp)
                vals[~mask] = 0
                # per-row sum (per active agent)
                row_sum = vals.sum(dim=1)
                # we need to map active-row-sums back to global agents — use first na agents
                agent_sum[:na] += row_sum  # approx (active ordering preserved early on)
            dd = done_dists[t]
            if dd.shape[1] > 0:
                v = dobs_fn(dd).sum(dim=1)
                obs_sum[:dd.shape[0]] += v
        return float(agent_sum.mean()), float(obs_sum.mean())

    # 0. current: center-dist, mask < safe, -exp(-d * 7 / safe)
    def f_current_pair(d):
        m = d < SAFE
        return torch.where(m, -torch.exp(-d * 7.0 / SAFE), torch.zeros_like(d))
    def f_current_obs(d):
        m = d < SAFE
        return torch.where(m, -torch.exp(-d * 7.0 / SAFE), torch.zeros_like(d))
    out["current (center, c=7, safe)"] = accumulate(f_current_pair, f_current_obs)

    # current with c=5 and c=3
    for c in (3.0, 5.0):
        def make_fn(c=c):
            def f(d):
                m = d < SAFE
                return torch.where(m, -torch.exp(-d * c / SAFE), torch.zeros_like(d))
            return f
        out[f"current (center, c={int(c)}, safe)"] = accumulate(make_fn(), make_fn())

    # 1. center-dist, larger threshold (sens_range), -exp(-d * 7 / safe)
    def f_thresh_sens(d):
        m = d < SENS
        return torch.where(m, -torch.exp(-d * 7.0 / SAFE), torch.zeros_like(d))
    out["center, mask<SENS=5, c=7"] = accumulate(f_thresh_sens, f_thresh_sens)

    # 2. linear penalty within safe_dist
    def f_linear(d):
        m = d < SAFE
        return torch.where(m, -(1.0 - d / SAFE), torch.zeros_like(d))
    out["center, linear -(1-d/safe), mask<safe"] = accumulate(f_linear, f_linear)

    # 3. edge-to-edge clamped + exp c=7
    def f_edge_clamp(d):
        edge = (d - 2 * R).clamp(min=0.0)
        m = edge < SAFE
        return torch.where(m, -torch.exp(-edge * 7.0 / SAFE), torch.zeros_like(d))
    out["edge clamped (d-2r,>=0), c=7"] = accumulate(f_edge_clamp, f_edge_clamp)

    # 4. edge-to-edge clamped + c=5
    def f_edge_clamp5(d):
        edge = (d - 2 * R).clamp(min=0.0)
        m = edge < SAFE
        return torch.where(m, -torch.exp(-edge * 5.0 / SAFE), torch.zeros_like(d))
    out["edge clamped, c=5"] = accumulate(f_edge_clamp5, f_edge_clamp5)

    # 5. edge-to-edge clamped + c=3
    def f_edge_clamp3(d):
        edge = (d - 2 * R).clamp(min=0.0)
        m = edge < SAFE
        return torch.where(m, -torch.exp(-edge * 3.0 / SAFE), torch.zeros_like(d))
    out["edge clamped, c=3"] = accumulate(f_edge_clamp3, f_edge_clamp3)

    # 6-...  edge-to-edge UNCLAMPED + various c (overlap blows up)
    for c in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0):
        def make_fn(c=c):
            def f(d):
                edge = d - 2 * R   # can be negative on overlap
                m = edge < SAFE
                return torch.where(m, -torch.exp(-edge * c / SAFE), torch.zeros_like(d))
            return f
        out[f"edge UNCLAMPED, c={int(c)}"] = accumulate(make_fn(), make_fn())
    return out


def main():
    for n in (4, 5, 6):
        z = Z7S[n]
        print(f"\n========== n={n}  Z7S ep0: agent_coll={z['agent_coll']:.3f}  obs_coll={z['obs_coll']:.3f} ==========")
        pair_dists, done_dists, _ = record_distances(n)
        results = eval_formulas(n, pair_dists, done_dists)
        print(f"  {'formula':<42} {'agent_coll':>12} {'obs_coll':>10}   agent_err   obs_err")
        for label, (ac, oc) in results.items():
            err_a = abs(ac - z["agent_coll"])
            err_o = abs(oc - z["obs_coll"])
            print(f"  {label:<42} {ac:>12.4f} {oc:>10.4f}   {err_a:>9.4f}   {err_o:>7.4f}")


if __name__ == "__main__":
    main()

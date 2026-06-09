"""MAPPO within-environment baseline (Reviewer round-2, point #3).

A self-contained on-policy MAPPO (Yu et al., 2021) trainer for the same
``DiffDriveParallelEnvDone`` task, reward, representation and budget as the
proposed method, so it is a *fair* within-environment baseline alongside the
``maddpg_obs`` CTDE baseline. The proposed method and MADDPG are off-policy
(replay buffer + ``train_loop``); MAPPO is on-policy, so it needs its own loop
rather than a ``MADDPGBase.learn`` override.

Design / fairness notes (for the response letter):
- **Shared actor, shared centralized critic** — like the rest of the codebase.
  Actor = Gaussian policy on per-agent obs ``o_i``. Critic = CTDE value on the
  **concatenation of all per-agent obs in index order** (same head_i<->agent_i
  binding as ``MADDPGSharedActorCriticIndependentObs``), outputs per-agent values.
- **Per-agent rewards** (the env's scalarized 9-component reward), per-agent GAE.
- **Terminated agents:** an agent that reaches its landmark is masked out from
  the step it becomes done onward (mirrors the ``is_valid`` handling in
  ``IDDPGWithoutS``); its terminal reaching step is kept with no bootstrap, and a
  truncated (still-active at horizon) agent bootstraps with V(last obs).
- **No orbit-restart, no HER offline relabel** — those are our pipeline, not the
  baseline. Same env / full reward / 1000-episode budget / pi/2 / no obstacles.
- Reuses ``MADDPGBase`` only for ``log_env_step_to_csv`` (identical per-step CSV
  -> ``result*.csv`` -> ``episode_summary.csv`` for the notebook), ``reward_sum``
  (9-component -> scalar), ``_save_meta_json`` and ``update_params_vectorized``.

Reported metric is the last-200 training-window SR/coverage WITH the policy's
on-policy stochasticity (PPO samples actions during rollouts) — the same
"with-noise" protocol as the main table and the MADDPG baseline.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Callable, Optional

import numpy as np
import torch

from config import device as DEFAULT_DEVICE, gamma as GAMMA
from models.mappo_nets import GaussianActor, CentralizedValue
from rl.maddpg import MADDPGBase


class ValueNorm:
    """Running mean/std normalizer for value targets (standard MAPPO trick).

    The critic predicts normalized values; GAE/returns are computed in real
    reward units (denormalized), and the critic is regressed against normalized
    returns. Stabilizes learning when episode returns span hundreds of units.
    """

    def __init__(self, device):
        self.device = device
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = 1e-4

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        bmean = x.mean()
        bvar = x.var(unbiased=False)
        bcount = x.numel()
        delta = bmean - self.mean
        tot = self.count + bcount
        self.mean = self.mean + delta * bcount / tot
        m_a = self.var * self.count
        m_b = bvar * bcount
        self.var = (m_a + m_b + delta.pow(2) * self.count * bcount / tot) / tot
        self.count = tot

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self.var + 1e-8) + self.mean

    def state_dict(self) -> dict:
        return {"mean": float(self.mean), "var": float(self.var), "count": float(self.count)}

    def load_state_dict(self, sd: dict) -> None:
        self.mean = torch.tensor(sd["mean"], device=self.device)
        self.var = torch.tensor(sd["var"], device=self.device)
        self.count = sd["count"]


class MAPPO(MADDPGBase):
    """On-policy MAPPO baseline. Drop-in for ``train_seeded.py --algorithm mappo``."""

    def __init__(
        self,
        env,
        reward_scales,
        device=DEFAULT_DEVICE,
        replay_buffer_size: int = 50000,   # accepted for launcher compat; unused
        batch_size: int = 128,             # accepted for launcher compat; PPO minibatches below
        use_tagged_replay_buffer: bool = False,
        # --- PPO hyperparameters (disclosed baseline config) ---
        rollout_episodes: int = 10,
        ppo_epochs: int = 10,
        num_minibatches: int = 4,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
    ) -> None:
        # NOTE: intentionally do NOT call MADDPGBase.__init__ — it allocates a
        # tagged replay buffer we never use. Set only what the reused helpers need.
        self.env = env
        self.device = device
        self.obs_dim = env.obs_dim
        self.state_dim = env.state_dim
        self.num_agents = env.num_agents
        self.action_dim = env.action_dim
        self.use_tagged_replay_buffer = False
        self.replay_buffer = None  # _save_meta_json tolerates None via getattr
        self.reward_scales = torch.tensor(
            reward_scales, dtype=torch.float32, device=self.device
        )

        # PPO config
        self.gamma = GAMMA
        self.rollout_episodes = rollout_episodes
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Networks (shared actor + centralized concat-obs value)
        self.actor = GaussianActor(
            self.obs_dim, self.action_dim, device=self.device,
            chckpnt_file="mappo_actor.pth",
        )
        self.critic = CentralizedValue(
            self.num_agents * self.obs_dim, self.num_agents, device=self.device,
            chckpnt_file="mappo_critic.pth",
        )
        self.max_action = self.actor.max_action
        self.value_norm = ValueNorm(self.device)

        # Bookkeeping (mirrors MADDPG attrs used by save/plot/resume).
        self.score_history = []
        self.actor_losses = []
        self.critic_losses = []

    # ----- abstract-method implementations (MADDPGBase is an ABC) -----
    def learn(self, buffer=None):
        raise NotImplementedError("MAPPO is on-policy; use the train_loop PPO update.")

    def load_actor(self):
        self.actor.load_checkpoint()

    def choose_actions(self, obs_list, use_noise=True):
        """Match the MADDPG interface (used by eval/try_actor with use_noise=False)."""
        if not torch.is_tensor(obs_list):
            obs_list = torch.tensor(obs_list, dtype=torch.float32)
        raw, _ = self.actor.act(obs_list.to(self.device), deterministic=not use_noise)
        return raw.clamp(-self.max_action, self.max_action)

    def save_checkpoint(self, file_pref=None):
        self.actor.save_checkpoint(file_prefix=file_pref)
        self.critic.save_checkpoint(file_prefix=file_pref)

    def load_checkpoint(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    # ----------------------------- rollout -----------------------------
    def _run_episode(self, episode_id: int, total_steps: int, max_steps: int):
        """Run one episode, log per-step CSV, return collected transitions + new total_steps."""
        env = self.env
        N = self.num_agents
        state, obs = env.reset_tensor()
        done = torch.zeros(N, dtype=torch.bool, device=self.device)

        obs_t, act_t, logp_t, val_t, rew_t, active_t, doneaf_t, jobs_t = (
            [], [], [], [], [], [], [], []
        )

        j = 0
        while j < max_steps and not done.all():
            active = (~done).clone()                       # active at action time
            joint_obs = obs.reshape(1, N * self.obs_dim)
            with torch.no_grad():
                raw, logprob = self.actor.act(obs, deterministic=False)   # [N,act],[N]
                # critic predicts normalized values; GAE works in real units.
                values = self.value_norm.denormalize(self.critic(joint_obs).squeeze(0))  # [N]
            env_action = raw.clamp(-self.max_action, self.max_action)
            next_state, next_obs, rewards, next_done = env.step_tensor(env_action)
            reward_scalar = self.reward_sum(env.current_rewards)           # [N]

            obs_t.append(obs)
            act_t.append(raw)
            logp_t.append(logprob)
            val_t.append(values)
            rew_t.append(reward_scalar)
            active_t.append(active)
            doneaf_t.append(next_done.clone())
            jobs_t.append(joint_obs.squeeze(0))

            self.log_env_step_to_csv(total_step=total_steps, episode_id=episode_id)

            done = next_done.clone().detach()
            obs = next_obs
            total_steps += 1
            j += 1

        # Bootstrap value for agents truncated at the horizon (still active).
        with torch.no_grad():
            boot_values = self.value_norm.denormalize(
                self.critic(obs.reshape(1, N * self.obs_dim)).squeeze(0))  # [N]

        traj = dict(
            obs=torch.stack(obs_t),          # [T,N,obs_dim]
            action=torch.stack(act_t),       # [T,N,act_dim]
            logprob=torch.stack(logp_t),     # [T,N]
            value=torch.stack(val_t),        # [T,N]
            reward=torch.stack(rew_t),       # [T,N]
            active=torch.stack(active_t),    # [T,N] bool
            done_after=torch.stack(doneaf_t),# [T,N] bool
            joint_obs=torch.stack(jobs_t),   # [T,N*obs_dim]
            boot_value=boot_values,          # [N]
        )
        return traj, total_steps

    def _gae_for_trajectory(self, traj):
        """Per-agent GAE over each agent's contiguous active prefix.

        Yields flattened samples: (joint_obs, agent_idx, obs_i, action_i,
        old_logprob, return, advantage).
        """
        T, N = traj["reward"].shape
        samples = {k: [] for k in
                   ("joint_obs", "agent_idx", "obs", "action", "logprob", "ret", "adv")}
        gamma, lam = self.gamma, self.gae_lambda

        for i in range(N):
            active_i = traj["active"][:, i]
            if not bool(active_i.any()):
                continue
            k = int(torch.nonzero(active_i).max().item())   # last active step
            terminal = bool(traj["done_after"][k, i])        # reached goal at k?

            r = traj["reward"][:k + 1, i]
            v = traj["value"][:k + 1, i]
            boot = torch.zeros((), device=self.device) if terminal else traj["boot_value"][i]

            adv = torch.zeros(k + 1, device=self.device)
            gae = torch.zeros((), device=self.device)
            for t in range(k, -1, -1):
                if t == k:
                    next_v = boot
                    nonterminal = 0.0 if terminal else 1.0
                else:
                    next_v = v[t + 1]
                    nonterminal = 1.0
                delta = r[t] + gamma * next_v * nonterminal - v[t]
                gae = delta + gamma * lam * nonterminal * gae
                adv[t] = gae
            ret = adv + v

            samples["joint_obs"].append(traj["joint_obs"][:k + 1])         # [k+1, N*obs]
            samples["agent_idx"].append(torch.full((k + 1,), i, dtype=torch.long,
                                                    device=self.device))
            samples["obs"].append(traj["obs"][:k + 1, i])                  # [k+1, obs]
            samples["action"].append(traj["action"][:k + 1, i])           # [k+1, act]
            samples["logprob"].append(traj["logprob"][:k + 1, i])         # [k+1]
            samples["ret"].append(ret)
            samples["adv"].append(adv)
        return samples

    def _ppo_update(self, batch):
        """PPO clipped-surrogate update over the collected on-policy batch."""
        joint_obs = batch["joint_obs"]      # [B, N*obs]
        agent_idx = batch["agent_idx"]      # [B]
        obs = batch["obs"]                  # [B, obs]
        action = batch["action"]            # [B, act]
        old_logprob = batch["logprob"]      # [B]
        returns = batch["ret"]              # [B]
        adv = batch["adv"]                  # [B]

        # Update value-target stats on real-scale returns, train critic in
        # normalized space (standard MAPPO value normalization).
        self.value_norm.update(returns.detach())
        returns_norm = self.value_norm.normalize(returns)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        B = obs.shape[0]
        mb_size = max(1, B // self.num_minibatches)

        a_losses, c_losses = [], []
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, mb_size):
                idx = perm[start:start + mb_size]
                new_logprob, entropy = self.actor.evaluate_actions(obs[idx], action[idx])
                ratio = (new_logprob - old_logprob[idx]).exp()
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = entropy.mean()

                values_all = self.critic(joint_obs[idx])               # [mb, N] (normalized)
                value_pred = values_all.gather(1, agent_idx[idx].unsqueeze(1)).squeeze(1)
                value_loss = (value_pred - returns_norm[idx]).pow(2).mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy_loss)

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                a_losses.append(float(policy_loss.detach()))
                c_losses.append(float(value_loss.detach()))
        return (float(np.mean(c_losses)) if c_losses else None,
                float(np.mean(a_losses)) if a_losses else None)

    # ----------------------------- main loop -----------------------------
    def train_loop(
        self,
        n_games: int = 1000,
        max_steps: int = 500,
        checkpoint_path: str = "training_state.pkl",
        meta_extra: Optional[dict] = None,
        meta_path: str = "meta.json",
        post_episode_callback: Optional[Callable[[int, bool], None]] = None,
        **_ignored,   # swallow DDPG-only kwargs the launcher passes
    ) -> None:
        # ----- resume -----
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                sd = pickle.load(f)
            start_episode = sd["episode"]
            self.score_history = sd.get("score_history", [])
            total_steps = sd.get("total_steps", 0)
            total_train_seconds = float(sd.get("total_train_seconds", 0.0))
            peak_gpu_bytes = int(sd.get("peak_gpu_bytes", 0))
            if sd.get("value_norm") is not None:
                self.value_norm.load_state_dict(sd["value_norm"])
            self.load_checkpoint()
            print(f"Resuming MAPPO from episode {start_episode}", flush=True)
        else:
            start_episode = 0
            self.score_history = []
            total_steps = 0
            total_train_seconds = 0.0
            peak_gpu_bytes = 0

        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        rollout = None  # accumulator across rollout_episodes

        def save_state(ep_completed, finished):
            sd = {
                "episode": ep_completed,
                "score_history": self.score_history,
                "total_steps": total_steps,
                "total_train_seconds": total_train_seconds,
                "peak_gpu_bytes": peak_gpu_bytes,
                "value_norm": self.value_norm.state_dict(),
            }
            with open(checkpoint_path, "wb") as f:
                pickle.dump(sd, f)
            self.save_checkpoint()
            self._save_meta_json(
                meta_path=meta_path, meta_extra=meta_extra, n_games_target=n_games,
                episodes_completed=ep_completed, total_steps=total_steps,
                total_train_seconds=total_train_seconds, peak_gpu_bytes=peak_gpu_bytes,
                finished=finished,
            )
            if post_episode_callback is not None:
                try:
                    post_episode_callback(ep_completed, finished)
                except Exception as exc:
                    print(f"post_episode_callback failed: {exc!r}", flush=True)

        episodes_in_rollout = 0
        for i in range(start_episode, n_games):
            t0 = time.perf_counter()
            traj, total_steps = self._run_episode(i, total_steps, max_steps)

            # accumulate this episode's GAE samples
            ep_samples = self._gae_for_trajectory(traj)
            if rollout is None:
                rollout = {k: [] for k in ep_samples}
            for k in rollout:
                rollout[k].extend(ep_samples[k])
            episodes_in_rollout += 1

            score = self.reward_from_rb(self.env.current_score).mean().cpu().item()
            self.score_history.append(score)
            done_count = int(self.env.dones.sum().item())
            print(f"[MAPPO] ep {i} score {score:.2f} done {done_count}/{self.num_agents} "
                  f"steps {total_steps}", flush=True)

            # PPO update at the end of each rollout window (and the final episode).
            is_last = (i == n_games - 1)
            if episodes_in_rollout >= self.rollout_episodes or is_last:
                batch = {k: torch.cat(v, dim=0) for k, v in rollout.items()}
                c_loss, a_loss = self._ppo_update(batch)
                self.critic_losses.append(c_loss)
                self.actor_losses.append(a_loss)
                print(f"[MAPPO] update @ep{i}: policy_loss {a_loss}, value_loss {c_loss}",
                      flush=True)
                rollout = None
                episodes_in_rollout = 0

            total_train_seconds += time.perf_counter() - t0
            if torch.cuda.is_available():
                try:
                    peak_gpu_bytes = max(peak_gpu_bytes, int(torch.cuda.max_memory_allocated()))
                except Exception:
                    pass

            if i > 0:
                save_state(i + 1, finished=is_last)

        save_state(n_games, finished=True)
        print("MAPPO training complete.", flush=True)

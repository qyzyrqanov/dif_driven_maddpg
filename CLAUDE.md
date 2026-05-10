# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Multi-agent reinforcement learning research codebase for cooperative coverage of landmarks by differential-drive robots. Multiple MADDPG/IDDPG variants train shared actor/critic networks against a custom PettingZoo `ParallelEnv` whose reward signal is shaped by a Hungarian assignment between agents and landmarks. Used for paper experiments and revision-driven ablations (see `revision_runs_and_results_plan.md`).

## Running training and evaluation

The project root must be on `PYTHONPATH` because absolute package imports like `from custom_envs.diff_driven...` and `from rl.maddpg import ...` are used everywhere. `run/*.py` scripts must be executed from the **directory where you want artifacts written**, not from `run/` — every script writes checkpoints, replay buffers, plots, and CSVs to the current working directory using bare filenames (e.g., `replay_buffer.pkl`, `training_state.pkl`, `simple_actor.pth`, `learning_curve_episode_*.png`).

Linux:
```bash
source .venv/bin/activate          # or .venvLin
export PYTHONPATH=$(pwd)
python run/train_done4.py          # or any other run/train_*.py
```

Windows shell scripts are in `run/run_train_*.ps1` (use `.venv3.10`); a couple of Linux equivalents live in `run/run_done5_ablation.sh` / `run/run_done6_ablation.sh`. Note these use hardcoded absolute paths (`/home/abz/PycharmProjects/...`) that may need updating.

There are no tests, lint, or build steps. Validation happens by inspecting `learning_curve_episode_*.png`, `actor_loss_episode_*.png`, `critic_loss_episode_*.png`, `trajectory_episode_*.png`, and `episode_log.txt` produced into the working directory.

## High-level architecture

### Three-layer design

1. **Environment** (`custom_envs/diff_driven/gym_env/centered_paralelenv/env.py`) — pure-tensor PettingZoo `ParallelEnv` on CUDA.
2. **Algorithms** (`rl/maddpg.py`) — abstract `MADDPGBase` plus several concrete variants sharing one `train_loop` / `main_loop`.
3. **Models** (`models/`) — `SimpleActor`, `SharedCritic`, `MultiheadCritic`.

`config.py` is the global default registry: `from config import *` is used inside the env, models, and algorithms. Dimension formulas (`obs_dim`, `state_dim`) are computed inside the env constructors based on `num_agents`, `num_obstacles`, and the `normalise` flag — they are **not** taken from `config.py`. Keep this in mind when changing env shape.

### Environment class hierarchy

All in `env.py`:
- `DiffDriveParallelEnv` — base. Continuous (dv_lin, dv_ang) actions; reward is a 9-component vector per agent.
- `DiffDriveParallelEnvAdj` — adds a smoothness/adjacency variant.
- `DiffDriveParallelEnvDone` — adds per-agent `dones` and per-landmark `covered` flags. Agents that reach a landmark are terminated (zeroed velocity, masked from Hungarian). Most current training uses this variant.
- `DiffDriveParallelEnvAssignFirstDone` — pre-assigns a landmark per agent and includes only the assigned one in observations.
- `DiffDriveParallelEnvDoneAdj` / `DiffDriveParallelEnvAssignFirstDoneAdj` — diamond-inheritance combinations.

Step protocol used by training:
```python
state, obs = env.reset_tensor()
state, obs, rewards, dones = env.step_tensor(actions_tensor)   # rewards: [N, 9]
```
`env.current_rewards` is `[N, 9]` (the 9 components). Algorithms convert this to scalar rewards by dot-product with `reward_scales`.

### The 9 reward components (fixed order)

Index | Component | Notes
:--:|---|---
0 | progressive | Δ Hungarian distance / v_lin_max
1 | distance (d_goal) | −per-agent assigned distance / env_size
2 | base / d_global | shared global Hungarian penalty
3 | reached goal | landmark-coverage bonus
4 | agent–agent collision | exponential proximity penalty (active agents)
5 | obstacle / done-agent collision | done agents act as static obstacles
6 | v_lin | linear-velocity shaping
7 | v_ang / directional | angular penalty (base) or cos-similarity to assigned landmark (`Done` variant uses index 7 for direction)
8 | time | per-step −1 for active agents

Every `run/train_*.py` defines a `scale = [...]` list of 9 floats matching this order. Algorithms compute `(components * reward_scales).sum(-1)`.

### Algorithm variants in `rl/maddpg.py`

`MADDPGBase` — abstract; owns the replay buffer, reward-component → scalar conversion (`reward_sum`), checkpoint plumbing, plotting, CSV logging (`log_env_step_to_csv`), and two near-identical loops (`main_loop` and `train_loop`; `main_loop` additionally supports an offline-replay augmentation when too few "tagged" transitions have been collected).

Concrete subclasses differ only in `_critic_dim()` and `learn()`:

- `MADDPGSharedActorCritic` — joint-action centralized critic outputting per-agent Q-values; one centralized critic update.
- `MADDPGSharedActorCriticIndependentQmean` — per-agent loop inserting differentiable agent-i action into joint, training critic per agent.
- `MADDPGSharedActorCriticIndependent` — accumulates per-agent actor losses then a single backward pass; freezes critic params during actor backward.
- `IDDPG` — critic takes `(state, single agent action)`, output dim 1.
- `IDDPGWithoutS` — fully independent: critic takes `(obs_i, action_i)` (no global state).

All variants use a **shared** actor and **shared** critic (single network applied to all agents); the differences are about what gets fed in and how losses are aggregated. Soft target updates use `update_params_vectorized` with `tau` from `config.py`.

### Replay buffer split

`tagged_replay_buffer.py` (default, `use_tagged_replay_buffer=True`) reserves 25% of capacity for "tagged" transitions — those where at least one agent transitioned from not-done to done. Sampling guarantees up to 25% tagged samples per batch when available. This stabilizes training when goal-reaching events are rare. The plain `replay_buffer.py` is used when the flag is off; it stores already-summed scalar rewards instead of the 9-component vector.

When tagged buffer is on, the buffer stores `[N, 9]` reward components and the algorithm's `reward_scales` can be changed mid-training (via the `rescale_env_rewards` callback in `train_loop`) without invalidating stored data.

### Offline replay augmentation (`main_loop` only)

When `total_tagged < 0.25 * batch_size`, after an episode the loop records initial agent state + the action sequence, then replays it in a copy of the env with **landmarks rotated to a PCA-aligned frame at the trajectory tail** (`offline_replay_success`). The replayed transitions are added to the buffer as additional training data. `train_loop` does not do this.

### Checkpointing

Each `run/train_*.py` script writes its artifacts (replay buffer, training-state pickle, `*.pth` model checkpoints, plots, `episode_log.txt`, episode-step CSV) to its CWD. Checkpoint files have hardcoded names (`shared_actor.pth`, `shared_critic.pth`, `shared_actor_target.pth`, `shared_critic_target.pth`, `replay_buffer.pkl`, `training_state.pkl`). Restarting the script from the same directory automatically resumes from the saved episode index.

The `experiments<seed>/` directories (`experiments0`, `experiments13`, `experiments41`, `experiments9832`) are completed run outputs — do not commit new ones casually.

## Conventions

- All math runs on `device` from `config.py` (CUDA if available); avoid moving tensors to CPU except for plotting, Hungarian assignment (uses `scipy.optimize.linear_sum_assignment` on CPU), or numpy interop.
- The env stores `agent_pos`, `agent_dir`, `agent_vel_lin`, `agent_vel_ang` directly as tensors. `env.copy()` clones them; `env.delete()` nulls them to free GPU memory before `del env; torch.cuda.empty_cache()`.
- New algorithm variants: subclass `MADDPGSharedActorCritic` and override `_critic_dim()` + `learn()`. Use `self.reward_from_rb(reward_components)` to get scalar rewards from a sampled batch — this handles the tagged-vs-untagged buffer difference.
- New env reward variants: override `_compute_rewards_tensor()` and write into `components[active_agents, k]` for the right indices, then set `self.current_rewards = components` and `self.old_hungarian = new_hungarian`. Do not change the order/length of the 9 components without updating every `scale=[...]` list in `run/train_*.py`.

## External vendored packages

The repo contains full source trees of `MARLlib/`, `Multi-Agent-Deep-Deterministic-Policy-Gradients/`, `gym/`, and `gym-0.20.0/`, all listed in `.gitignore`. They are reference implementations / bundled installs, not part of the active codebase — don't edit them.

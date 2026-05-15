import time
import json
import socket
import platform

import torch
from pettingzoo import ParallelEnv


from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnv, DiffDriveParallelEnvDone
from models.simpleactor import SimpleActor
from models.simplecritic import SharedCritic
import tagged_replay_buffer, replay_buffer
from abc import ABC, abstractmethod
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from config import *
from typing import Union, Optional, Tuple, Any, Callable, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pandas as pd


class MADDPGBase(ABC):
    def __init__(
            self,
            env: DiffDriveParallelEnv,
            reward_scales:Sequence[float],
            device: Union[str, torch.device] = device,
            replay_buffer_size: int = replay_buffer_size,
            batch_size: int = batch_size,
            use_tagged_replay_buffer: bool = True,
    ) -> None:
        """
        Initializes the base MADDPG class.

        Args:
            env (DiffDriveParallelEnv): The multi-agent environment.
            device (str | torch.device): Target device for computation (e.g., 'cuda' or 'cpu').
            replay_buffer_size (int): Maximum capacity of the replay buffer.

        Sets:
            self.env: The environment instance.
            self.replay_buffer: A ReplayBuffer instance.
            self.obs_dim: Observation dimension.
            self.state_dim: Global state dimension.
            self.device: Device used for all models and data.
        """
        self.env=env
        self.use_tagged_replay_buffer=use_tagged_replay_buffer
        if use_tagged_replay_buffer:
            self.replay_buffer=tagged_replay_buffer.ReplayBuffer(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=env.action_dim,
                                        device=device, num_agents=env.num_agents, replay_buffer_size=replay_buffer_size, batch_size=batch_size)
        else:
            self.replay_buffer = replay_buffer.ReplayBuffer(obs_dim=env.obs_dim, state_dim=env.state_dim,
                                                                   action_dim=env.action_dim,
                                                                   device=device, num_agents=env.num_agents,
                                                                   replay_buffer_size=replay_buffer_size,
                                                                   batch_size=batch_size)
        self.obs_dim=env.obs_dim
        self.state_dim=env.state_dim

        self.device=device
        self.reward_scales = torch.tensor(reward_scales, dtype=torch.float32, device=self.device)

    @staticmethod
    @torch.no_grad()
    def update_params_vectorized(
            network: torch.nn.Module,
            target_network: torch.nn.Module,
            tau: float
    ) -> None:
        """
        Performs a soft update of target network parameters:
            θ_target ← τ * θ_online + (1 - τ) * θ_target

        Args:
            network (torch.nn.Module): Source network whose parameters are used in update.
            target_network (torch.nn.Module): Target network to be softly updated.
            tau (float): Soft update coefficient in [0, 1].

        Notes:
            Assumes both networks:
            - Have identical architectures,
            - Reside on the same device.
        """
        vec_net = parameters_to_vector(network.parameters())
        vec_target = parameters_to_vector(target_network.parameters())
        updated = tau * vec_net + (1.0 - tau) * vec_target
        vector_to_parameters(updated, target_network.parameters())

    @abstractmethod
    def learn(self, buffer = None):
        pass

    @abstractmethod
    def load_actor(self):
        pass

    @abstractmethod
    def choose_actions(self, obs_list, use_noise=True):
        pass

    @abstractmethod
    def save_checkpoint(self, file_pref=None):
        pass
    @abstractmethod
    def load_checkpoint(self):
        pass

    def log_env_step_to_csv(
            self,
            total_step: int,
            episode_id: int,
            csv_path: str = "rewards.csv"
    ) -> None:
        """
        Append one row from self.env to CSV:
          - total_step (arg)
          - episode_id (arg)
          - timestep (from env)
          - done_count (from env.dones)
          - hung_dist_agent{i} (N cols)
          - agent{i}_vel_lin (N cols)
          - agent{i}_vel_ang (N cols)
          - weighted reward components: agent{i}_comp{j} (9N cols)
        """
        env = self.env
        N = env._num_agents

        # --- Checks ---
        if not hasattr(self, "reward_scales"):
            raise RuntimeError("self.reward_scales is required (length 9).")
        if not hasattr(env, "current_rewards"):
            raise RuntimeError("env.current_rewards missing. Call env.step/step_tensor first.")
        if not hasattr(env, "old_hungarian"):
            raise RuntimeError("env.old_hungarian missing. Make sure _reset_hungarian ran and a step was taken.")
        if not hasattr(env, "dones"):
            raise RuntimeError("env.dones missing. Ensure environment is initialized.")

        # --- Validate shapes ---
        comps = env.current_rewards  # (N, 9)
        if comps.shape != (N, 9):
            raise ValueError(f"env.current_rewards must be shape ({N}, 9), got {comps.shape}")
        hung = env.old_hungarian
        if hung.shape != (N,):
            raise ValueError(f"env.old_hungarian must be shape ({N},), got {hung.shape}")
        vel_lin = env.agent_vel_lin
        vel_ang = env.agent_vel_ang
        if vel_lin.shape != (N,) or vel_ang.shape != (N,):
            raise ValueError("agent_vel_lin and agent_vel_ang must each be shape (N,)")

        # --- Scale reward components ---
        scales = torch.as_tensor(self.reward_scales, device=comps.device, dtype=comps.dtype)
        if scales.numel() != 9:
            raise ValueError(f"reward_scales must have 9 elements, got {scales.numel()}")
        weighted = comps * scales  # (N, 9)

        # --- Flatten reward components to (1, 9N) ---
        flat_row = weighted.reshape(1, -1).detach().float().cpu().numpy()  # (1, 9N)

        # --- Column names ---
        hung_cols = [f"hung_dist_agent{i}" for i in range(N)]
        vel_lin_cols = [f"agent{i}_vel_lin" for i in range(N)]
        vel_ang_cols = [f"agent{i}_vel_ang" for i in range(N)]
        comp_cols = [f"agent{i}_comp{j}" for i in range(N) for j in range(1, 10)]

        # --- Scalar values ---
        timestep_val = int(env.timestep)
        done_count_val = int(env.dones.sum().item())

        # --- Build row dict ---
        row = {
            "total_step": [total_step],
            "episode_id": [episode_id],
            "timestep": [timestep_val],
            "done_count": [done_count_val],
        }

        # Hungarian distances
        hung_np = hung.detach().float().cpu().numpy().flatten().tolist()
        for name, val in zip(hung_cols, hung_np):
            row[name] = [val]

        # Linear velocities
        vel_lin_np = vel_lin.detach().float().cpu().numpy().flatten().tolist()
        for name, val in zip(vel_lin_cols, vel_lin_np):
            row[name] = [val]

        # Angular velocities
        vel_ang_np = vel_ang.detach().float().cpu().numpy().flatten().tolist()
        for name, val in zip(vel_ang_cols, vel_ang_np):
            row[name] = [val]

        # Weighted reward components
        for name, val in zip(comp_cols, flat_row.flatten().tolist()):
            row[name] = [val]

        # --- Append to CSV ---
        df_row = pd.DataFrame(row)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        write_header = not os.path.exists(csv_path)
        df_row.to_csv(csv_path, mode="a", header=write_header, index=False)
    def main_loop(
            self,
            n_games: int = n_games,
            train_each: int = train_each,
            evaluate: bool = False,
            checkpoint_path: str = "training_state.pkl",
            patience=patience,
            score_avg_window=score_avg_window,
            max_steps=max_steps,
            start_training_after=start_training_after,
            rescale_env_rewards=None,
            min_episodes_before_early_stop: int = min_episodes_before_early_stop,
    ) -> None:
        """
        Resumable training loop with checkpointing.

        Args:
            n_games (int): Total episodes to run.
            train_each (int): Frequency to trigger learning.
            evaluate (bool): If True, only evaluates.
            checkpoint_path (str): Path to save/load training state.
        """

        # === Load previous training state if exists ===
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                state_dict = pickle.load(f)
            start_episode = state_dict["episode"]
            self.score_history = state_dict["score_history"]
            best_score = state_dict["best_score"]
            self.replay_buffer.load("replay_buffer.pkl")
            episodes_without_improvement = state_dict["episodes_without_improvement"]
            total_steps = state_dict['total_steps']
            self.actor_losses = state_dict['actor_losses']
            self.critic_losses = state_dict['critic_losses']
            self.load_checkpoint()
            if 'total_tagged' in state_dict:
                total_tagged = state_dict['total_tagged']
            else:
                total_tagged = total_steps

            print(f"Resuming from episode {start_episode}")
        else:
            start_episode = 0
            self.score_history = []
            best_score = -float("inf")
            episodes_without_improvement = 0
            total_steps = 0
            self.actor_losses = []
            self.critic_losses = []
            total_tagged = 0

        for i in range(start_episode, n_games):

            offline_necessary= i>0 and total_tagged<self.replay_buffer.batch_size*0.25

            state, obs = self.env.reset_tensor()
            done = torch.full((self.env.num_agents,), False, dtype=torch.bool, device=self.device)
            episode_trajectory = [self.env.agent_pos.cpu().clone().detach().numpy()]

            if offline_necessary:
                init_agent_pos =  self.env.agent_pos.clone().detach().to(device=device)
                init_agent_dir = self.env.agent_dir.clone().detach().to(device=device)
                # last_agent_pos = self.env.agent_pos
                init_vel_lin=self.env.agent_vel_lin.clone().detach().to(device=device)
                init_vel_ang=self.env.agent_vel_ang.clone().detach().to(device=device)

                episode_actions = []
            j = 0
            tagged_count=0
            while j < max_steps and not done.all():
                # print(f'step: {total_steps}')
                if evaluate:
                    self.env.render()
                    time.sleep(0.1)
                # print(f'chosing action by obs')
                actions = self.choose_actions(obs)
                # print(f'taking actions')
                if offline_necessary:
                    episode_actions.append(actions.clone().detach().to(device=device))
                next_state, next_obs, rewards, next_done = self.env.step_tensor(actions)

                tagged = ((~done) & next_done).any().item()
                if tagged:
                    print(f'tagged on step:{j}/{total_steps}:\ndone: {done}\n next_done: {next_done}')
                    tagged_count += 1
                done = next_done.clone().detach().to(device=device)

                # print(f'saving rb')
                self.add_record_to_rb(state, obs, actions, rewards, next_state, next_obs, done, tagged=tagged)

                state = next_state
                obs = next_obs





                if not evaluate and total_steps % train_each == 0:
                    if total_steps < start_training_after:
                        print(f'replay buffer not ready: {total_steps}/{start_training_after}')
                    else:
                        if rescale_env_rewards != None:
                            print('rescaling')
                            self.reward_scales = torch.tensor(rescale_env_rewards(total_steps), dtype=torch.float32,
                                                              device=self.device)
                        critic_loss, actor_loss = self.learn()
                        self.actor_losses.append(actor_loss)
                        self.critic_losses.append(critic_loss)
                        print(f"Trained at step {total_steps}, actor loss: {actor_loss}, critic loss : {critic_loss}")
                episode_trajectory.append(self.env.agent_pos.cpu().clone().detach().numpy())
                self.log_env_step_to_csv(total_step=total_steps, episode_id=i)
                j = j + 1
                total_steps += 1
            scores= self.reward_from_rb(self.env.current_score)
            score = scores.mean().cpu().item()
            self.score_history.append(score)
            avg_score = torch.tensor(self.score_history[-score_avg_window:], device='cpu').float().mean().item()
            total_tagged += tagged_count

            print(f"Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Tagged count: {tagged_count}")
            self.log_episode(episode_id=f'{i}')
            if not evaluate and avg_score > best_score and i > min_episodes_before_early_stop:
                best_score = avg_score
                # self.save_checkpoint(file_pref=f'best_{i}_{best_score}_{i}')
                print("Checkpoint saved (best model)")
                episodes_without_improvement = 0
            else:
                if i > min_episodes_before_early_stop:
                    episodes_without_improvement += 1

            # Save training state every 5 episodes
            if i > 0:
                print("Training progress saving.")
                state_dict = {
                    "episode": i + 1,
                    "score_history": self.score_history,
                    "best_score": best_score,
                    'episodes_without_improvement': episodes_without_improvement,
                    'total_steps': total_steps,
                    'actor_losses': self.actor_losses,
                    'critic_losses': self.critic_losses

                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state_dict, f)

                self.replay_buffer.save("replay_buffer.pkl")

                self.save_checkpoint()

                self.plot_learning_curve()
                self.plot_actor_loss()
                self.plot_critic_loss()
                # self.save_actor(f'simple_actor_{i}.pth')
                self.plot_episode_gone_trajectory(np.stack(episode_trajectory), f'{i}_tagged_{tagged_count}_score_{score}')

                print("Training progress saved.")
                if i % 10 == 0:
                    # self.save_checkpoint(file_pref=f'episode_{i}_')
                    # self.replay_buffer.save(f"replay_buffer_{i}.pkl")
                    with open(f'episode_{i}_{checkpoint_path}', "wb") as f:
                        pickle.dump(state_dict, f)
                    self.plot_episode_new_trajectory(episode=f'{i}_1')
                    self.log_episode(episode_id=f'{i}_1')
                    self.plot_episode_new_trajectory(episode=f'{i}_2')
                    self.log_episode(episode_id=f'{i}_2')

                    self.plot_episode_new_trajectory(episode=f'{i}_3')
                    self.log_episode(episode_id=f'{i}_3')


                    print("plotted")
            if episodes_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered: No improvement in {patience} episodes.")
                print("Training progress saving.")
                state_dict = {
                    "episode": i + 1,
                    "score_history": self.score_history,
                    "best_score": best_score,
                    'episodes_without_improvement': episodes_without_improvement,
                    'total_steps': total_steps,
                    'actor_losses': self.actor_losses,
                    'critic_losses': self.critic_losses

                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state_dict, f)

                self.replay_buffer.save("replay_buffer.pkl")
                self.save_checkpoint()
                print("Training progress saved.")

                self.plot_learning_curve()
                self.plot_actor_loss()
                self.plot_critic_loss()
                self.plot_episode_gone_trajectory(np.stack(episode_trajectory), episode=f'{i}_tagged_{tagged_count}_score_{score}')
                break
            if offline_necessary and tagged_count<3:
                # self.plot_episode_gone_trajectory(np.stack(episode_trajectory), episode=f'{i}_score_{score}')
                num_samples = self.env.num_landmarks
                base = len(episode_trajectory) - (num_samples+1)* 3 # get last n*3 steps
                # assert base >= 0, "episode_trajectory too short"


                # Store last positions into 3 groups of landmark s
                rand_list = base+torch.randperm(num_samples)*3

                print(f'replaying episode {i}:rnd_list: {rand_list}')

                # For each of the 3 sets
                last_positions=torch.zeros((self.env.num_landmarks, 2), device=self.device)
                for l in range(self.env.num_landmarks):
                        idx = rand_list[l].item()
                        pos_np = episode_trajectory[idx][l]
                        last_positions[l, :] = torch.tensor(pos_np, device=self.device)
                        print(f'last_positions_list[{l}, :] = episode_trajectory[{idx}][{l}]={episode_trajectory[idx][l]}')

                print(f'setting landmark on:{last_positions}')

                replay_trajectory, env,tagged_count, score =self.offline_replay_success(
                        agent_actions_list=episode_actions,
                        last_agent_pos=last_positions,
                        init_agent_pos=init_agent_pos,
                        init_agent_vel_ang=init_vel_ang,
                        init_agent_vel_lin=init_vel_lin,
                        init_agent_headings=init_agent_dir,
                        max_steps=max_steps)

                self.plot_episode_gone_trajectory(np.stack(replay_trajectory), episode=f'{i}_replayed_tagged_{tagged_count}_', env=env)
                self.log_episode(episode_id=f'{i}_replayed', env=env)
                env.delete()
                del env
        print("Training complete.")
    def add_record_to_rb(self, state, obs, actions, rewards, next_state, next_obs, done, tagged=False):
        if self.use_tagged_replay_buffer:
            self.replay_buffer.add(state, obs, actions, rewards, next_state, next_obs, done, tagged=tagged)
        else:
            self.replay_buffer.add(state, obs, actions, self.reward_sum(rewards), next_state, next_obs, done)
    def offline_replay_success(self, agent_actions_list:list, init_agent_pos, init_agent_headings, last_agent_pos,
                               init_agent_vel_ang, init_agent_vel_lin, max_steps=max_steps):
        env=self.env.copy()
        env.reset_tensor()
        new_landmarks = last_agent_pos.clone().detach()  # shape: (num_agents, 2)

        # Step 1: Compute centroid as new origin (mean of new landmark positions)
        origin = new_landmarks.mean(dim=0)

        # Step 2: PCA to align the x-axis with the principal direction of landmarks
        centered = new_landmarks - origin
        cov = centered.T @ centered
        eigvals, eigvecs = torch.linalg.eigh(cov)
        x_axis = eigvecs[:, -1] / torch.norm(eigvecs[:, -1])  # dominant eigenvector
        y_axis = torch.tensor([-x_axis[1], x_axis[0]], device=x_axis.device)
        rot_matrix = torch.stack([x_axis, y_axis])  # shape: (2, 2)

        # Step 3: Define coordinate transform (shift + rotate)
        def transform(pos: torch.Tensor):
            return (pos - origin) @ rot_matrix.T

        # === Apply transform to landmarks, agent pos/dir, obstacle pos ===
        env.landmarks = transform(new_landmarks)

        env.agent_pos = transform(init_agent_pos.clone().detach())
        init_agent_headings = init_agent_headings.clone().detach()
        # Agent headings → rotate unit vectors, then re-encode angle
        dir_x = torch.cos(init_agent_headings)
        dir_y = torch.sin(init_agent_headings)
        dir_vecs = torch.stack([dir_x, dir_y], dim=1)  # shape: (num_agents, 2)
        rotated_dir_vecs = dir_vecs @ rot_matrix.T
        env.agent_dir = torch.atan2(rotated_dir_vecs[:, 1], rotated_dir_vecs[:, 0])

        # Obstacles
        env.obstacle_pos = transform(self.env.obstacle_pos.clone().detach())

        # Obstacle radii remain unchanged (not coordinates)
        env.obstacle_radius = self.env.obstacle_radius.clone().detach()

        # Velocities (not rotated, assuming in local agent frame)
        env.agent_vel_lin = init_agent_vel_lin.clone().detach()
        env.agent_vel_ang = init_agent_vel_ang.clone().detach()

        done = torch.full((env.num_agents,), False, dtype=torch.bool, device=self.device)
        env.done=torch.full((env.num_agents,), False, dtype=torch.bool, device=self.device)
        env.covered=torch.full((env.num_agents,), False, dtype=torch.bool, device=self.device)

        env._reset_hungarian()
        env._init_static_state_part()
        obs = env.get_all_obs_tensor()
        state = env.state_tensor()
        episode_trajectory=[env.agent_pos.cpu().clone().detach().numpy()]
        j = 0
        do_nothing=torch.zeros_like(agent_actions_list[0], device=self.device)
        tagged_count=0

        while j < max_steps and not done.all():

            if len(agent_actions_list)  <=j:
                print(f'do nothing from {j}')
                actions=do_nothing
            else:

                actions=agent_actions_list[j].clone().detach()

            actions[done] = torch.tensor([0.0, 0.0], device=actions.device)

            next_state, next_obs, rewards, next_done = env.step_tensor(actions)

            tagged = ((~done) & next_done).any().item()
            if tagged:
                print(f'tagged on step:{j}:\ndone: {done}\n next_done: {next_done}')
                tagged_count=tagged_count+1


            done = next_done.clone().detach().to(device=device)
            self.add_record_to_rb(state, obs, actions, rewards, next_state, next_obs, done, tagged=tagged)

            state = next_state
            obs = next_obs
            episode_trajectory.append(env.agent_pos.cpu().clone().detach().numpy())

            j=j+1
            if j%1000==0:
                print(f'replayed {j} steps')
        score = self.reward_from_rb(self.env.current_score).mean().cpu().item()
        print(f'replay finished, tagged  count: {tagged_count}, score: {score}')
        return episode_trajectory, env, tagged_count, score


    def plot_learning_curve(self, episode=0):
        plt.figure(figsize=(8, 5))
        plt.plot(self.score_history)
        plt.title('Learning Curve (Score)')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.grid(True)
        plt.savefig(f'learning_curve_episode_{episode}.png', dpi=300)
        plt.close()

    def plot_actor_loss(self, episode=0):
        plt.figure(figsize=(8, 5))
        plt.plot(self.actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'actor_loss_episode_{episode}.png', dpi=300)
        plt.close()

    def plot_critic_loss(self, episode=0):
        plt.figure(figsize=(8, 5))
        plt.plot(self.critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'critic_loss_episode_{episode}.png', dpi=300)
        plt.close()
    def reward_from_rb(self, reward_components: torch.Tensor) -> torch.Tensor:

        if not self.use_tagged_replay_buffer:
            return reward_components
        else:
            return self. reward_sum(reward_components)
    def reward_sum(self, reward_components: torch.Tensor):
        """
        Converts multi-component rewards into total rewards.
        Args:
            reward_components: shape [N, 9] or [B, N, 9]
            reward_scales: shape [9]
        Returns:
            reward: shape [N] or [B, N]
        """

        if reward_components.ndim == 2:
            # [N, 9]
            return (reward_components * self.reward_scales).sum(dim=-1)  # [N]
        elif reward_components.ndim == 3:
            # [B, N, 9]
            return (reward_components * self.reward_scales.view(1, 1, -1)).sum(dim=-1)  # [B, N]
        else:
            raise ValueError(f"Invalid reward_components shape: {reward_components.shape}")

    def try_actor(self, reset=True):
        print('trying current actor')
        done = torch.full((self.env.num_agents,), False, dtype=torch.bool)
        N = self.env.num_agents

        j = 0
        if reset:
            state, obs = self.env.reset_tensor()
        else:
            state=self.env.state_tensor()
            obs=self.env.get_all_obs_tensor()

        time_to_complete = torch.zeros(N, dtype=torch.int32, device=device)
        distance_until_done = torch.zeros(N, dtype=torch.float32, device=device)    # positions
        prev_pos = self.env.agent_pos.detach().clone().to(device)  # (N,2)
        episode_trajectory = [self.env.agent_pos.cpu().clone().detach().numpy()]
        while j < max_steps and not done.all():
            actions = self.choose_actions(obs, use_noise=False)
            state, obs, rewards, done = self.env.step_tensor(actions)
            # episode_trajectory.append(self.env.agent_pos.cpu().clone().detach().numpy())
            cur_pos = self.env.agent_pos.detach().clone().to(device)  # (N,2)
            step_dist = torch.linalg.vector_norm(cur_pos - prev_pos, dim=1)  # (N,)

            # mask for agents that are still NOT done at this step
            not_done = ~done

            # update metrics "in place"
            time_to_complete += not_done.to(torch.int32)
            distance_until_done += step_dist * not_done.to(torch.float32)

            # record + advance
            episode_trajectory.append(cur_pos.cpu().clone().detach().numpy())
            prev_pos = cur_pos

            j = j + 1
            if j%100==0:
                print(f'{j} steps passed')

        score = self.env.current_score
        print(f'score: {score}, done: {done.sum().item()}')
        return episode_trajectory, score, time_to_complete.detach().cpu().numpy(), distance_until_done.detach().cpu().numpy()


    def plot_episode_gone_trajectory(self, trajectory, episode, env = None):
        if env == None:
            env=self.env
        # trajectory shape: (T, num_agents, 2)
        num_agents = trajectory.shape[1]

        fig, ax = plt.subplots(figsize=(7, 7))
        # half = env.env_size.item() / 2
        # ax.set_xlim(-half, half)
        # ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.set_title(f"Episode {episode} Trajectories")

        # Obstacles
        for i in range(env.num_obstacles):
            pos = env.obstacle_pos[i].cpu().numpy()
            rad = env.obstacle_radius[i].item()
            circle = plt.Circle(pos, rad, color='gray', alpha=0.5)
            ax.add_patch(circle)

        # Agent trajectories
        colors = plt.cm.get_cmap('tab10', num_agents)
        for agent_idx in range(num_agents):
            path = trajectory[:, agent_idx]
            ax.plot(path[:, 0], path[:, 1], color=colors(agent_idx), linewidth=1.5)
            ax.plot(path[0, 0], path[0, 1], 'o', color='blue')  # start
            ax.plot(path[-1, 0], path[-1, 1], 'o', color='green')  # end
        # Landmarks
        for lm in env.landmarks.cpu().numpy():
            ax.plot(lm[0], lm[1], 'rx', markersize=8, label='Landmark')


        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'trajectory_episode_{episode}.png', dpi=300)
        plt.close()

    def plot_episode_new_trajectory(self, episode):
        trajectory, _, _, _ = self.try_actor()
        self.plot_episode_gone_trajectory(np.stack(trajectory), episode)
    def save_actor(self, file_name:str = 'simple_actor.pth'):
        pass

    def train_loop(
            self,
            n_games: int = n_games,
            train_each: int = train_each,
            evaluate: bool = False,
            checkpoint_path: str = "training_state.pkl",
            patience=patience,
            score_avg_window=score_avg_window,
            max_steps=max_steps,
            start_training_after=start_training_after,
            rescale_env_rewards=None,
            min_episodes_before_early_stop:int=min_episodes_before_early_stop,
            meta_extra: Optional[dict] = None,
            meta_path: str = "meta.json",
            post_episode_callback: Optional[Callable[[int, bool], None]] = None,
    ) -> None:
        """
        Resumable training loop with checkpointing.

        Args:
            n_games (int): Total episodes to run.
            train_each (int): Frequency to trigger learning.
            evaluate (bool): If True, only evaluates.
            checkpoint_path (str): Path to save/load training state.
            meta_extra (dict, optional): Static info merged into ``meta.json``
                (e.g. ``{"seed": 9832, "mode": "full"}``). Useful for the
                compute-cost / reproducibility table.
            meta_path (str): Path of the human-readable JSON metadata file.
            post_episode_callback (callable, optional): Called after a saved
                episode checkpoint is written. Receives
                ``(episodes_completed, finished)``.
        """

        # === Load previous training state if exists ===
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                state_dict = pickle.load(f)
            start_episode = state_dict["episode"]
            self.score_history = state_dict["score_history"]
            best_score = state_dict["best_score"]
            self.replay_buffer.load("replay_buffer.pkl")
            episodes_without_improvement = state_dict["episodes_without_improvement"]
            total_steps = state_dict['total_steps']
            self.actor_losses=state_dict['actor_losses']
            self.critic_losses=state_dict['critic_losses']
            # Compute-cost accumulators (safe defaults for older pickles).
            total_train_seconds = float(state_dict.get('total_train_seconds', 0.0))
            peak_gpu_bytes = int(state_dict.get('peak_gpu_bytes', 0))
            self.load_checkpoint()

            print(f"Resuming from episode {start_episode} "
                  f"(prior train time: {total_train_seconds/3600:.2f} h, "
                  f"peak GPU: {peak_gpu_bytes/1e9:.2f} GB)")
        else:
            start_episode = 0
            self.score_history = []
            best_score = -float("inf")
            episodes_without_improvement = 0
            total_steps = 0
            self.actor_losses=[]
            self.critic_losses=[]
            total_train_seconds = 0.0
            peak_gpu_bytes = 0

        # Reset within-session peak so torch.cuda.max_memory_allocated() reflects
        # this process; we max() it against the persisted all-time peak below.
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        for i in range(start_episode, n_games):

            episode_start_time = time.perf_counter()

            state, obs = self.env.reset_tensor()
            done = torch.full((self.env.num_agents,), False, dtype=torch.bool, device=self.device)
            episode_trajectory = [self.env.agent_pos.cpu().clone().detach().numpy()]
            tagged_count=0
            j=0
            while j <max_steps and not done.all():
                # print(f'step: {total_steps}')
                if evaluate:
                    self.env.render()
                    time.sleep(0.1)
                # print(f'chosing action by obs')
                actions = self.choose_actions(obs)
                # print(f'taking actions')
                next_state, next_obs, rewards, next_done = self.env.step_tensor(actions)
                tagged = ((~done) & next_done).any().item()
                done=next_done.clone().detach()
                # print(f'saving rb')
                self.add_record_to_rb(state, obs, actions, rewards, next_state, next_obs, done, tagged=tagged)

                state = next_state
                obs = next_obs

                total_steps += 1
                if(tagged):
                    tagged_count=tagged_count+1
                if not evaluate and total_steps % train_each == 0:
                    if total_steps<start_training_after:
                        print(f'replay buffer not ready: {total_steps}/{start_training_after}')
                    else:

                        if rescale_env_rewards != None:
                            self.reward_scales = torch.tensor(rescale_env_rewards(total_steps), dtype=torch.float32,
                                                              device=self.device)
                            print(f'rescaled to{self.reward_scales}')
                        critic_loss, actor_loss  = self.learn()
                        self.actor_losses.append(actor_loss)
                        self.critic_losses.append(critic_loss)
                        print(f"Trained at step {total_steps}, actor loss: {actor_loss}, critic loss : {critic_loss}")
                episode_trajectory.append(self.env.agent_pos.cpu().clone().detach().numpy())
                self.log_env_step_to_csv(total_step=total_steps, episode_id=i)
                j=j+1
            # Episode completed (loop exited normally — not interrupted).
            # Accumulate ONLY succeeded episodes so the running total survives
            # restarts faithfully; a Ctrl-C mid-episode discards that episode's
            # time, which is intentional.
            episode_duration = time.perf_counter() - episode_start_time
            total_train_seconds += episode_duration
            if torch.cuda.is_available():
                try:
                    peak_gpu_bytes = max(
                        peak_gpu_bytes, int(torch.cuda.max_memory_allocated())
                    )
                except Exception:
                    pass
            scores = self.reward_from_rb(self.env.current_score)
            score = scores.mean().cpu().item()
            self.score_history.append(score)
            avg_score = torch.tensor(self.score_history[-score_avg_window:], device='cpu').float().mean().item()

            print(f"Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Tagged count: {tagged_count}")
            self.log_episode(episode_id=f'{i}')

            print(f"Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}")

            if not evaluate and avg_score > best_score and i>min_episodes_before_early_stop:
                best_score = avg_score
                self.save_checkpoint(file_pref=f'best_{best_score}_{i}')
                print("Checkpoint saved (best model)")
                episodes_without_improvement=0
            else:
                if i>min_episodes_before_early_stop:
                    episodes_without_improvement+=1

            # Preserve the existing checkpoint cadence.
            if i>0:

                print("Training progress saving.")
                state_dict = {
                    "episode": i + 1,
                    "score_history": self.score_history,
                    "best_score": best_score,
                'episodes_without_improvement': episodes_without_improvement,
                    'total_steps':total_steps,
                    'actor_losses':self.actor_losses,
                    'critic_losses':self.critic_losses,
                    'total_train_seconds': total_train_seconds,
                    'peak_gpu_bytes': peak_gpu_bytes,


                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state_dict, f)


                self.replay_buffer.save("replay_buffer.pkl")

                self.save_checkpoint()

                self._save_meta_json(
                    meta_path=meta_path,
                    meta_extra=meta_extra,
                    n_games_target=n_games,
                    episodes_completed=i + 1,
                    total_steps=total_steps,
                    total_train_seconds=total_train_seconds,
                    peak_gpu_bytes=peak_gpu_bytes,
                    finished=False,
                )
                if post_episode_callback is not None:
                    post_episode_callback(i + 1, False)

                self.plot_learning_curve()
                self.plot_actor_loss()
                self.plot_critic_loss()
                # self.save_actor(f'simple_actor_{i}.pth')
                self.plot_episode_gone_trajectory(np.stack(episode_trajectory), f'{i}_tagged_ {tagged_count} _ score_{score}_')
                print("Training progress saved.")
            if i%10==0 and i>0:
                self.save_checkpoint(file_pref=f'episode_{i}_')
                self.replay_buffer.save(f"replay_buffer_{i}.pkl")
                with open(f'episode_{i}_{checkpoint_path}', "wb") as f:
                    pickle.dump(state_dict, f)
                self.plot_episode_new_trajectory(episode=f'{i}_1')
                self.log_episode(episode_id=f'{i}_1')
                self.plot_episode_new_trajectory(episode=f'{i}_2')
                self.log_episode(episode_id=f'{i}_2')
                self.plot_episode_new_trajectory(episode=f'{i}_3')
                self.log_episode(episode_id=f'{i}_3')

            if episodes_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered: No improvement in {patience} episodes.")
                print("Training progress saving.")
                state_dict = {
                    "episode": i + 1,
                    "score_history": self.score_history,
                    "best_score": best_score,
                    'episodes_without_improvement': episodes_without_improvement,
                    'total_steps': total_steps,
                    'actor_losses': self.actor_losses,
                    'critic_losses': self.critic_losses,
                    'total_train_seconds': total_train_seconds,
                    'peak_gpu_bytes': peak_gpu_bytes,

                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state_dict, f)

                self.replay_buffer.save("replay_buffer.pkl")
                self.save_checkpoint()
                self._save_meta_json(
                    meta_path=meta_path,
                    meta_extra=meta_extra,
                    n_games_target=n_games,
                    episodes_completed=i + 1,
                    total_steps=total_steps,
                    total_train_seconds=total_train_seconds,
                    peak_gpu_bytes=peak_gpu_bytes,
                    finished=True,
                )
                if post_episode_callback is not None:
                    post_episode_callback(i + 1, True)
                print("Training progress saved.")

                self.plot_learning_curve()
                self.plot_actor_loss()
                self.plot_critic_loss()
                self.plot_episode_gone_trajectory(np.stack(episode_trajectory), episode=i)
                break
        else:
            # Loop exited because the for-range completed all n_games. Mark finished.
            self._save_meta_json(
                meta_path=meta_path,
                meta_extra=meta_extra,
                n_games_target=n_games,
                episodes_completed=n_games,
                total_steps=total_steps,
                total_train_seconds=total_train_seconds,
                peak_gpu_bytes=peak_gpu_bytes,
                finished=True,
            )
            if post_episode_callback is not None:
                post_episode_callback(n_games, True)
        print("Training complete.")

    def _save_meta_json(
        self,
        *,
        meta_path: str,
        meta_extra: Optional[dict],
        n_games_target: int,
        episodes_completed: int,
        total_steps: int,
        total_train_seconds: float,
        peak_gpu_bytes: int,
        finished: bool,
    ) -> None:
        """Persist training compute/reproducibility metadata to ``meta_path``.

        ``total_train_seconds`` accumulates only over completed episodes, so it
        survives interruption-and-resume: a crashed mid-episode does not get
        counted, and the next launch picks up the running total from the
        existing pickle.
        """
        meta = {
            "num_agents": int(self.env.num_agents),
            "reward_scales": [float(x) for x in self.reward_scales.tolist()],
            "n_games_target": int(n_games_target),
            "episodes_completed": int(episodes_completed),
            "total_steps": int(total_steps),
            "total_train_seconds": float(total_train_seconds),
            "total_train_hours": float(total_train_seconds) / 3600.0,
            "peak_gpu_bytes": int(peak_gpu_bytes),
            "peak_gpu_gb": float(peak_gpu_bytes) / 1e9,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(self.device),
            "batch_size": int(getattr(self.replay_buffer, "batch_size", 0)),
            "replay_buffer_size": int(getattr(self.replay_buffer, "max_size", 0)),
            "use_tagged_replay_buffer": bool(self.use_tagged_replay_buffer),
            "finished": bool(finished),
            "last_save_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if torch.cuda.is_available():
            try:
                meta["gpu_name"] = torch.cuda.get_device_name(0)
                meta["gpu_total_memory_gb"] = (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                )
                meta["cuda_version"] = torch.version.cuda
            except Exception:
                pass
        if meta_extra:
            for k, v in meta_extra.items():
                meta[k] = v
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def log_episode(self, episode_id, env=None):
        if env==None:
            env=self.env
        scores = self.reward_from_rb(env.current_score)
        score = scores.mean().cpu().item()
        # self.score_history.append(score)
        # avg_score = torch.tensor(self.score_history[-score_avg_window:], device='cpu').float().mean().item()
        num_done_agents = env.dones.sum().item()

        with open('episode_log.txt', 'a') as f:
            print(f"Episode {episode_id}, Mean Score: {score:.2f}, Tagged count: {num_done_agents}",
                  file=f)
            print(
                f'reward components:{self.env.current_score}\nX\nreward_scales:{self.reward_scales}\n=\nScores:{scores}', file=f)

    def offline_replay_graph(self, trajectories, init_headings):
        env=self.env.copy()
        new_landmarks = []
        replay_actions_list=[]
        print('Optimizing roads:')
        for agent_id in range(env.num_agents):
            traj = trajectories[:, agent_id].clone().detach().to(self.device)
            heading = init_headings[agent_id].clone().detach().to(self.device)
            steps_counts, dists, prev_inx, last_dir, actions, furthest_idx = env.graph_search_cuda(traj, heading)
            print(f'road for agent_{agent_id} optimized')
            new_landmarks.append(traj[furthest_idx].clone().detach())
            ind=furthest_idx
            replay_actions=[]
            while(ind > 0):
                replay_actions.append(actions[ind].clone().detach())
                ind=prev_inx[ind]
            replay_actions.reverse()  # reverse to forward order
            replay_actions_list.append(replay_actions)
        new_landmarks = torch.stack(new_landmarks)
        print(f'reforming env: ')
            # Step 1: Compute centroid as new origin
        origin = new_landmarks.mean(dim=0)

            # Step 2: PCA for principal axis alignment
        centered = new_landmarks - origin
        cov = centered.T @ centered
        eigvals, eigvecs = torch.linalg.eigh(cov)
        x_axis = eigvecs[:, -1] / torch.norm(eigvecs[:, -1])
        y_axis = torch.stack([-x_axis[1], x_axis[0]])
        rot_matrix = torch.stack([x_axis, y_axis]).to(device=self.device)  # (2, 2)

            # Step 3: Apply translation and rotation to landmarks, agents, obstacles
        def transform(pos):
                return (pos - origin.to(device)) @ rot_matrix.T

            # Transform all components
        env.landmarks = transform(new_landmarks)
        env.agent_pos = transform(trajectories[0].clone().detach())
        env.obstacle_pos = transform(env.obstacle_pos.clone().detach())

                # Rotate headings accordingly (assumes 2D vectors for directions)
        agent_dir = (rot_matrix @ torch.stack([
                    torch.cos(init_headings),
                    torch.sin(init_headings)
                ])).T
        env.agent_dir = torch.atan2(agent_dir[:, 1], env.agent_dir[:, 0])

        env._init_static_state_part()
        env._reset_hungarian()

        observation =env.get_all_obs_tensor()
        state=env.state_tensor()
        done = torch.full((env.num_agents,), False, dtype=torch.bool, device=self.device)
        # --- Prepare actions tensor with padding ---
        max_len = max(len(actions) for actions in replay_actions_list)
        print(f'Starting replay: maxlenth of the replaying episode : {max_len}')
        padded_actions = torch.ones((max_len, env.num_agents, 2), device=device)

        for agent_id, actions_list in enumerate(replay_actions_list):
            for t, action in enumerate(actions_list):
                padded_actions[t, agent_id] = action.to(device)
        if self.use_tagged_replay_buffer:
            self.offline_buffer=tagged_replay_buffer.ReplayBuffer(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=env.action_dim,
                                        device=device, num_agents=env.num_agents, replay_buffer_size=max_len, batch_size=batch_size)
        else:
            self.offline_buffer=replay_buffer.ReplayBuffer(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=env.action_dim,
                                        device=device, num_agents=env.num_agents, replay_buffer_size=max_len, batch_size=batch_size)

        j=0
        resulted_trajectory=[env.agent_pos.cpu().clone().detach().numpy()]
        while j < max_len and not done.all():
            actions = padded_actions[j]
            actions = torch.nan_to_num(actions, nan=1.0)
            # print(f'taking actions')
            next_state, next_obs, rewards, done = env.step_tensor(actions)
            # print(f'saving rb')
            self.offline_buffer.add(state, observation, actions, rewards, next_state, next_obs, done)
            resulted_trajectory.append(env.agent_pos.cpu().clone().detach().numpy())

            state = next_state
            observation = next_obs
            j=j+1
        print(f'replay successfull')
        return resulted_trajectory

class MADDPGSharedActorCritic(MADDPGBase):
    def __init__(
        self,
        env: DiffDriveParallelEnv,
            reward_scales: Sequence[float],
            batch_size: int = batch_size,
            replay_buffer_size: int = replay_buffer_size,
        device: Union[str, torch.device] = device,
            use_tagged_replay_buffer: bool = True,
    ) -> None:
        """
        Initializes MADDPG agent with shared actor and critic networks.

        Args:
            env (DiffDriveParallelEnv): Environment instance.
            device (str | torch.device): Device for model placement.

        Sets:
            self.critic (SharedCritic): Centralized critic for all agents.
            self.critic_target (SharedCritic): Target critic network.
            self.actor (SimpleActor): Shared actor across all agents.
            self.actor_target (SimpleActor): Target actor network.
        """
        super().__init__(env,reward_scales=reward_scales, device=device, batch_size = batch_size, replay_buffer_size = replay_buffer_size,use_tagged_replay_buffer=use_tagged_replay_buffer)
        critic_input_dim, critic_output_dim=self._critic_dim()
        self.critic=SharedCritic(input_dim=critic_input_dim, output_dim=critic_output_dim, device=self.device, chckpnt_file='shared_critic.pth')
        self.critic_target=SharedCritic(input_dim=critic_input_dim, output_dim=critic_output_dim,   device=self.device, chckpnt_file='shared_critic_target.pth')
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor=SimpleActor(env.obs_dim, env.action_dim, device=device, chckpnt_file='shared_actor.pth')
        self.actor_target=SimpleActor(env.obs_dim, env.action_dim, device=device, chckpnt_file='shared_actor_target.pth')
        self.actor_target.load_state_dict(self.actor.state_dict())
    def _critic_dim(self):
        return self.env.state_dim +  self.env.num_agents * self.env.action_dim, self.env.num_agents

    def learn(self, buffer=None) -> tuple[None, None] | tuple[float, float]:
        """
        Performs one training step for actor and critic using properly masked joint updates,
        safely handling newly and long-done agents.
        """
        buffer = buffer or self.replay_buffer
        obs, next_obs, state, next_state, action, reward_components, done = buffer.sample()
        reward=self.reward_from_rb(reward_components)
        B, N, act_dim = action.shape

        # === ACTIVE MASK ===
        active_mask = (~done).float()  # [B, N]

        # === CRITIC UPDATE ===
        with torch.no_grad():
            next_action = self.actor_target(next_obs)  # [B, N, act_dim]
            masked_next_action = next_action * active_mask.unsqueeze(-1)  # mask inactive agents
            joint_next_action = masked_next_action.view(B, N * act_dim)

            next_q = self.critic_target(next_state, joint_next_action)  # [B, 1]

            masked_reward = reward * active_mask  # zero reward for done agents
            total_reward = masked_reward.sum(dim=1, keepdim=True)  # [B, 1]

            y = total_reward + gamma * next_q  # TD target

        joint_action = action.view(B, N * act_dim)
        q_pred = self.critic(state, joint_action)  # [B, 1]

        critic_loss = F.mse_loss(q_pred, y)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # === ACTOR UPDATE ===
        pred_actions = self.actor(obs)  # [B, N, act_dim]
        masked_pred_actions = pred_actions * active_mask.unsqueeze(-1)  # mask inactive agents
        joint_pred_actions = masked_pred_actions.view(B, N * act_dim)

        actor_loss = -self.critic(state, joint_pred_actions).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # === SOFT TARGET UPDATES ===
        self.update_params_vectorized(self.critic, self.critic_target, tau)
        self.update_params_vectorized(self.actor, self.actor_target, tau)

        return critic_loss.item(), actor_loss.item()

    def  load_actor(self):
        """
        Loads the saved weights of the actor network from the checkpoint path.

        Assumes:
            - self.actor is an instance of SimpleActor.
            - SimpleActor has a method `.load_checkpoint()` that handles device mapping.
        """
        self.actor.load_checkpoint()

    def choose_actions(
            self,
            obs_list: torch.Tensor,  # shape: [num_agents, obs_dim], on self.device
            use_noise: bool = True
    ) -> torch.Tensor:  # shape: [num_agents, act_dim], on self.device
        """
        Selects actions for all agents using the shared actor.

        Args:
            obs_list (torch.Tensor): A tensor of observations for each agent.
                - Shape: [num_agents, obs_dim]
                - Type: torch.FloatTensor
                - Must be on the same device as the actor network (e.g., CUDA or CPU)

            use_noise (bool): Whether to include exploration noise during action selection.
                - True during training for exploration
                - False during evaluation

        Returns:
            torch.Tensor: Actions for all agents.
                - Shape: [num_agents, act_dim]
                - Type: torch.FloatTensor
                - On the same device as self.actor
        """
        return self.actor.choose_action(obs_list, use_noise=use_noise, eval_mode=True)

    def save_checkpoint(self, file_pref=None):
        self.actor.save_checkpoint(file_prefix=file_pref)
        self.critic.save_checkpoint(file_prefix=file_pref)
        self.critic_target.save_checkpoint(file_prefix=file_pref)
        self.actor_target.save_checkpoint(file_prefix=file_pref)
    def load_checkpoint(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()
        self.actor_target.load_checkpoint()
    def save_actor(self, file_name:str = 'simple_actor.pth'):
        self.actor.save_checkpoint(file_name)

class MADDPGSharedActorCriticIndependentQmean(MADDPGSharedActorCritic):

    def learn(self, buffer=None):
        buffer = buffer or self.replay_buffer
        obs, next_obs, state, next_state, action, reward_components, done = buffer.sample()
        reward = self.reward_from_rb(reward_components)
        B, N, act_dim = action.shape

        critic_losses = []
        actor_losses = []

        for i in range(N):
            # === Valid if agent not done OR has nonzero reward (include final step) ===
            is_valid = (done[:, i] == False) | (reward[:, i].abs() > 1e-5)
            if is_valid.sum() == 0:
                continue

            # === Critic update ===
            with torch.no_grad():
                next_actions_all = self.actor_target(next_obs)  # [B, N, act_dim]
                next_actions_i = next_actions_all.clone()
                agent_i_next_action = self.actor_target(next_obs)[:, i].unsqueeze(1)
                next_actions_i[:, i:i + 1] = agent_i_next_action  # insert differentiable part
                joint_next_action = next_actions_i.view(B, N * act_dim)
                next_q = self.critic_target(next_state, joint_next_action)

                y = reward[:, i].unsqueeze(1) + gamma * next_q  # TD target

            joint_action = action.view(B, N * act_dim)
            q_pred = self.critic(state, joint_action)

            critic_loss = F.mse_loss(q_pred[is_valid], y[is_valid])
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(critic_loss)

            # === Actor update ===
            pred_actions_all = self.actor(obs)  # [B, N, act_dim]
            pred_actions_i = pred_actions_all.detach().clone()
            agent_i_pred = self.actor(obs)[:, i].unsqueeze(1)  # [B, 1, act_dim]
            pred_actions_i[:, i:i + 1] = agent_i_pred  # reintegrate differentiable action

            joint_pred_actions = pred_actions_i.view(B, N * act_dim)
            actor_loss = -self.critic(state, joint_pred_actions)[is_valid].mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            actor_losses.append(actor_loss)

        # === Soft Target Updates ===
        self.update_params_vectorized(self.critic, self.critic_target, tau)
        self.update_params_vectorized(self.actor, self.actor_target, tau)

        if not actor_losses:
            return None, None

        return torch.stack(critic_losses).mean().item(), torch.stack(actor_losses).mean().item()

    def train_critic_only(self, reward_scales, num_passes=10, batch_size=128, gamma=0.95, tau=0.01):
        """
        Train the critic using the existing replay buffer and new reward scales.
        Actor remains frozen.

        Args:
            reward_scales (tensor): shape [num_reward_components]
            num_passes (int): how many full passes over the buffer to make
            batch_size (int): batch size per gradient step
        """

        self.actor.eval()  # Freeze actor
        for param in self.actor.parameters():
            param.requires_grad = False

        rb_size = self.replay_buffer.size
        total_steps = num_passes * (rb_size // batch_size)
        critic_losses = []
        steps = 0

        while steps < total_steps:
            obs, next_obs, state, next_state, action, reward_components, done = self.replay_buffer.sample(batch_size)
            reward = (reward_components * reward_scales.view(1, 1, -1)).sum(dim=-1)  # [B, N]
            B, N, act_dim = action.shape

            for i in range(N):
                is_valid = (done[:, i] == False) | (reward[:, i].abs() > 1e-5)
                if is_valid.sum() == 0:
                    continue

                with torch.no_grad():
                    next_actions_all = self.actor_target(next_obs)
                    next_actions_i = next_actions_all.clone()
                    agent_i_next_action = self.actor_target(next_obs)[:, i].unsqueeze(1)
                    next_actions_i[:, i:i + 1] = agent_i_next_action
                    joint_next_action = next_actions_i.view(B, N * act_dim)
                    next_q = self.critic_target(next_state, joint_next_action)
                    y = reward[:, i].unsqueeze(1) + gamma * next_q

                joint_action = action.view(B, N * act_dim)
                q_pred = self.critic(state, joint_action)

                critic_loss = F.mse_loss(q_pred[is_valid], y[is_valid])
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
                critic_losses.append(critic_loss.item())

                steps += 1
                if steps >= total_steps:
                    break

            # Optional: update critic target softly
            self.update_params_vectorized(self.critic, self.critic_target, tau)

        print(f"Finished training critic-only for {steps} steps ({num_passes}x over buffer).")
        return sum(critic_losses) / len(critic_losses) if critic_losses else None


class MADDPGSharedActorCriticIndependent(MADDPGSharedActorCritic):
    def learn(self, buffer=None):
        buffer = buffer or self.replay_buffer
        obs, next_obs, state, next_state, action, reward_components, done = buffer.sample()
        reward = self.reward_from_rb(reward_components)  # shape: [B, N]
        B, N, act_dim = action.shape

        critic_losses = []
        actor_losses = []

        # === Critic update (single step) ===
        with torch.no_grad():
            # Target actions and Q-values
            next_actions_all = self.actor_target(next_obs)  # [B, N, act_dim]
            joint_next_action = next_actions_all.reshape(B, N * act_dim)
            next_q = self.critic_target(next_state, joint_next_action)  # [B, N]

            # Mask terminals to avoid bootstrapping on done
            not_done = (~done).float()  # [B, N]
            y = reward + gamma * not_done * next_q  # [B, N]

            # Use current policy actions but stop grad for critic update
            current_actions_all = self.actor(obs)  # [B, N, act_dim]

        joint_action = current_actions_all.reshape(B, N * act_dim)
        q_pred = self.critic(state, joint_action)  # [B, N]

        # Valid samples: either not done or has a non-trivial reward
        is_valid = (~done) | (reward.abs() > 1e-5)  # [B, N]
        critic_loss = F.mse_loss(q_pred[is_valid], y[is_valid])
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        critic_losses.append(critic_loss.detach())

        # === Actor update (accumulate per-agent loss; single backward) ===
        pred_actions_all = self.actor(obs)  # [B, N, act_dim] (live graph)

        # Freeze critic params; let gradients flow through it to actions only
        for p in self.critic.parameters():
            p.requires_grad_(False)

        actor_loss_total = None
        for i in range(N):
            mask = is_valid[:, i]
            if mask.sum() == 0:
                continue

            # Copy all actions detached, then reinsert live action for agent i
            pred_actions_i = pred_actions_all.detach().clone()  # [B, N, act_dim]
            pred_actions_i[:, i] = pred_actions_all[:, i]  # keep i-th path live

            # Critic expects joint action
            joint_pred_actions = pred_actions_i.reshape(B, N * act_dim)

            # Q for agent i
            q_vals_i = self.critic(state, joint_pred_actions)[:, i]  # [B]
            loss_i = -q_vals_i[mask].mean()

            actor_losses.append(loss_i.detach())
            actor_loss_total = loss_i if actor_loss_total is None else (actor_loss_total + loss_i)

        if actor_loss_total is not None:
            self.actor.optimizer.zero_grad()
            actor_loss_total.backward()  # single backward for the shared graph
            self.actor.optimizer.step()

        # Unfreeze critic params
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # === Soft Target Updates ===
        self.update_params_vectorized(self.critic, self.critic_target, tau)
        self.update_params_vectorized(self.actor, self.actor_target, tau)

        # === Return metrics ===
        if actor_loss_total is None:
            return torch.stack(critic_losses).mean().item(), None
        return torch.stack(critic_losses).mean().item(), torch.stack(actor_losses).mean().item()


class IDDPG(MADDPGSharedActorCritic):
    def _critic_dim(self):
        return self.env.state_dim +  self.env.action_dim, 1

    def learn(self, buffer=None):
        buffer = buffer or self.replay_buffer

        # === Sample from buffer ===
        obs, next_obs, state, next_state, action, reward_components, done = buffer.sample()
        reward = self.reward_from_rb(reward_components)  # [B, N]
        B, N, act_dim = action.shape

        critic_losses = []
        actor_losses = []

        # === Compute target Q-values ===
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)  # [B, N, act_dim]
            q_targets = []
            for i in range(N):
                a_i_next = next_actions[:, i]  # [B, act_dim]
                q_i_target = self.critic_target(next_state, a_i_next).squeeze(-1)  # [B]
                y_i = reward[:, i] + gamma * q_i_target * (~done[:, i])
                q_targets.append(y_i)
            y = torch.stack(q_targets, dim=1)  # [B, N]

        # === Compute predicted Q-values ===
        q_preds = []
        for i in range(N):
            a_i = action[:, i]  # [B, act_dim]
            q_i_pred = self.critic(state, a_i).squeeze(-1)  # [B]
            q_preds.append(q_i_pred)
        q_pred = torch.stack(q_preds, dim=1)  # [B, N]

        # === Critic loss ===
        is_valid = (~done) | (reward.abs() > 1e-5)  # include last meaningful steps
        critic_loss = F.mse_loss(q_pred[is_valid], y[is_valid])
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        critic_losses.append(critic_loss)

        # === Actor loss ===
        pred_actions = self.actor(obs)  # [B, N, act_dim]
        self.actor.optimizer.zero_grad()
        actor_losses = []

        for i in range(N):
            if is_valid[:, i].sum() == 0:
                continue

            a_i = pred_actions[:, i]  # [B, act_dim]
            q_i = self.critic(state, a_i).squeeze(-1)  # Q(s, pi(o_i))
            actor_loss_i = -q_i[is_valid[:, i]].mean()
            actor_losses.append(actor_loss_i)

        if actor_losses:
            actor_loss_total = torch.stack(actor_losses).mean()
            actor_loss_total.backward()
            self.actor.optimizer.step()
        else:
            actor_loss_total = None

        # === Soft target updates ===
        self.update_params_vectorized(self.critic, self.critic_target, tau)
        self.update_params_vectorized(self.actor, self.actor_target, tau)

        # === Return loss values ===
        return critic_loss.item(), actor_loss_total.item() if actor_loss_total is not None else None


class IDDPGWithoutS(MADDPGSharedActorCritic):
    def _critic_dim(self):
        return self.env.obs_dim + self.env.action_dim, 1

    def learn(self, buffer=None):
        buffer = buffer or self.replay_buffer

        # === Sample batch ===
        obs, next_obs, _, _, action, reward_components, done = buffer.sample()
        reward = self.reward_from_rb(reward_components)  # [B, N]
        B, N, act_dim = action.shape

        critic_losses = []
        actor_losses = []

        # === Target Q-values ===
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)  # [B, N, act_dim]
            q_targets = []
            for i in range(N):
                o_i_next = next_obs[:, i]  # [B, obs_dim]
                a_i_next = next_actions[:, i]  # [B, act_dim]
                q_i_target = self.critic_target(o_i_next, a_i_next).squeeze(-1)  # [B]
                y_i = reward[:, i] + gamma * q_i_target * (~done[:, i])
                q_targets.append(y_i)
            y = torch.stack(q_targets, dim=1)  # [B, N]

        # === Predicted Q-values ===
        q_preds = []
        for i in range(N):
            o_i = obs[:, i]  # [B, obs_dim]
            a_i = action[:, i]  # [B, act_dim]
            q_i_pred = self.critic(o_i, a_i).squeeze(-1)  # [B]
            q_preds.append(q_i_pred)
        q_pred = torch.stack(q_preds, dim=1)  # [B, N]

        # === Critic loss ===
        is_valid = (~done) | (reward.abs() > 1e-5)
        critic_loss = F.mse_loss(q_pred[is_valid], y[is_valid])
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # === Actor loss ===
        pred_actions = self.actor(obs)  # [B, N, act_dim]
        self.actor.optimizer.zero_grad()
        actor_losses = []

        for i in range(N):
            if is_valid[:, i].sum() == 0:
                continue
            o_i = obs[:, i]  # [B, obs_dim]
            a_i = pred_actions[:, i]  # [B, act_dim]
            q_i = self.critic(o_i, a_i).squeeze(-1)  # Q(o_i, a_i)
            actor_loss_i = -q_i[is_valid[:, i]].mean()
            actor_losses.append(actor_loss_i)

        if actor_losses:
            actor_loss_total = torch.stack(actor_losses).mean()  # mean to avoid coupling
            actor_loss_total.backward()
            self.actor.optimizer.step()
        else:
            actor_loss_total = None

        # === Soft Target Updates ===
        self.update_params_vectorized(self.critic, self.critic_target, tau)
        self.update_params_vectorized(self.actor, self.actor_target, tau)

        # === Return scalar losses ===
        return critic_loss.item(), actor_loss_total.item() if actor_loss_total is not None else None



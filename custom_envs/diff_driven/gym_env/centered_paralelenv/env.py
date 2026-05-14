from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torch
from typing import Union, Optional, Tuple, Dict

from config import *


class DiffDriveParallelEnv(ParallelEnv):
    metadata = {"render_modes": [render_mode], "name": env_name}

    def __init__(
            self,
            num_agents: int = num_agents,
            env_size: float = env_size,
            num_obstacles: int = num_obstacles,
            v_lin_max: float = v_lin_max,
            v_ang_max: float = v_ang_max,
            agent_radius: float = agent_radius,
            safe_dist: float = safe_dist,
            sens_range: float = sens_range,
            obstacle_size_min: float = obstacle_size_min,
            obstacle_size_max: float = obstacle_size_max,
            device: Union[str, torch.device] = device,
            normalise=normalise,


    ):

        super().__init__()
        self.normalise=normalise
        self.device=device
        self.obstacle_size_min=obstacle_size_min
        self.obstacle_size_max=obstacle_size_max
        self._num_agents = num_agents
        self.agents:list[str] = [f"agent_{i}" for i in range(self._num_agents)]
        self.possible_agents = self.agents[:]
        # World settings (converted to CUDA tensors)
        self.env_size = torch.tensor(env_size, device=device)
        self.num_landmarks = self._num_agents
        self.num_obstacles = num_obstacles
        self.v_lin_max = torch.tensor(v_lin_max, device=device)
        self.v_ang_max = torch.tensor(v_ang_max, device=device)
        self.agent_radius = torch.tensor(agent_radius, device=device)
        self.safe_dist = torch.tensor(safe_dist, device=device)
        self.sens_range = torch.tensor(sens_range, device=device)
        self.timestep = 0


        self.action_dim=2
        if normalise:
            self.obs_dim = 3 * self.num_obstacles + 2 * self.num_landmarks + (self.num_agents - 1) * 4
            self.state_dim = 4 * self.num_agents + 2 * self.num_landmarks + 3 * self.num_obstacles
        else:
            self.obs_dim=2*self.num_obstacles+2*self.num_landmarks+(self.num_agents-1)*3+self.action_dim
            self.state_dim=3*self.num_agents+2*self.num_landmarks+3*self.num_obstacles


        # Agent states (as tensors)
        self.agent_pos = torch.zeros((self._num_agents, 2), device=device)
        self.agent_vel_lin = torch.zeros((self._num_agents,), device=device)
        self.agent_vel_ang = torch.zeros((self._num_agents,), device=device)
        self.agent_dir = torch.zeros((self._num_agents,), device=device)

        # Landmarks and obstacles
        self.landmarks = torch.zeros((self.num_landmarks, 2), device=device)
        self.obstacle_pos = torch.zeros((self.num_obstacles, 2), device=device)
        self.obstacle_radius = torch.zeros((self.num_obstacles,), device=device)

        # Rendering
        self.fig = None
        self.ax = None
        # self.score=torch.zeros(self.num_agents, device=device)

    def copy(self):
        # Create a new instance with the same constructor arguments
        new_env = self.__class__(
            num_agents=self._num_agents,
            env_size=float(self.env_size),
            num_obstacles=self.num_obstacles,
            v_lin_max=float(self.v_lin_max),
            v_ang_max=float(self.v_ang_max),
            agent_radius=float(self.agent_radius),
            safe_dist=float(self.safe_dist),
            sens_range=float(self.sens_range),
            obstacle_size_min=float(self.obstacle_size_min),
            obstacle_size_max=float(self.obstacle_size_max),
            device=self.device,
            normalise=self.normalise,
        )

        # Clone dynamic tensors
        new_env.agent_pos = self.agent_pos.clone().detach()
        new_env.agent_dir = self.agent_dir.clone().detach()
        new_env.agent_vel_lin = self.agent_vel_lin.clone().detach()
        new_env.agent_vel_ang = self.agent_vel_ang.clone().detach()
        new_env.landmarks = self.landmarks.clone().detach()
        new_env.obstacle_pos = self.obstacle_pos.clone().detach()
        new_env.obstacle_radius = self.obstacle_radius.clone().detach()
        new_env.timestep = self.timestep

        # Optionally copy additional attributes if present
        if hasattr(self, "dv_lin_max"):
            new_env.dv_lin_max = self.dv_lin_max
        if hasattr(self, "dv_ang_max"):
            new_env.dv_ang_max = self.dv_ang_max
        if hasattr(self, "old_agent_vel_lin"):
            new_env.old_agent_vel_lin = self.old_agent_vel_lin.clone().detach()
        if hasattr(self, "old_agent_vel_ang"):
            new_env.old_agent_vel_ang = self.old_agent_vel_ang.clone().detach()
        if hasattr(self, "dones"):
            new_env.dones = self.dones.clone().detach()
        if hasattr(self, "covered"):
            new_env.covered = self.covered.clone().detach()
        if hasattr(self, "old_hungarian"):
            new_env.old_hungarian = self.old_hungarian.clone().detach()
        if hasattr(self, "static_state_tensor"):
            new_env.static_state_tensor = self.static_state_tensor.clone().detach()

        return new_env

    def delete(self):
        """
        Explicitly deletes all major tensors and fields to free GPU memory.
        Should be followed by: del env; torch.cuda.empty_cache()
        """
        tensor_attrs = [
            # Agent state
            "agent_pos", "agent_dir",
            "agent_vel_lin", "agent_vel_ang",
            "old_agent_vel_lin", "old_agent_vel_ang",

            # Environment state
            "landmarks", "obstacle_pos", "obstacle_radius",
            "static_state_tensor",

            # DoneAdj extensions
            "dones", "covered", "old_hungarian", "current_rewards",

            # Graph search fields if retained
            "last_dir", "prev_inx", "actions", "velocities",
        ]

        for attr in tensor_attrs:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, torch.Tensor):
                    setattr(self, attr, None)

        # Clean up rendering objects
        if hasattr(self, "fig") and self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        # Observation/action/state space dictionaries (not on GPU, but large)
        for attr in ["observation_spaces", "action_spaces"]:
            if hasattr(self, attr):
                setattr(self, attr, {})

        # Agent list
        for attr in ["agents", "possible_agents"]:
            if hasattr(self, attr):
                setattr(self, attr, [])

        # Reset timestep
        self.timestep = 0

    def _reset_episode(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.timestep = 0
        self._init_landmarks()  # fills self.landmarks
        self.agents = self.possible_agents[:]
        # Reinitialize all entities
        self._init_obstacles()  # fills self.obstacle_pos, self.obstacle_radius
        self._init_agents()  # fills self.agent_pos, self.agent_vel_lin, etc.
        # self.score=torch.zeros(self.num_agents, device=device)
        self._init_static_state_part()
        self._reset_hungarian()
        self.current_score = torch.zeros(self._num_agents, 9, device=device)

    def action_to_tensor(self, action):
        return torch.tensor([ 2*action[0]/self.v_lin_max - 1, action/self.v_ang_max])

    def _reset_hungarian(self):
        self.old_hungarian=self.get_hungarian_distances()

    def reset_tensor(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self._reset_episode(seed)
        state = self.state_tensor()
        observations=self.get_all_obs_tensor()
        return state, observations

    def reset(self, seed: Optional[int] = None, options=None) -> Dict[str, np.ndarray]:
        self._reset_episode()
        # Get observations (in CUDA, then detach+cpu if gym requires)
        observations = self.get_all_obs_dict()
        return observations

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],  # rewards
        Dict[str, bool],  # terminations
        Dict[str, bool],  # truncations
        Dict[str, dict]  # infos
    ]:
        # Convert actions (dict of np arrays) to a tensor on CUDA
        action_tensor = torch.stack([
            torch.as_tensor(actions[agent], device=self.device, dtype=torch.float32)
            for agent in self.agents
        ])

        self._make_step(action_tensor)
        # Generate new observations and rewards
        observations=self.get_all_obs_dict()
        rewards={agent_id:self.current_rewards[idx].item() for idx, agent_id in enumerate(self.agents)}
        terminations={ agent_id: self.get_dones_tensor()[idx].item() for idx, agent_id in enumerate(self.agents)}
        truncations={ agent_id: False for agent_id in self.agents}
        infos={ agent_id: {} for agent_id in self.agents}
        return observations, rewards, terminations, truncations, infos

    def step_tensor(self, actions_tensor: torch.Tensor) -> Tuple[
        torch.Tensor,  # state: [state_dim]
        torch.Tensor,  # observations: [N, obs_dim]
        torch.Tensor,  # rewards: [N]
        torch.Tensor  # dones: [N], dtype=bool
    ]:
        self._make_step(actions_tensor)
        state = self.state_tensor()
        observations = self.get_all_obs_tensor()

        dones = self.get_dones_tensor()
        return state, observations, self.current_rewards, dones

    def _make_step(self, action_tensor: torch.Tensor):
        """
        Applies the action tensor to update the environment state.

        Args:
            action_tensor (torch.Tensor): Tensor of shape [num_agents, 2],
                containing (dv_lin, dv_ang) for each agent on the same device.

        Returns:
            torch.Tensor: Per-agent reward vector of shape [num_agents],
                on the same device as the input.
        """
        # Apply actions, update movement, handle collisions
        self._apply_actions(action_tensor)  # dV_lin and dV_ang
        self._update_positions()  # uses self.agent_vel_lin, self.agent_dir
        # self._handle_collisions()  # modifies self.agent_pos if needed
        self.timestep += 1
        self._compute_rewards_tensor()
        self.current_score += self.current_rewards

    def get_all_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns per-agent observations as a dictionary (CPU numpy arrays)."""
        return {
            agent_id: self.get_observation(idx).detach().cpu().numpy()
            for idx, agent_id in enumerate(self.agents)
        }

    def get_all_obs_tensor(self) -> torch.Tensor:
        """Returns stacked per-agent observations as a CUDA tensor of shape [num_agents, obs_dim]."""
        return torch.stack([self.get_observation(idx) for idx in range(self._num_agents)], dim=0)


    def get_dones_tensor(self) -> torch.Tensor:
        """Returns done mask as a tensor of shape [num_agents], dtype=torch.bool."""
        return torch.full((self.num_agents,),False , dtype=torch.bool, device=self.device)

    def render(self) -> None:
        """
        Renders the environment using matplotlib:
        - Draws agents with direction arrows
        - Obstacles as gray circles
        - Landmarks as red crosses
        """
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        self.ax.set_xlim(0, self.env_size.item())
        self.ax.set_ylim(0, self.env_size.item())
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Step {self.timestep}")

        # Draw obstacles
        for i in range(self.num_obstacles):

            pos = self.obstacle_pos[i].detach().cpu().numpy()
            radius = self.obstacle_radius[i].item()
            circle = plt.Circle(pos, radius, color='gray', alpha=0.5)
            self.ax.add_patch(circle)

        # Draw landmarks
        for lm in self.landmarks.detach().cpu().numpy():
            self.ax.plot(lm[0], lm[1], 'rx', markersize=8)

        # Draw agents
        for i in range(self._num_agents):
            pos = self.agent_pos[i].detach().cpu().numpy()
            angle_deg = self.agent_dir[i].item()
            angle_rad = angle_deg

            circle = plt.Circle(pos, self.agent_radius.item(), color='blue', alpha=0.6)
            self.ax.add_patch(circle)

            dx = self.agent_radius.item() * np.cos(angle_rad)
            dy = self.agent_radius.item() * np.sin(angle_rad)
            self.ax.arrow(pos[0], pos[1], dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')

        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def _init_static_state_part(self):
        """
        Computes and stores the static part of the global state tensor.

        Includes:
            - Landmark positions: shape [num_landmarks, 2]
            - Obstacle positions and radii: shape [num_obstacles, 3]

        Normalization if self.normalise is True:
            - Positions in [-env_size/2, env_size/2] → divided by (env_size / 2)
            - Radii in [obstacle_size_min, obstacle_size_max] → min-max normalized

        Stores:
            self.static_state_tensor: 1D tensor of shape [2L + 3M], on self.device
        """
        device = self.device
        normalize = self.normalise

        # === 1. Landmarks ===
        lm_dists = torch.norm(self.landmarks, dim=1)
        self.lm_order = torch.argsort(lm_dists)
        lm_sorted = self.landmarks[self.lm_order]  # shape: [L, 2]

        if normalize:
            lm_sorted = lm_sorted / (self.env_size / 2)  # → [-1, 1]

        lm_flat = lm_sorted.flatten()

        # === 2. Obstacles ===
        ob_dists = torch.norm(self.obstacle_pos, dim=1)
        ob_order = torch.argsort(ob_dists)
        ob_pos_sorted = self.obstacle_pos[ob_order]  # shape: [M, 2]
        ob_radii_sorted = self.obstacle_radius[ob_order].unsqueeze(1)  # shape: [M, 1]

        if normalize:
            ob_pos_sorted = ob_pos_sorted / (self.env_size / 2)  # → [-1, 1]
            ob_radii_sorted = (ob_radii_sorted - self.obstacle_size_min) / (
                    self.obstacle_size_max - self.obstacle_size_min
            )  # → [0, 1]

        ob_state = torch.cat([ob_pos_sorted, ob_radii_sorted], dim=1).flatten()  # shape: [3M]

        # === Final static state ===
        self.static_state_tensor = torch.cat([lm_flat, ob_state], dim=0).to(torch.float32).to(device)

    def state_tensor(self) -> torch.Tensor:
        """
        Returns the full global state tensor for the current timestep.

        Structure (sorted by distance to origin for consistency):
            [landmarks (2L) | obstacles (3M) | agents (5N)]

        Agent fields:
            - position: (x, y)
            - direction: θ (as sin, cos if normalized)
            - linear velocity
            - angular velocity

        Normalization if self.normalise is True:
            - Positions: divided by (env_size / 2)
            - Velocities: divided by v_lin_max / v_ang_max
            - Angle: converted to sin(θ), cos(θ)

        Returns:
            torch.Tensor: 1D state vector of shape [2L + 3M + (5 or 6)×N], on self.device
        """
        device = self.device
        normalize = self.normalise

        # Sort agents by distance to origin
        ag_dists = torch.norm(self.agent_pos, dim=1)
        self.ag_order = torch.argsort(ag_dists)

        ag_pos = self.agent_pos[self.ag_order]  # (N, 2)
        ag_dir = self.agent_dir[self.ag_order]  # (N,)

        if normalize:
            ag_pos = ag_pos / (self.env_size / 2)  # → [-1, 1]
            ag_dir_sin = torch.sin(ag_dir).unsqueeze(1)
            ag_dir_cos = torch.cos(ag_dir).unsqueeze(1)
            ag_state = torch.cat([ag_pos, ag_dir_sin, ag_dir_cos], dim=1).flatten()  # (6N,)
        else:
            ag_dir = ag_dir.unsqueeze(1)
            ag_state = torch.cat([ag_pos, ag_dir], dim=1).flatten()  # (5N,)

        # Combine with static state
        return torch.cat([self.static_state_tensor, ag_state], dim=0).float().to(device)

    def state(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Global state as float32 numpy array on CPU.
        """
        return self.state_tensor().detach().cpu().numpy().astype(np.float32)


    def _init_agents(self):
        """
        Initializes agent positions, directions, and velocities while avoiding collisions
        with other agents and obstacles. All values are stored as CUDA tensors.
        """

        self.agent_pos = []
        self.agent_dir = []
        self.agent_vel_lin = []
        self.agent_vel_ang = []

        placed_positions = []

        for _ in range(self._num_agents):
            while True:
                half = self.env_size.item() / 2 - self.agent_radius.item()
                pos = np.random.uniform(-half, half, size=2)
                angle = np.random.uniform(-torch.pi, torch.pi, )

                # Check collision with other agents
                collision = False
                for other_pos in placed_positions:
                    if np.linalg.norm(pos - other_pos) < 2 * self.agent_radius.item():
                        collision = True
                        break

                # Check collision with obstacles
                if not collision and all(
                        np.linalg.norm(pos - ob_pos.detach().cpu().numpy()) > self.agent_radius.item() + ob_rad.item()
                        for ob_pos, ob_rad in zip(self.obstacle_pos, self.obstacle_radius)
                ):
                    break

            placed_positions.append(pos)
            self.agent_pos.append(torch.tensor(pos, dtype=torch.float32, device=self.device))
            self.agent_dir.append(torch.tensor(angle, dtype=torch.float32, device=self.device))
            self.agent_vel_lin.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))
            self.agent_vel_ang.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))

        self.agent_pos = torch.stack(self.agent_pos)
        self.agent_dir = torch.stack(self.agent_dir)
        self.agent_vel_lin = torch.stack(self.agent_vel_lin)
        self.agent_vel_ang = torch.stack(self.agent_vel_ang)

    def triangle_tree_layout(self):
        # Define the vertical spacing
        h = self.env_size*3 / 2  # Half height of square
        dy = h / 2  # Vertical distance between levels
        sqrt3 = torch.sqrt(torch.tensor(3.0))
        # Level 0 (top)
        top = torch.tensor([0.0, dy])

        # Level 1 (2 points)
        mid_left = torch.tensor([-dy / sqrt3, 0.0])
        mid_right = torch.tensor([dy / sqrt3, 0.0])

        # Level 2 (3 points)
        bottom_y = -dy
        bottom_radius = 2 * dy / sqrt3
        bottom_left = torch.tensor([-bottom_radius, bottom_y])
        bottom_center = torch.tensor([0.0, bottom_y])
        bottom_right = torch.tensor([bottom_radius, bottom_y])

        return torch.stack([top, mid_left, mid_right, bottom_left, bottom_center, bottom_right])

    def _init_landmarks(self):
        """
        Initializes landmark positions with minimum separation and transforms them
        into a new coordinate system centered at the landmarks' centroid and aligned
        with the principal axis of their spatial distribution.

        After this, all coordinates are assumed to be in the new system,
        and the original world frame is discarded.
        """
        landmarks = []
        attempts = 0
        max_attempts = 1000
        min_dist = 10 * self.agent_radius.item()

        while len(landmarks) < self.num_landmarks and attempts < max_attempts:
            candidate = (torch.rand(2, device=self.device) - 0.5) * self.env_size
            valid = True
            for existing in landmarks:
                if torch.norm(candidate - existing) < min_dist:
                    valid = False
                    break
            if valid:
                landmarks.append(candidate)
            attempts += 1

        if len(landmarks) < self.num_landmarks:
            raise RuntimeError("Failed to place all landmarks with minimum separation.")

        # Stack into a tensor (L, 2) — original frame
        landmarks_tensor =torch.stack(landmarks, dim=0)
        # landmarks_tensor = self.triangle_tree_layout().to(device=self.device)
        # Define new coordinate system:
        # Step 1: origin = centroid
        origin = landmarks_tensor.mean(dim=0)

        # Step 2: PCA for principal direction
        centered = landmarks_tensor - origin
        cov = centered.T @ centered
        eigvals, eigvecs = torch.linalg.eigh(cov)
        x_axis = eigvecs[:, -1] / torch.norm(eigvecs[:, -1])
        y_axis = torch.stack([-x_axis[1], x_axis[0]])
        rot_matrix = torch.stack([x_axis, y_axis])  # (2, 2)

        # Step 3: Transform landmarks into new global coordinate frame
        landmarks_aligned = centered @ rot_matrix.T  # (L, 2)

        # After PCA rotation, landmarks_aligned is centered at (0, 0)

        min_xy, _ = landmarks_aligned.min(dim=0)
        max_xy, _ = landmarks_aligned.max(dim=0)
        extent = torch.max(torch.abs(min_xy), torch.abs(max_xy))  # (x_max, y_max)

        # Scale to fit within bounds [-env_size/2, env_size/2] with margin
        margin = 5
        scale_factor = (self.env_size / 2 - margin) / extent
        scale = torch.min(scale_factor)

        landmarks_aligned = landmarks_aligned * scale

        # Save landmarks in new frame
        self.landmarks = landmarks_aligned

    def _init_obstacles(self):
        """
        Initializes obstacle positions and radii uniformly within environment bounds.
        Ensures obstacles do not overlap landmarks by at least (agent_radius + obstacle_radius).
        """
        self.obstacle_pos = torch.zeros((self.num_obstacles, 2), device=self.device)
        self.obstacle_radius = (
                torch.rand(self.num_obstacles, device=self.device) * (self.obstacle_size_max - self.obstacle_size_min)
                + self.obstacle_size_min
        )

        for i in range(self.num_obstacles):
            while True:
                # Sample a random position within the environment bounds
                pos = (torch.rand((2,), device=self.device) - 0.5) * self.env_size

                # Compute distances to all landmarks
                distances = torch.norm(self.landmarks - pos, dim=1)

                # Required minimum distance: agent radius + obstacle radius
                min_allowed_dist = 2*self.agent_radius + self.obstacle_radius[i]

                if torch.all(distances > min_allowed_dist):
                    self.obstacle_pos[i] = pos
                    break

    def _apply_actions(self, action_tensor: torch.Tensor):
        """
        Applies delta actions to agent velocities (linear and angular),
        with clamping based on environment constraints.
        Ignores agents that are marked done.
        """
        # Assumes action_tensor ∈ [-1, 1]
        self.old_agent_vel_lin = self.agent_vel_lin.clone().detach()
        self.old_agent_vel_ang = self.agent_vel_ang.clone().detach()
        action_tensor=action_tensor.to(device=self.device)
        active_agents = ~self.get_dones_tensor()  # Boolean mask

        # Only update active agents' velocities
        self.agent_vel_lin[active_agents] = 0.5 * (action_tensor[active_agents, 0] + 1) * self.v_lin_max
        self.agent_vel_ang[active_agents] = action_tensor[active_agents, 1] * self.v_ang_max

    def _update_positions(self):
        """
        Updates positions such that:
        - Agents stop exactly at touching point, no overlap.
        - They stop moving if touching occurs.
        - Allows free rotation always.
        """
        device = self.device
        active_agents = ~self.get_dones_tensor()

        theta_rad = self.agent_dir[active_agents]
        dx = self.agent_vel_lin[active_agents] * torch.cos(theta_rad)
        dy = self.agent_vel_lin[active_agents] * torch.sin(theta_rad)
        delta = torch.stack([dx, dy], dim=1)
        current_pos = self.agent_pos[active_agents]
        target_pos = current_pos + delta
        allowed_pos = target_pos.clone()



        # Function to limit motion up to touching
        def project_to_touch(pos, center, radius_sum):
            vec = pos - center
            dist = torch.norm(vec, dim=-1, keepdim=True)
            vec_norm = vec / dist.clamp(min=1e-8)
            touch_point = center + vec_norm * radius_sum
            needs_project = dist.squeeze(-1) < radius_sum - 1e-6
            return torch.where(needs_project.unsqueeze(-1), touch_point, pos)

        # # Bounds (walls)
        # allowed_pos[:, 0] = torch.clamp(allowed_pos[:, 0], min=min_bound, max=max_bound)
        # allowed_pos[:, 1] = torch.clamp(allowed_pos[:, 1], min=min_bound, max=max_bound)

        # Obstacles
        for ob_pos, ob_r in zip(self.obstacle_pos.to(device), self.obstacle_radius.to(device)):
            allowed_pos = project_to_touch(allowed_pos, ob_pos.unsqueeze(0), self.agent_radius + ob_r)

        # Done agents
        done_pos = self.agent_pos[self.get_dones_tensor()]
        for da in done_pos:
            allowed_pos = project_to_touch(allowed_pos, da.unsqueeze(0), 2 * self.agent_radius)

        # Active vs Active (pairwise check)
        for i in range(allowed_pos.shape[0]):
            for j in range(i + 1, allowed_pos.shape[0]):
                center_i, center_j = allowed_pos[i], allowed_pos[j]
                vec = center_i - center_j
                dist = vec.norm()
                if dist < 2 * self.agent_radius - 1e-6:
                    norm_vec = vec / (dist + 1e-8)
                    midpoint = (center_i + center_j) / 2
                    allowed_pos[i] = midpoint + norm_vec * self.agent_radius
                    allowed_pos[j] = midpoint - norm_vec * self.agent_radius

        # If position was adjusted, stop linear motion
        position_changed = (allowed_pos - target_pos).norm(dim=1) > 1e-6
        self.agent_vel_lin[active_agents] = torch.where(
            position_changed,
            torch.zeros_like(self.agent_vel_lin[active_agents]),
            self.agent_vel_lin[active_agents]
        )

        self.agent_pos[active_agents] = allowed_pos

        # Always allow rotation
        self.agent_dir[active_agents] = (self.agent_dir[active_agents] + self.agent_vel_ang[
            active_agents] + torch.pi) % (2 * torch.pi) - torch.pi

    def get_observation(self, idx: int) -> torch.Tensor:
        device = self.device
        normalize = self.normalise

        pos = self.agent_pos[idx].to(device)  # (2,)
        heading = self.agent_dir[idx].to(device)

        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        rot = torch.stack([
            torch.stack([cos_h, sin_h]),
            torch.stack([-sin_h, cos_h])
        ])  # (2, 2)

        # Other agents
        mask = torch.arange(self._num_agents, device=device) != idx
        other_pos = self.agent_pos[mask]
        rel_pos_global = other_pos - pos
        dist_to_agents = torch.norm(rel_pos_global, dim=1)
        sorted_idx = torch.argsort(dist_to_agents)

        rel_pos = (rel_pos_global[sorted_idx]) @ rot.T
        rel_dir = self.agent_dir[mask][sorted_idx] - heading
        rel_dir = torch.atan2(torch.sin(rel_dir), torch.cos(rel_dir))

        if normalize:
            rel_pos = rel_pos / self.env_size
            rel_dir_sin = torch.sin(rel_dir).unsqueeze(1)
            rel_dir_cos = torch.cos(rel_dir).unsqueeze(1)
        else:
            rel_dir = rel_dir.unsqueeze(1)

        # Landmarks
        rel_lm_global = self.landmarks - pos
        dist_to_lm = torch.norm(rel_lm_global, dim=1)
        sorted_lm_idx = torch.argsort(dist_to_lm)

        rel_lm = (rel_lm_global[sorted_lm_idx]) @ rot.T
        if normalize:
            rel_lm = rel_lm / self.env_size

        # Obstacles
        obs_vec = self.obstacle_pos - pos
        center_dists = torch.norm(obs_vec, dim=1)
        edge_dists = center_dists - self.obstacle_radius
        in_range = edge_dists < self.sens_range
        obs_idx = torch.argsort(edge_dists)
        obs_idx = obs_idx[in_range[obs_idx]]

        edge_dists = edge_dists[obs_idx]
        if normalize:
            edge_dists = edge_dists / self.sens_range

        obs_vec_local = obs_vec[obs_idx] @ rot.T
        obs_angles = torch.atan2(obs_vec_local[:, 1], obs_vec_local[:, 0])

        if normalize:
            obs_angle_sin = torch.sin(obs_angles)
            obs_angle_cos = torch.cos(obs_angles)

        # Pad obstacles
        pad_len = self.num_obstacles - len(obs_idx)

        def pad(x, value=0):
            if pad_len > 0:
                return torch.cat([x, torch.full((pad_len,), value, device=device)], dim=0)
            return x

        edge_dists = pad(edge_dists)
        if normalize:
            obs_angle_sin = pad(obs_angle_sin)
            obs_angle_cos = pad(obs_angle_cos)
        else:
            obs_angles = pad(obs_angles)

        # Final concat
        components = [rel_pos.flatten()]

        if normalize:
            components.extend([rel_dir_sin.flatten(), rel_dir_cos.flatten()])
        else:
            components.append(rel_dir.flatten())

        components.extend([
            rel_lm.flatten(),
            edge_dists,
        ])

        if normalize:
            components.extend([obs_angle_sin, obs_angle_cos])
        else:
            components.append(obs_angles)

        return torch.cat(components, dim=0).float().to(device)

    def _compute_rewards_tensor(self) -> torch.Tensor:
        """
        Computes rewards for all agents based on:
          - Normalized distance to assigned landmarks (Hungarian matching)
          - Local penalties for proximity to agents and obstacles (within safe_dist)
          - Local stopping reward when agent is centered on its assigned landmark (within agent_radius)

        Returns:
            torch.Tensor: A 1D tensor of shape (num_agents,) representing individual rewards.
        """

        N = self._num_agents
        device = self.device
        components = torch.zeros(N, 9, device=device)

        # --- Hungarian assignment (global goal reward) ---
        new_hungarian=self.get_hungarian_distances()
        d_global=-new_hungarian.mean()/self.env_size
        d_goal = -new_hungarian/self.env_size
        # d_penalty=-self.distance_penalty_scale* torch.log1p(new_hungarian / self.env_size)


        # Exponential goal reward (tau=1)
        # rewards = d_goal+d_global


        progressive = (self.old_hungarian - new_hungarian)/self.v_lin_max
        # rewards+=progressive
        components[0] = progressive  # progressive
        components[ 1] = d_goal  # dist-to-goal
        components[ 2] = d_global  # global penalty (shared to all active)

        # --- Local stopping reward (if within agent_radius of assigned landmark) ---
        stop_dist=self.agent_radius+self.v_lin_max

        inside_landmark = new_hungarian < stop_dist
        stop_bonus =  inside_landmark.float()/new_hungarian.clamp(min=epsilon)
        # rewards += stop_bonus
        components[3] = stop_bonus
        # --- Agent–Agent penalty (within safe_dist) ---
        delta = self.agent_pos.unsqueeze(1) - self.agent_pos.unsqueeze(0)  # (N, N, 2)
        dist_matrix = torch.norm(delta, dim=2) - 2 * self.agent_radius  # (N, N)
        aa_mask = (dist_matrix < self.safe_dist) & (~torch.eye(N, dtype=torch.bool, device=device))

        # Inverse distance penalty, clamped to avoid division by zero
        inv_dist_penalty = torch.zeros_like(dist_matrix)
        inv_dist_penalty[aa_mask] = 1.0 / dist_matrix[aa_mask].clamp(min=epsilon)

        penalty = inv_dist_penalty.sum(dim=1)
        # rewards -= penalty
        components[ 4] = -penalty
        # --- Agent–Obstacle penalty (within safe_dist) ---
        ap = self.agent_pos.unsqueeze(1)  # (N, 1, 2)
        ob = self.obstacle_pos.unsqueeze(0)  # (1, M, 2)
        dist_ap_ob = torch.norm(ap - ob, dim=2)  # (N, M)

        effective_dist = dist_ap_ob - self.obstacle_radius.unsqueeze(0) - self.agent_radius  # (N, M)
        ob_mask = effective_dist < self.safe_dist

        inv_dist_penalty = torch.zeros_like(effective_dist)
        inv_dist_penalty[ob_mask] = 1.0 / effective_dist[ob_mask].clamp(min=epsilon)

        penalty =  inv_dist_penalty.sum(dim=1)
        components[ 5] = -penalty

        #_____________________________________VELOCIT REWARD_____________________________
        # Linear velocity reward (encourage moving forward)
        lin_penalty = -(1.0 - (self.agent_vel_lin / self.v_lin_max))
        # Angular velocity penalty (discourage sharp turns)
        ang_penalty = -self.agent_vel_ang.abs() / self.v_ang_max

        # Combine with inherited rewards
        components[6] = lin_penalty
        components[7] = ang_penalty
        components[8] = 0

        # rewards /= scalar_for_sensitivity
        self.current_rewards = components
        self.old_hungarian=new_hungarian
        return components

    def get_hungarian_distances(self) -> torch.Tensor:
        """
        Returns the assigned distances between agents and landmarks using the Hungarian algorithm.
        Output is per-agent (not averaged).

        Returns:
            torch.Tensor: Shape [num_agents], distances after optimal assignment.
        """
        with torch.no_grad():
            device = self.device
            agent_pos = self.agent_pos.detach().cpu()
            landmarks = self.landmarks.detach().cpu()

            cost_matrix = torch.cdist(agent_pos, landmarks)  # (N, N)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_dists = cost_matrix[row_ind, col_ind]  # shape: (N,)

            return torch.tensor(assigned_dists, dtype=torch.float32, device=device)

    def graph_search_cuda(self, trajectory, init_heading):
        device = self.device
        N = trajectory.size(0)

        steps_counts = torch.full((N,), float('inf'), device=device)
        dists = torch.full((N,), float('inf'), device=device)
        prev_inx = torch.full((N,), -1, dtype=torch.long, device=device)
        last_dir = torch.full((N,), float('nan'), device=device)
        actions = torch.full((N, 2), float('nan'), device=device)  # [distance, heading change]

        steps_counts[0] = 0
        dists[0] = 0.0
        last_dir[0] = init_heading
        actions[0] = torch.tensor([0.0, 0.0], device=device)

        visited = torch.zeros(N, dtype=torch.bool, device=device)
        visited[0] = True
        current_front = torch.tensor([0], dtype=torch.long, device=device)

        while current_front.numel() > 0:
            current_steps = steps_counts[current_front]
            current_pos = trajectory[current_front]
            current_heading = last_dir[current_front]

            vectors = trajectory.unsqueeze(0) - current_pos.unsqueeze(1)  # [batch, N, 2]
            distances = torch.norm(vectors, dim=2)  # [batch, N]
            headings = torch.atan2(vectors[..., 1], vectors[..., 0])  # [batch, N]
            heading_diff = (headings - current_heading.unsqueeze(1) + torch.pi) % (
                        2 * torch.pi) - torch.pi  # [batch, N]

            linear_mask = (distances <= self.v_lin_max) & (distances > 0)
            angular_mask = heading_diff.abs() <= self.v_ang_max
            reachable_mask = linear_mask & angular_mask & ~visited.unsqueeze(0)

            batch_idx, point_idx = reachable_mask.nonzero(as_tuple=True)
            step_dists = distances[batch_idx, point_idx]
            angle_diffs = heading_diff[batch_idx, point_idx]
            cumulative_steps = current_steps[batch_idx] + 1
            cumulative_dists = dists[current_front[batch_idx]] + step_dists

            better = (cumulative_steps < steps_counts[point_idx]) | (
                    (cumulative_steps == steps_counts[point_idx]) & (cumulative_dists < dists[point_idx])
            )

            updates = point_idx[better]
            steps_counts[updates] = cumulative_steps[better]
            dists[updates] = cumulative_dists[better]
            prev_inx[updates] = current_front[batch_idx[better]]
            last_dir[updates] = headings[batch_idx[better], point_idx[better]]
            actions[updates] = torch.stack([step_dists[better], angle_diffs[better]], dim=1)
            visited[updates] = True

            current_front = updates.unique()

        max_steps = steps_counts.max()
        furthest_candidates = (steps_counts == max_steps).nonzero(as_tuple=True)[0]
        furthest_idx = dists[furthest_candidates].argmax().item()
        furthest_idx = furthest_candidates[furthest_idx].item()

        return steps_counts, dists, prev_inx, last_dir, actions, furthest_idx


class DiffDriveParallelEnvAdj(DiffDriveParallelEnv):
    def __init__(
            self,
            num_agents: int = num_agents,
            env_size: float = env_size,
            num_obstacles: int = num_obstacles,
            v_lin_max: float = v_lin_max,
            v_ang_max: float = v_ang_max,
            dv_lin_max: float = dv_lin_max,
            dv_ang_max: float = dv_ang_max,
            agent_radius: float = agent_radius,
            safe_dist: float = safe_dist,
            sens_range: float = sens_range,
            obstacle_size_min: float = obstacle_size_min,
            obstacle_size_max: float = obstacle_size_max,
            device: Union[str, torch.device] = device,
            normalise=normalise,

    ):
        self.dv_lin_max = dv_lin_max
        self.dv_ang_max = dv_ang_max
        # self.velocity_reward_scale=velocity_reward_scale
        super().__init__(
            num_agents=num_agents,
            env_size=env_size,
            num_obstacles=num_obstacles,
            v_lin_max=v_lin_max,
            v_ang_max=v_ang_max,
            agent_radius=agent_radius,
            safe_dist=safe_dist,
            sens_range=sens_range,
            obstacle_size_min=obstacle_size_min,
            obstacle_size_max=obstacle_size_max,
            device=device,
            normalise=normalise,
        )

        if normalise:
            self.obs_dim = 3 * self.num_obstacles + 2 * self.num_landmarks + (self.num_agents - 1) * 6 + self.action_dim
            self.state_dim = 6 * self.num_agents + 2 * self.num_landmarks + 3 * self.num_obstacles
        else:
            self.obs_dim=2*self.num_obstacles+2*self.num_landmarks+(self.num_agents-1)*5+self.action_dim
            self.state_dim=5*self.num_agents+2*self.num_landmarks+3*self.num_obstacles

    def action_to_tensor(self, action):
        return torch.tensor([ action[0]/self.dv_lin_max, action[1]/self.dv_ang_max])
    def _apply_actions(self, action_tensor: torch.Tensor):
        """
        Applies delta actions to agent velocities (linear and angular),
        with clamping based on environment constraints.
        Ignores agents marked as done.
        """
        self.old_agent_vel_lin = self.agent_vel_lin.clone().detach()
        self.old_agent_vel_ang = self.agent_vel_ang.clone().detach()

        active_agents = ~self.get_dones_tensor()  # Boolean mask

        # Only apply actions to active agents
        dv_lin = torch.zeros_like(self.agent_vel_lin)
        dv_ang = torch.zeros_like(self.agent_vel_ang)

        dv_lin[active_agents] = action_tensor[active_agents, 0] * float(self.dv_lin_max)
        dv_ang[active_agents] = action_tensor[active_agents, 1] * float(self.dv_ang_max)

        # Update velocities only for active agents
        self.agent_vel_lin[active_agents] = (self.agent_vel_lin[active_agents] + dv_lin[active_agents]).clamp(
            0.0, float(self.v_lin_max)
        )
        self.agent_vel_ang[active_agents] = (self.agent_vel_ang[active_agents] + dv_ang[active_agents]).clamp(
            -float(self.v_ang_max), float(self.v_ang_max)
        )

    def get_observation(self, idx: int) -> torch.Tensor:
        device = self.device
        normalize = self.normalise

        pos = self.agent_pos[idx].to(device)  # (2,)
        heading = self.agent_dir[idx].to(device)

        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        rot = torch.stack([
            torch.stack([cos_h, sin_h]),
            torch.stack([-sin_h, cos_h])
        ])  # (2, 2)

        # Own motion
        own_lin = self.agent_vel_lin[idx].unsqueeze(0).to(device)
        own_ang = self.agent_vel_ang[idx].unsqueeze(0).to(device)
        if normalize:
            own_lin = own_lin / self.v_lin_max
            own_ang = own_ang / self.v_ang_max

        # Other agents
        mask = torch.arange(self._num_agents, device=device) != idx
        other_pos = self.agent_pos[mask]
        rel_pos_global = other_pos - pos
        dist_to_agents = torch.norm(rel_pos_global, dim=1)
        sorted_idx = torch.argsort(dist_to_agents)

        rel_pos = (rel_pos_global[sorted_idx]) @ rot.T
        rel_dir = self.agent_dir[mask][sorted_idx] - heading
        rel_dir = torch.atan2(torch.sin(rel_dir), torch.cos(rel_dir))

        if normalize:
            rel_pos = rel_pos / self.env_size
            rel_dir_sin = torch.sin(rel_dir).unsqueeze(1)
            rel_dir_cos = torch.cos(rel_dir).unsqueeze(1)
        else:
            rel_dir = rel_dir.unsqueeze(1)

        lin_vels = self.agent_vel_lin[mask][sorted_idx].unsqueeze(1).to(device)
        ang_vels = self.agent_vel_ang[mask][sorted_idx].unsqueeze(1).to(device)
        if normalize:
            lin_vels = lin_vels / self.v_lin_max
            ang_vels = ang_vels / self.v_ang_max

        # Landmarks
        rel_lm_global = self.landmarks - pos
        dist_to_lm = torch.norm(rel_lm_global, dim=1)
        sorted_lm_idx = torch.argsort(dist_to_lm)

        rel_lm = (rel_lm_global[sorted_lm_idx]) @ rot.T
        if normalize:
            rel_lm = rel_lm / self.env_size

        # Obstacles
        obs_vec = self.obstacle_pos - pos
        center_dists = torch.norm(obs_vec, dim=1)
        edge_dists = center_dists - self.obstacle_radius
        in_range = edge_dists < self.sens_range
        obs_idx = torch.argsort(edge_dists)
        obs_idx = obs_idx[in_range[obs_idx]]

        edge_dists = edge_dists[obs_idx]
        if normalize:
            edge_dists = edge_dists / self.sens_range

        obs_vec_local = obs_vec[obs_idx] @ rot.T
        obs_angles = torch.atan2(obs_vec_local[:, 1], obs_vec_local[:, 0])

        if normalize:
            obs_angle_sin = torch.sin(obs_angles)
            obs_angle_cos = torch.cos(obs_angles)

        # Pad obstacles
        pad_len = self.num_obstacles - len(obs_idx)

        def pad(x, value=0):
            if pad_len > 0:
                return torch.cat([x, torch.full((pad_len,), value, device=device)], dim=0)
            return x

        edge_dists = pad(edge_dists)
        if normalize:
            obs_angle_sin = pad(obs_angle_sin)
            obs_angle_cos = pad(obs_angle_cos)
        else:
            obs_angles = pad(obs_angles)

        # Final concat
        components = [
            own_lin, own_ang,
            rel_pos.flatten(),
        ]

        if normalize:
            components.extend([rel_dir_sin.flatten(), rel_dir_cos.flatten()])
        else:
            components.append(rel_dir.flatten())

        components.extend([
            lin_vels.flatten(),
            ang_vels.flatten(),
            rel_lm.flatten(),
            edge_dists,
        ])

        if normalize:
            components.extend([obs_angle_sin, obs_angle_cos])
        else:
            components.append(obs_angles)

        return torch.cat(components, dim=0).float().to(device)

    def state_tensor(self) -> torch.Tensor:
        """
        Returns the full global state tensor for the current timestep.

        Structure (sorted by distance to origin for consistency):
            [landmarks (2L) | obstacles (3M) | agents (5N)]

        Agent fields:
            - position: (x, y)
            - direction: θ (as sin, cos if normalized)
            - linear velocity
            - angular velocity

        Normalization if self.normalise is True:
            - Positions: divided by (env_size / 2)
            - Velocities: divided by v_lin_max / v_ang_max
            - Angle: converted to sin(θ), cos(θ)

        Returns:
            torch.Tensor: 1D state vector of shape [2L + 3M + (5 or 6)×N], on self.device
        """
        device = self.device
        normalize = self.normalise

        # Sort agents by distance to origin
        ag_dists = torch.norm(self.agent_pos, dim=1)
        self.ag_order = torch.argsort(ag_dists)

        ag_pos = self.agent_pos[self.ag_order]  # (N, 2)
        ag_dir = self.agent_dir[self.ag_order]  # (N,)
        ag_vlin = self.agent_vel_lin[self.ag_order].unsqueeze(1)  # (N, 1)
        ag_vang = self.agent_vel_ang[self.ag_order].unsqueeze(1)  # (N, 1)

        if normalize:
            ag_pos = ag_pos / (self.env_size / 2)  # → [-1, 1]
            ag_dir_sin = torch.sin(ag_dir).unsqueeze(1)
            ag_dir_cos = torch.cos(ag_dir).unsqueeze(1)
            ag_vlin = ag_vlin / self.v_lin_max
            ag_vang = ag_vang / self.v_ang_max
            ag_state = torch.cat([ag_pos, ag_dir_sin, ag_dir_cos, ag_vlin, ag_vang], dim=1).flatten()  # (6N,)
        else:
            ag_dir = ag_dir.unsqueeze(1)
            ag_state = torch.cat([ag_pos, ag_dir, ag_vlin, ag_vang], dim=1).flatten()  # (5N,)

        # Combine with static state
        return torch.cat([self.static_state_tensor, ag_state], dim=0).float().to(device)

    def _reset_episode(self, seed: Optional[int] = None) -> None:
        super()._reset_episode(seed)
        # Set initial random linear velocity in [0, v_lin_max]
        self.agent_vel_lin = torch.rand(self._num_agents, device=self.device) * self.v_lin_max

        # Set initial random angular velocity in [-v_ang_max, v_ang_max]
        self.agent_vel_ang = (torch.rand(self._num_agents, device=self.device) * 2 - 1) * self.v_ang_max

    def graph_search_cuda(self, trajectory, init_heading):
        device = self.device
        N = trajectory.shape[0]

        steps_counts = torch.full((N,), float('inf'), device=device)
        dists = torch.full((N,), float('inf'), device=device)
        prev_inx = torch.full((N,), -1, dtype=torch.long, device=device)
        last_dir = torch.full((N,), float('nan'), device=device)
        actions = torch.full((N, 2), float('nan'), device=device)
        velocities = torch.full((N, 2), float('nan'), device=device)

        steps_counts[0] = 0
        dists[0] = 0.0
        last_dir[0] = init_heading
        velocities[0] = torch.tensor([0.0, 0.0], device=device)
        actions[0] = torch.tensor([0.0, 0.0], device=device)

        visited = torch.zeros(N, dtype=torch.bool, device=device)
        visited[0] = True

        current_front = torch.tensor([0], dtype=torch.long, device=device)
        current_velocities = torch.zeros((1, 2), device=device)  # [lin, ang]

        dv_lins = torch.tensor([self.dv_lin_max, -self.dv_lin_max, 0.0], device=device)
        dv_angs = torch.tensor([self.dv_ang_max, -self.dv_ang_max, 0.0], device=device)
        candidates = torch.cartesian_prod(dv_lins, dv_angs)  # [K, 2]

        while current_front.numel() > 0:
            current_steps = steps_counts[current_front]
            current_pos = trajectory[current_front]
            current_heading = last_dir[current_front]
            current_vel_lin = current_velocities[:, 0]
            current_vel_ang = current_velocities[:, 1]

            # All combinations of deltas applied to current velocities
            new_lin = (current_vel_lin.unsqueeze(1) + candidates[:, 0]).clamp(0.0, self.v_lin_max)
            new_ang = (current_vel_ang.unsqueeze(1) + candidates[:, 1]).clamp(-self.v_ang_max, self.v_ang_max)

            mask_non_zero = (new_lin.abs() > epsilon) | (new_ang.abs() > epsilon)

            new_lin = new_lin[mask_non_zero]
            new_ang = new_ang[mask_non_zero]

            repeat_pos = current_pos.repeat_interleave(new_lin.size(0), dim=0)
            repeat_heading = current_heading.repeat_interleave(new_lin.size(0), dim=0)

            heading_after = repeat_heading + new_ang
            dx = new_lin * torch.cos(heading_after)
            dy = new_lin * torch.sin(heading_after)
            new_pos = repeat_pos + torch.stack([dx, dy], dim=1)

            distances = torch.cdist(new_pos.unsqueeze(0), trajectory.unsqueeze(0)).squeeze(0)
            reachable_mask = distances <= (self.dv_lin_max + epsilon)
            point_idx = reachable_mask.nonzero(as_tuple=True)[1]
            step_dists = distances[:, point_idx]

            cumulative_steps = current_steps.repeat_interleave(new_lin.size(0)) + 1
            cumulative_dists = dists[current_front.repeat_interleave(new_lin.size(0))] + step_dists

            better = (cumulative_steps < steps_counts[point_idx]) | (
                    (cumulative_steps == steps_counts[point_idx]) & (cumulative_dists < dists[point_idx])
            )

            updates = point_idx[better]
            steps_counts[updates] = cumulative_steps[better]
            dists[updates] = cumulative_dists[better]
            prev_inx[updates] = current_front.repeat_interleave(new_lin.size(0))[better]
            last_dir[updates] = heading_after[better]
            velocities[updates] = torch.stack([new_lin[better], new_ang[better]], dim=1)
            actions[updates] = candidates[mask_non_zero][better]
            visited[updates] = True

            current_front = updates.unique()
            current_velocities = velocities[current_front]

        max_steps = steps_counts.max()
        furthest_candidates = (steps_counts == max_steps).nonzero(as_tuple=True)[0]
        furthest_idx = dists[furthest_candidates].argmax().item()
        furthest_idx = furthest_candidates[furthest_idx].item()

        return steps_counts, dists, prev_inx, last_dir, actions, furthest_idx


class DiffDriveParallelEnvDone(DiffDriveParallelEnv):
    def __init__(
            self,
            num_agents: int = num_agents,
            env_size: float = env_size,
            num_obstacles: int = num_obstacles,
            v_lin_max: float = v_lin_max,
            v_ang_max: float = v_ang_max,
            agent_radius: float = agent_radius,
            safe_dist: float = safe_dist,
            sens_range: float = sens_range,
            obstacle_size_min: float = obstacle_size_min,
            obstacle_size_max: float = obstacle_size_max,
            device: Union[str, torch.device] = device,
            normalise=normalise,

    ):
        super().__init__(
            num_agents=num_agents,
            env_size=env_size,
            num_obstacles=num_obstacles,
            v_lin_max=v_lin_max,
            v_ang_max=v_ang_max,
            agent_radius=agent_radius,
            safe_dist=safe_dist,
            sens_range=sens_range,
            obstacle_size_min=obstacle_size_min,
            obstacle_size_max=obstacle_size_max,
            device=device,
            normalise=normalise,


        )
        if normalise:
            self.obs_dim = 3 * self.num_obstacles + 3 * self.num_landmarks + (self.num_agents - 1) * 5
            self.state_dim = 5 * self.num_agents + 3 * self.num_landmarks + 3 * self.num_obstacles
        else:
            self.obs_dim=2*self.num_obstacles+3*self.num_landmarks+(self.num_agents-1)*4+self.action_dim
            self.state_dim=4*self.num_agents+3*self.num_landmarks+3*self.num_obstacles
    def _reset_episode(self, seed: Optional[int] = None) -> None:
        self.dones = torch.full((self.num_agents,), False, dtype=torch.bool, device=self.device)
        self.covered = torch.full((self.num_landmarks,), False, dtype=torch.bool, device=self.device)
        super()._reset_episode(seed)
    def _reset_hungarian(self):
        self.old_hungarian, _, _, _ = self.get_hungarian_distances()

    def get_dones_tensor(self) -> torch.Tensor:
        """Returns done mask as a tensor of shape [num_agents], dtype=torch.bool."""
        return self.dones
    def state_tensor(self) -> torch.Tensor:
        """
        Returns the full global state tensor for the current timestep.

        Structure (sorted by distance to origin for consistency):
            [landmarks (2L) | obstacles (3M) | agents (5N)]

        Agent fields:
            - position: (x, y)
            - direction: θ (as sin, cos if normalized)
            - linear velocity
            - angular velocity

        Normalization if self.normalise is True:
            - Positions: divided by (env_size / 2)
            - Velocities: divided by v_lin_max / v_ang_max
            - Angle: converted to sin(θ), cos(θ)

        Returns:
            torch.Tensor: 1D state vector of shape [2L + 3M + (5 or 6)×N], on self.device
        """


        # Combine with static state
        device = self.device
        super_state=super().state_tensor()
        return torch.cat([
            super_state,
            self.dones[self.ag_order].to(device).float(),
            self.covered[self.lm_order].to(device).float()
        ], dim=0).float().to(device)

    def get_observation(self, idx: int) -> torch.Tensor:
        device = self.device
        normalize = self.normalise

        pos = self.agent_pos[idx].to(device)  # (2,)
        heading = self.agent_dir[idx].to(device)

        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        rot = torch.stack([
            torch.stack([cos_h, sin_h]),
            torch.stack([-sin_h, cos_h])
        ])  # (2, 2)

        # Other agents
        mask = torch.arange(self._num_agents, device=device) != idx
        other_pos = self.agent_pos[mask]
        rel_pos_global = other_pos - pos
        dist_to_agents = torch.norm(rel_pos_global, dim=1)
        sorted_idx = torch.argsort(dist_to_agents)

        rel_pos = (rel_pos_global[sorted_idx]) @ rot.T
        rel_dir = self.agent_dir[mask][sorted_idx] - heading
        rel_dir = torch.atan2(torch.sin(rel_dir), torch.cos(rel_dir))

        if normalize:
            rel_pos = rel_pos / self.env_size
            rel_dir_sin = torch.sin(rel_dir).unsqueeze(1)
            rel_dir_cos = torch.cos(rel_dir).unsqueeze(1)
        else:
            rel_dir = rel_dir.unsqueeze(1)
        dones=self.dones[mask][sorted_idx]

        # Landmarks
        rel_lm_global = self.landmarks - pos
        dist_to_lm = torch.norm(rel_lm_global, dim=1)
        sorted_lm_idx = torch.argsort(dist_to_lm)

        rel_lm = (rel_lm_global[sorted_lm_idx]) @ rot.T
        if normalize:
            rel_lm = rel_lm / self.env_size
        covered=self.covered[sorted_lm_idx]

        # Obstacles
        obs_vec = self.obstacle_pos - pos
        center_dists = torch.norm(obs_vec, dim=1)
        edge_dists = center_dists - self.obstacle_radius - self.agent_radius
        in_range = edge_dists < self.sens_range
        obs_idx = torch.argsort(edge_dists)
        obs_idx = obs_idx[in_range[obs_idx]]

        edge_dists = edge_dists[obs_idx]
        if normalize:
            edge_dists = edge_dists / self.sens_range

        obs_vec_local = obs_vec[obs_idx] @ rot.T
        obs_angles = torch.atan2(obs_vec_local[:, 1], obs_vec_local[:, 0])

        if normalize:
            obs_angle_sin = torch.sin(obs_angles)
            obs_angle_cos = torch.cos(obs_angles)

        # Pad obstacles
        pad_len = self.num_obstacles - len(obs_idx)

        def pad(x, value=0):
            if pad_len > 0:
                return torch.cat([x, torch.full((pad_len,), value, device=device)], dim=0)
            return x

        edge_dists = pad(edge_dists)
        if normalize:
            obs_angle_sin = pad(obs_angle_sin)
            obs_angle_cos = pad(obs_angle_cos)
        else:
            obs_angles = pad(obs_angles)

        # Final concat
        components = [
            rel_pos.flatten(),
            dones.flatten(),
            # self.dones[idx].unsqueeze(0)
        ]

        if normalize:
            components.extend([rel_dir_sin.flatten(), rel_dir_cos.flatten()])
        else:
            components.append(rel_dir.flatten())

        components.extend([
            rel_lm.flatten(),
            covered,
            edge_dists,
        ])

        if normalize:
            components.extend([obs_angle_sin, obs_angle_cos])
        else:
            components.append(obs_angles)

        return torch.cat(components, dim=0).float().to(device)

    def _compute_rewards_tensor(self) -> torch.Tensor:
        """
        Computes rewards for all agents:
          - Global progress via Hungarian assignment (active agents only)
          - Penalty for active-agent proximity
          - Penalty for active-agent vs obstacle proximity
          - Penalty for active-agent vs inactive-agent proximity (done agents are static obstacles)
          - Reward for covering landmarks (reached_goal_scale)

        Updates:
            self.dones
            self.covered
            self.current_rewards
            self.old_hungarian

        Returns:
            torch.Tensor: [num_agents] individual rewards
        """
        N = self._num_agents
        device = self.device
        # rewards = torch.zeros(N, device=device)
        components = torch.zeros(N, 9, device=device)

        active_agents = ~self.dones
        active_indices = active_agents.nonzero(as_tuple=True)[0]
        active_agent_pos = self.agent_pos[active_agents]

        # ---------- Global Goal Reward (Hungarian) ----------
        new_hungarian, covered_agent_indices, covered_landmark_indices, assigned_landmark_indices = self.get_hungarian_distances()

        d_global = -new_hungarian.sum() / (self.env_size * self.num_agents)
        d_goal = -new_hungarian / self.env_size
        d_goal = torch.nan_to_num(d_goal, nan=0.0)

        progressive = torch.nan_to_num(
            (self.old_hungarian - new_hungarian) / self.v_lin_max, nan=0.0)
        progressive = torch.clamp(progressive, min=-1.0, max=1.0)
        # rewards[active_agents] =  progressive[active_agents]+d_global+d_goal[active_agents]
        components[active_agents, 0] = progressive[active_agents]  # progressive
        components[active_agents, 1] = d_goal[active_agents]  # dist-to-goal
        components[active_agents, 2] = d_global  # global penalty (shared to all active)

        # ---------- Reward for Newly Covered Landmarks ----------
        # rewards[covered_agent_indices] += scalar_for_sensitivity*self.reached_goal_scale
        # components[covered_agent_indices, 3] = 1
        components[covered_agent_indices, 3] = 1
        # self.dones[covered_agent_indices] = True
        self.covered[covered_landmark_indices] = True

        # ---------- Active-Agent vs Active-Agent Penalty ----------
        if active_agent_pos.shape[0] > 1:
            delta = active_agent_pos[:, None, :] - active_agent_pos[None, :, :]
            dist_matrix = torch.norm(delta, dim=2)  # (N_active, N_active)

            # Mask: exclude self-comparison and distances beyond safe_dist
            aa_mask = (dist_matrix < self.safe_dist) & (
                ~torch.eye(dist_matrix.shape[0], device=device, dtype=torch.bool))

            # Inverse distance penalty (avoid division by zero)
            inv_dist_penalty = torch.zeros_like(dist_matrix)
            # inv_dist_penalty[aa_mask] =  1-(dist_matrix[aa_mask]/self.safe_dist)
            inv_dist_penalty[aa_mask] = -torch.exp(-dist_matrix[aa_mask] * 7.0 / self.safe_dist)
            penalty = inv_dist_penalty.sum(dim=1)
            # rewards[active_agents] -= penalty
            components[active_agents, 4] = penalty
        # ---------- Active-Agent vs Obstacles & Done-Agents Penalty ----------
        done_agent_pos = self.agent_pos[self.dones]  # Done agents as static obstacles
        num_done = int(done_agent_pos.shape[0])

        # Stack obstacle positions and radii with done agents (radius = agent_radius)
        obstacle_like_pos = torch.cat([self.obstacle_pos, done_agent_pos], dim=0)  # [M + K, 2]
        obstacle_like_radius = torch.cat([
            self.obstacle_radius.to(device),
            torch.full((num_done,), float(self.agent_radius), device=device)
        ], dim=0)

        ap = active_agent_pos.unsqueeze(1)  # (N_active, 1, 2)
        ob = obstacle_like_pos.unsqueeze(0)  # (1, M+K, 2)
        dist_ap_ob = torch.norm(ap - ob, dim=2)  # (N_active, M+K)

        effective_dist = dist_ap_ob  # (N_active, M+K)
        ob_mask = effective_dist < self.safe_dist

        # Inverse distance penalty for obstacles
        inv_dist_penalty = torch.zeros_like(effective_dist)
        # inv_dist_penalty[ob_mask] = 1-(effective_dist[ob_mask]/self.safe_dist)
        inv_dist_penalty[ob_mask] = -torch.exp(-effective_dist[ob_mask] * 7.0 / self.safe_dist)

        penalty = inv_dist_penalty.sum(dim=1)
        # rewards[active_agents] -= penalty
        components[active_agents, 5] = penalty
        # ______________________________Velocity Reward__________________________________________________
        # --- Smoothness incentives for active agents only ---
        # Linear velocity reward (encourage moving forward)
        # lin_penalty =torch.zeros_like(rewards)
        # ang_penalty =torch.zeros_like(rewards)
        lin_penalty = -(1.0 - (self.agent_vel_lin[active_agents].abs() / self.v_lin_max))
        # Angular velocity penalty (discourage sharp turns)
        # ang_penalty = - self.agent_vel_ang[active_agents].abs() / self.v_ang_max

        # rewards[active_agents]+=lin_penalty[active_agents]+ang_penalty[active_agents]
        components[active_agents, 6] = lin_penalty
        # components[active_agents, 7] = ang_penalty

        target_pos = torch.full_like(self.agent_pos, float("nan"), device=device)
        if assigned_landmark_indices.shape[0] > 0:
            target_pos[active_agents] = self.landmarks[assigned_landmark_indices[active_agents]]

        to_landmark = target_pos - self.agent_pos
        to_landmark_norm = torch.norm(to_landmark, dim=1, keepdim=True).clamp(min=1e-8)
        to_landmark_dir = to_landmark / to_landmark_norm

        agent_theta = self.agent_dir
        agent_dir_vector = torch.stack([torch.cos(agent_theta), torch.sin(agent_theta)], dim=1)

        cos_sim = (agent_dir_vector * to_landmark_dir).sum(dim=1)
        directional_reward = cos_sim - 1.0
        directional_reward[~active_agents] = 0.0

        components[:, 7] = directional_reward
        # ------------------------------------Time penalty-------------------------------------------------------
        # rewards[active_agents] += -scalar_for_sensitivity * self.time_penalty_scale
        components[active_agents, 8] = -1

        # ---------- Finalize ----------
        # rewards=rewards/scalar_for_sensitivity
        self.current_rewards = components  # rewards
        self.old_hungarian = new_hungarian
        self.terminate_agents(covered_agent_indices)
        return components

    def terminate_agents(self, indices):
        self.dones[indices] = True
        self.agent_vel_lin[indices]=0
        self.agent_vel_ang[indices]=0

    def get_hungarian_distances(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Keep everything on CPU
            agent_pos = self.agent_pos.detach().cpu()
            landmarks = self.landmarks.detach().cpu()
            device = self.device

            active_agent_mask = ~self.dones.cpu()
            active_landmark_mask = ~self.covered.cpu()

            active_agents = agent_pos[active_agent_mask]
            active_landmarks = landmarks[active_landmark_mask]

            num_agents = self.agent_pos.shape[0]
            assigned_dists = torch.full((num_agents,), 0.0)  # stays CPU

            if active_agents.shape[0] == 0 or active_landmarks.shape[0] == 0:
                return (
                    assigned_dists.to(device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                )

            # Hungarian assignment (CPU)
            cost_matrix = torch.cdist(active_agents, active_landmarks)
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu())
            row_ind = torch.tensor(row_ind, dtype=torch.long)
            col_ind = torch.tensor(col_ind, dtype=torch.long)

            active_assigned_dists = cost_matrix[row_ind, col_ind]

            full_agent_indices = active_agent_mask.nonzero(as_tuple=True)[0][row_ind]
            full_landmark_indices = active_landmark_mask.nonzero(as_tuple=True)[0][col_ind]

            assigned_dists[full_agent_indices] = active_assigned_dists

            covered_mask = active_assigned_dists < float(self.agent_radius)
            covered_agent_indices = full_agent_indices[covered_mask]
            covered_landmark_indices = full_landmark_indices[covered_mask]
            assigned_landmark_indices = torch.full((num_agents,), -1,
                                                   dtype=torch.long)  # -1 = unassigned (e.g., done agents)
            assigned_landmark_indices[full_agent_indices] = full_landmark_indices

            # Only move to CUDA ONCE here for return
            return (
                assigned_dists.to(device),
                covered_agent_indices.to(device),
                covered_landmark_indices.to(device),
                assigned_landmark_indices.to(device),

            )

class DiffDriveParallelEnvDoneAdj(DiffDriveParallelEnvDone, DiffDriveParallelEnvAdj):
    def __init__(
            self,
            num_agents: int = num_agents,
            env_size: float = env_size,
            num_obstacles: int = num_obstacles,
            v_lin_max: float = v_lin_max,
            v_ang_max: float = v_ang_max,
            dv_lin_max: float = dv_lin_max,
            dv_ang_max: float = dv_ang_max,
            agent_radius: float = agent_radius,
            safe_dist: float = safe_dist,
            sens_range: float = sens_range,
            obstacle_size_min: float = obstacle_size_min,
            obstacle_size_max: float = obstacle_size_max,
            device: Union[str, torch.device] = device,
            normalise=normalise,


    ):
        DiffDriveParallelEnvAdj.__init__(self,
            num_agents=num_agents,
            env_size=env_size,
            num_obstacles=num_obstacles,
            v_lin_max=v_lin_max,
            v_ang_max=v_ang_max,
            agent_radius=agent_radius,
            safe_dist=safe_dist,
            sens_range=sens_range,
            obstacle_size_min=obstacle_size_min,
            obstacle_size_max=obstacle_size_max,
            device=device,
            normalise=normalise,
            dv_lin_max=dv_lin_max,
            dv_ang_max=dv_ang_max,

        )
        if normalise:
            self.obs_dim = 3 * self.num_obstacles + 3 * self.num_landmarks + (self.num_agents - 1) * 7 + self.action_dim
            self.state_dim = 7 * self.num_agents + 3 * self.num_landmarks + 3 * self.num_obstacles
        else:
            self.obs_dim=2*self.num_obstacles+3*self.num_landmarks+(self.num_agents-1)*6+self.action_dim
            self.state_dim=6*self.num_agents+3*self.num_landmarks+3*self.num_obstacles

    def get_observation(self, idx: int) -> torch.Tensor:
        device = self.device
        normalize = self.normalise

        pos = self.agent_pos[idx].to(device)  # (2,)
        heading = self.agent_dir[idx].to(device)

        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        rot = torch.stack([
            torch.stack([cos_h, sin_h]),
            torch.stack([-sin_h, cos_h])
        ])  # (2, 2)

        # Own motion
        own_lin = self.agent_vel_lin[idx].unsqueeze(0).to(device)
        own_ang = self.agent_vel_ang[idx].unsqueeze(0).to(device)
        if normalize:
            own_lin = own_lin / self.v_lin_max
            own_ang = own_ang / self.v_ang_max

        # Other agents
        mask = torch.arange(self._num_agents, device=device) != idx
        other_pos = self.agent_pos[mask]
        rel_pos_global = other_pos - pos
        dist_to_agents = torch.norm(rel_pos_global, dim=1)
        sorted_idx = torch.argsort(dist_to_agents)

        rel_pos = (rel_pos_global[sorted_idx]) @ rot.T
        rel_dir = self.agent_dir[mask][sorted_idx] - heading
        rel_dir = torch.atan2(torch.sin(rel_dir), torch.cos(rel_dir))

        if normalize:
            rel_pos = rel_pos / self.env_size
            rel_dir_sin = torch.sin(rel_dir).unsqueeze(1)
            rel_dir_cos = torch.cos(rel_dir).unsqueeze(1)
        else:
            rel_dir = rel_dir.unsqueeze(1)

        lin_vels = self.agent_vel_lin[mask][sorted_idx].unsqueeze(1).to(device)
        ang_vels = self.agent_vel_ang[mask][sorted_idx].unsqueeze(1).to(device)
        if normalize:
            lin_vels = lin_vels / self.v_lin_max
            ang_vels = ang_vels / self.v_ang_max
        dones=self.dones[mask][sorted_idx]

        # Landmarks
        rel_lm_global = self.landmarks - pos
        dist_to_lm = torch.norm(rel_lm_global, dim=1)
        sorted_lm_idx = torch.argsort(dist_to_lm)

        rel_lm = (rel_lm_global[sorted_lm_idx]) @ rot.T
        if normalize:
            rel_lm = rel_lm / self.env_size
        covered=self.covered[sorted_lm_idx]

        # Obstacles
        obs_vec = self.obstacle_pos - pos
        center_dists = torch.norm(obs_vec, dim=1)
        edge_dists = center_dists - self.obstacle_radius-self.agent_radius
        in_range = edge_dists < self.sens_range
        obs_idx = torch.argsort(edge_dists)
        obs_idx = obs_idx[in_range[obs_idx]]

        edge_dists = edge_dists[obs_idx]
        if normalize:
            edge_dists = edge_dists / self.sens_range

        obs_vec_local = obs_vec[obs_idx] @ rot.T
        obs_angles = torch.atan2(obs_vec_local[:, 1], obs_vec_local[:, 0])

        if normalize:
            obs_angle_sin = torch.sin(obs_angles)
            obs_angle_cos = torch.cos(obs_angles)

        # Pad obstacles
        pad_len = self.num_obstacles - len(obs_idx)

        def pad(x, value=0):
            if pad_len > 0:
                return torch.cat([x, torch.full((pad_len,), value, device=device)], dim=0)
            return x

        edge_dists = pad(edge_dists)
        if normalize:
            obs_angle_sin = pad(obs_angle_sin)
            obs_angle_cos = pad(obs_angle_cos)
        else:
            obs_angles = pad(obs_angles)

        # Final concat
        components = [
            own_lin, own_ang,
            rel_pos.flatten(),
            # self.dones[idx].unsqueeze(0)
        ]

        if normalize:
            components.extend([rel_dir_sin.flatten(), rel_dir_cos.flatten()])
        else:
            components.append(rel_dir.flatten())

        components.extend([
            lin_vels.flatten(),
            ang_vels.flatten(),
            dones,
            rel_lm.flatten(),
            covered,
            edge_dists,
        ])

        if normalize:
            components.extend([obs_angle_sin, obs_angle_cos])
        else:
            components.append(obs_angles)

        return torch.cat(components, dim=0).float().to(device)

class DiffDriveParallelEnvAssignFirstDone(DiffDriveParallelEnvDone):
    def __init__(
            self,
            num_agents: int = num_agents,
            env_size: float = env_size,
            num_obstacles: int = num_obstacles,
            v_lin_max: float = v_lin_max,
            v_ang_max: float = v_ang_max,
            agent_radius: float = agent_radius,
            safe_dist: float = safe_dist,
            sens_range: float = sens_range,
            obstacle_size_min: float = obstacle_size_min,
            obstacle_size_max: float = obstacle_size_max,
            device: Union[str, torch.device] = device,
            normalise=normalise,

    ):
        super().__init__(
            num_agents=num_agents,
            env_size=env_size,
            num_obstacles=num_obstacles,
            v_lin_max=v_lin_max,
            v_ang_max=v_ang_max,
            agent_radius=agent_radius,
            safe_dist=safe_dist,
            sens_range=sens_range,
            obstacle_size_min=obstacle_size_min,
            obstacle_size_max=obstacle_size_max,
            device=device,
            normalise=normalise,


        )
        if normalise:
            self.obs_dim = 3 * (self.num_obstacles+self.num_agents - 1) + 2  #other agents as obstacle + own landmark position
            self.state_dim = 5 * self.num_agents + 3 * self.num_landmarks + 3 * self.num_obstacles
        else:
            self.obs_dim=2*(self.num_obstacles+self.num_agents-1)+2 #
            self.state_dim=4*self.num_agents+3*self.num_landmarks+3*self.num_obstacles


    def get_observation(self, idx: int) -> torch.Tensor:
        device = self.device
        normalize = self.normalise

        pos = self.agent_pos[idx].to(device)  # (2,)
        heading = self.agent_dir[idx].to(device)

        # Rotation matrix for ego-centric coordinates
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        rot = torch.stack([
            torch.stack([cos_h, sin_h]),
            torch.stack([-sin_h, cos_h])
        ])  # (2, 2)

        # ===================== Landmark (only assigned one) =====================
        lm_idx = self.assigned_landmark_ids[idx]
        rel_lm_global = self.landmarks[lm_idx].to(device) - pos  # (2,)
        rel_lm_local = rel_lm_global @ rot.T
        if normalize:
            rel_lm_local = rel_lm_local / self.env_size

        # ===================== Other Agents (as obstacles) =====================
        mask = torch.arange(self._num_agents, device=device) != idx
        other_pos = self.agent_pos[mask]  # (N-1, 2)
        rel_pos_global = other_pos - pos  # (N-1, 2)
        dists = torch.norm(rel_pos_global, dim=1) - self.agent_radius  # edge-to-edge

        in_range = dists < self.sens_range
        sorted_idx = torch.argsort(dists)
        sorted_idx = sorted_idx[in_range[sorted_idx]]

        selected = rel_pos_global[sorted_idx]  # (M, 2)
        rel_pos_local = selected @ rot.T
        edge_dists = dists[sorted_idx]
        angles = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0])

        if normalize:
            edge_dists = edge_dists / self.sens_range

        if normalize:
            agent_angle_sin = torch.sin(angles)
            agent_angle_cos = torch.cos(angles)
        else:
            agent_angles = angles

        # ===================== Obstacles =====================
        obs_vec = self.obstacle_pos - pos  # (num_obstacles, 2)
        center_dists = torch.norm(obs_vec, dim=1)
        edge_dists_obs = center_dists - self.obstacle_radius  # (num_obstacles,)
        in_range = edge_dists_obs < self.sens_range
        obs_idx = torch.argsort(edge_dists_obs)
        obs_idx = obs_idx[in_range[obs_idx]]

        obs_vec_local = obs_vec[obs_idx] @ rot.T
        edge_dists_obs = edge_dists_obs[obs_idx]
        angles_obs = torch.atan2(obs_vec_local[:, 1], obs_vec_local[:, 0])

        if normalize:
            edge_dists_obs = edge_dists_obs / self.sens_range

        if normalize:
            obs_angle_sin = torch.sin(angles_obs)
            obs_angle_cos = torch.cos(angles_obs)
        else:
            obs_angles = angles_obs

        # ===================== Padding =====================
        def pad(x, target_len, value=0.0):
            pad_len = target_len - x.shape[0]
            if pad_len > 0:
                return torch.cat([x, torch.full((pad_len,), value, device=device)], dim=0)
            return x

        max_agents = self._num_agents - 1
        max_obs = self.num_obstacles

        edge_dists = pad(edge_dists, max_agents)
        if normalize:
            agent_angle_sin = pad(agent_angle_sin, max_agents)
            agent_angle_cos = pad(agent_angle_cos, max_agents)
        else:
            agent_angles = pad(agent_angles, max_agents)

        edge_dists_obs = pad(edge_dists_obs, max_obs)
        if normalize:
            obs_angle_sin = pad(obs_angle_sin, max_obs)
            obs_angle_cos = pad(obs_angle_cos, max_obs)
        else:
            obs_angles = pad(obs_angles, max_obs)

        # ===================== Final observation =====================
        components = [rel_lm_local]  # assigned landmark (2,)

        components.append(edge_dists)  # (N-1,)
        if normalize:
            components.extend([agent_angle_sin, agent_angle_cos])
        else:
            components.append(agent_angles)

        components.append(edge_dists_obs)
        if normalize:
            components.extend([obs_angle_sin, obs_angle_cos])
        else:
            components.append(obs_angles)

        return torch.cat(components, dim=0).float().to(device)

    def get_hungarian_distances(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the assigned distances between agents and landmarks using the Hungarian algorithm.
        Also returns the assignment: landmark index assigned to each agent.

        Returns:
            assigned_dists (torch.Tensor): Shape [num_agents], distances after optimal assignment.
            assignment (torch.Tensor): Shape [num_agents], assignment[i] is the index of landmark assigned to agent i.
        """
        with torch.no_grad():
            device = self.device
            agent_pos = self.agent_pos.detach().cpu()
            landmarks = self.landmarks.detach().cpu()

            cost_matrix = torch.cdist(agent_pos, landmarks)  # Shape: [N, N]
            row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

            assigned_dists = cost_matrix[row_ind, col_ind]  # shape: (N,)
            assignment = torch.zeros(agent_pos.shape[0], dtype=torch.long)
            assignment[row_ind] = torch.tensor(col_ind, dtype=torch.long)

            return (
                torch.tensor(assigned_dists, dtype=torch.float32, device=device),
                assignment.to(device)
            )
    def _reset_hungarian(self):
        self.old_hungarian, self.assigned_landmark_ids = self.get_hungarian_distances()
    def _compute_rewards_tensor(self) -> torch.Tensor:
        """
        Computes rewards for all agents:
          - Global progress via Hungarian assignment (active agents only)
          - Penalty for active-agent proximity
          - Penalty for active-agent vs obstacle proximity
          - Penalty for active-agent vs inactive-agent proximity (done agents are static obstacles)
          - Reward for covering landmarks (reached_goal_scale)

        Updates:
            self.dones
            self.covered
            self.current_rewards
            self.old_hungarian

        Returns:
            torch.Tensor: [num_agents] individual rewards
        """
        N = self._num_agents
        device = self.device
        # rewards = torch.zeros(N, device=device)
        components = torch.zeros(N, 9, device=device)

        active_agents = ~self.dones
        active_indices = active_agents.nonzero(as_tuple=True)[0]
        active_agent_pos = self.agent_pos[active_agents]

        # ---------- Global Goal Reward (Hungarian) ----------
        new_hungarian = torch.norm(self.landmarks[self.assigned_landmark_ids] - self.agent_pos, dim=1)


        d_global = 0
        d_goal =  -new_hungarian/self.env_size
        # d_goal = torch.nan_to_num(d_goal, nan=0.0)

        progressive =  torch.nan_to_num(
            (self.old_hungarian - new_hungarian) / self.v_lin_max, nan=0.0)
        progressive = torch.clamp(progressive, min=-1.0, max=1.0)

        # rewards[active_agents] =  progressive[active_agents]+d_global+d_goal[active_agents]
        components[active_agents, 0] = progressive[active_agents]  # progressive
        components[active_agents, 1] = d_goal[active_agents]  # dist-to-goal
        components[active_agents, 2] = d_global  # global penalty (shared to all active)

        # ---------- Reward for Newly Covered Landmarks ----------
        # rewards[covered_agent_indices] += scalar_for_sensitivity*self.reached_goal_scale
        # components[covered_agent_indices, 3] = 1
        covered_agent_indices = ((new_hungarian < self.agent_radius) & active_agents).nonzero(as_tuple=True)[0]
        covered_landmark_indices = self.assigned_landmark_ids[covered_agent_indices]
        self.covered[covered_landmark_indices] = True
        components[covered_agent_indices, 3] = 1
        # self.dones[covered_agent_indices] = True
        # self.covered[covered_landmark_indices] = True

        # ---------- Active-Agent vs Active-Agent Penalty ----------
        if active_agent_pos.shape[0] > 1:
            delta = active_agent_pos[:, None, :] - active_agent_pos[None, :, :]
            dist_matrix = torch.norm(delta, dim=2)   # (N_active, N_active)

            # Mask: exclude self-comparison and distances beyond safe_dist
            aa_mask = (dist_matrix < self.safe_dist) & (
                ~torch.eye(dist_matrix.shape[0], device=device, dtype=torch.bool))

            # Inverse distance penalty (avoid division by zero)
            inv_dist_penalty = torch.zeros_like(dist_matrix)
            # inv_dist_penalty[aa_mask] =  1-(dist_matrix[aa_mask]/self.safe_dist)
            inv_dist_penalty[aa_mask] = -torch.exp(-dist_matrix[aa_mask] * 7.0 / self.safe_dist)
            penalty = inv_dist_penalty.sum(dim=1)
            # rewards[active_agents] -= penalty
            components[active_agents, 4] = penalty
        # ---------- Active-Agent vs Obstacles & Done-Agents Penalty ----------
        done_agent_pos = self.agent_pos[self.dones]  # Done agents as static obstacles
        num_done = int(done_agent_pos.shape[0])

        # Stack obstacle positions and radii with done agents (radius = agent_radius)
        obstacle_like_pos = torch.cat([self.obstacle_pos, done_agent_pos], dim=0)  # [M + K, 2]
        obstacle_like_radius = torch.cat([
            self.obstacle_radius.to(device),
            torch.full((num_done,), float(self.agent_radius), device=device)
        ], dim=0)

        ap = active_agent_pos.unsqueeze(1)  # (N_active, 1, 2)
        ob = obstacle_like_pos.unsqueeze(0)  # (1, M+K, 2)
        dist_ap_ob = torch.norm(ap - ob, dim=2)  # (N_active, M+K)

        effective_dist = dist_ap_ob  # (N_active, M+K)
        ob_mask = effective_dist < self.safe_dist

        # Inverse distance penalty for obstacles
        inv_dist_penalty = torch.zeros_like(effective_dist)
        # inv_dist_penalty[ob_mask] = 1-(effective_dist[ob_mask]/self.safe_dist)
        inv_dist_penalty[ob_mask]=-torch.exp(-effective_dist[ob_mask] * 7.0 / self.safe_dist)

        penalty =  inv_dist_penalty.sum(dim=1)
        # rewards[active_agents] -= penalty
        components[active_agents, 5] = penalty
        #______________________________Velocity Reward__________________________________________________
        # --- Smoothness incentives for active agents only ---
        # Linear velocity reward (encourage moving forward)
        # lin_penalty =torch.zeros_like(rewards)
        # ang_penalty =torch.zeros_like(rewards)
        lin_penalty = -(1.0 - self.agent_vel_lin[active_agents] / self.v_lin_max)
        # Angular velocity penalty (discourage sharp turns)
        # ang_penalty = - self.agent_vel_ang[active_agents].abs() / self.v_ang_max

        # rewards[active_agents]+=lin_penalty[active_agents]+ang_penalty[active_agents]
        components[active_agents, 6] = lin_penalty
        # components[active_agents, 7] = ang_penalty
        #------------------------------------Time penalty-------------------------------------------------------
        # rewards[active_agents] += -scalar_for_sensitivity * self.time_penalty_scale
        target_pos = torch.full_like(self.agent_pos, float("nan"), device=device)
        target_pos = self.landmarks[self.assigned_landmark_ids]

        to_landmark = target_pos[active_agents] - self.agent_pos[active_agents]
        # to_landmark_norm = torch.norm(to_landmark, dim=1, keepdim=True).clamp(min=1e-8)
        to_landmark_dir = to_landmark / new_hungarian[active_agents].unsqueeze(1)

        agent_theta = self.agent_dir[active_agents]
        agent_dir_vector = torch.stack([torch.cos(agent_theta), torch.sin(agent_theta)], dim=1)

        cos_sim = (agent_dir_vector * to_landmark_dir).sum(dim=1)
        directional_reward = cos_sim - 1.0


        components[active_agents, 7] = directional_reward




        components[active_agents, 8] = -1


        # ---------- Finalize ----------
        # rewards=rewards/scalar_for_sensitivity
        self.current_rewards = components #rewards
        self.old_hungarian = new_hungarian
        self.terminate_agents(covered_agent_indices)
        return components

class DiffDriveParallelEnvAssignFirstDoneAdj(DiffDriveParallelEnvAssignFirstDone, DiffDriveParallelEnvDoneAdj):
    def __init__(
            self,
            num_agents: int = num_agents,
            env_size: float = env_size,
            num_obstacles: int = num_obstacles,
            v_lin_max: float = v_lin_max,
            v_ang_max: float = v_ang_max,
            dv_lin_max: float = dv_lin_max,
            dv_ang_max: float = dv_ang_max,
            agent_radius: float = agent_radius,
            safe_dist: float = safe_dist,
            sens_range: float = sens_range,
            obstacle_size_min: float = obstacle_size_min,
            obstacle_size_max: float = obstacle_size_max,
            device: Union[str, torch.device] = device,
            normalise=normalise,


    ):
        DiffDriveParallelEnvDoneAdj.__init__(self,
            num_agents=num_agents,
            env_size=env_size,
            num_obstacles=num_obstacles,
            v_lin_max=v_lin_max,
            v_ang_max=v_ang_max,
            agent_radius=agent_radius,
            safe_dist=safe_dist,
            sens_range=sens_range,
            obstacle_size_min=obstacle_size_min,
            obstacle_size_max=obstacle_size_max,
            device=device,
            normalise=normalise,
            dv_lin_max=dv_lin_max,
            dv_ang_max=dv_ang_max,

        )
        if normalise:
            self.obs_dim = 3 * (self.num_obstacles+self.num_agents - 1) + 2  + self.action_dim
            self.state_dim = 7 * self.num_agents + 3 * self.num_landmarks + 3 * self.num_obstacles
        else:
            self.obs_dim=2*(self.num_obstacles+self.num_agents-1)+2+self.action_dim
            self.state_dim=6*self.num_agents+3*self.num_landmarks+3*self.num_obstacles


    def get_observation(self, idx: int) -> torch.Tensor:
        return   torch.cat([
            super().get_observation(idx),
            self.agent_vel_lin[idx].unsqueeze(0),
            self.agent_vel_ang[idx].unsqueeze(0)
        ], dim=0).float().to(device)
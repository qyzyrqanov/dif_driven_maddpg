
import torch
import numpy as np


import torch

# ====== Environment settings ======
env_name = "diff_drive_multiagent"
render_mode = "human"  # or "none"

# ====== Environment settings ======
# Number of agents (should match number of landmarks)
num_agents = 6
epsilon = 1e-8
# Number of obstacles (moderate complexity)
num_obstacles = 0  # 1 obstacle per agent for balanced navigation

# Environment size (2D continuous space)
env_size = 20  # Map is 100x100 meters

# Episode length
max_steps = 500  # Long enough for full behavior to emerge, short enough for stable training

# ====== Observation, State, Action dimensions ======
# obs_dim = 78     # Per-agent observation dimension
# state_dim = 81   # Global state used by the critic
# action_dim = 2   # dVlin and dVang per agent

# Observation and action space bounds
obs_low = -1.0
obs_high = 1.0
act_low = -1.0
act_high = 1.0

# ====== Motion parameters ======
v_lin_max = 1.0          # Max linear velocity
v_ang_max = torch.pi/9       # Max angular velocity (degrees)
dv_lin_max =  v_lin_max/4  # Max delta linear velocity per step
dv_ang_max = v_ang_max/4       # Max delta angular velocity per step


# ====== Reward settings ======
collision_penalty_scale = 100
reached_goal_scale = 100
base_penalty_scale=10
distance_penalty_scale = 10   #negative
velocity_penalty_scale_linear = 0
velocity_penalty_scale_angular = 0
time_penalty_scale = 0
progressive_reward_scale = 0   # Encourage global progress

# ====== Replay Buffer and Training ======
replay_buffer_size = 131_072  # Large enough to avoid overfitting, safe for 6GB GPU (store on CPU)
batch_size = 128               # Optimized for 6GB GPU, adjust if needed
start_training_after = 8192 # 1000   # Sooner training for small buffer

# ====== Network architecture ======
# hidden_dim_actor = 256      # Sufficient for obs_dim=78, 2-layer ReLU net
# hidden_dim_critic = 256     # For input_dim = state_dim + joint_actions = 99
# num_critic_heads = 3        # Average over 3 heads for ensemble stability

# ====== Exploration Noise ======
std_scale = 0.3             # Gaussian noise std = 0.3 * max_action
use_noise = True            # Use noise during training, not evaluation

# CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Agent Size and Safety Settings
agent_radius = 0.5
safe_dist = v_lin_max
sens_range = 5 * v_lin_max


obstacle_size_min=1
obstacle_size_max=2

# === Training and Replay ===
gamma = 0.99                        # Discount factor
tau = 0.005                         # For soft update of target networks


#n_games
n_games = 25_000  # Total number of games to train the agents
train_each=128

critic_lr: float = 1e-3
critic_ckpt: str = 'shared_critic.pth'
actor_lr = 1e-3

normalise=True # normalise the observations and state

# === Training Configuration ===



patience = 256                   # Max episodes with no improvement before early stopping
min_episodes_before_early_stop = 128  # Minimum number of episodes before early stopping is considered

score_avg_window = 64            # Number of recent episodes to average for performance evaluation

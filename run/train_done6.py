import torch
import numpy as np
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import DiffDriveParallelEnvDone
from rl.maddpg import MADDPGSharedActorCriticIndependent, MADDPGSharedActorCriticIndependentQmean, IDDPG, IDDPGWithoutS

seed=9832
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

scale=[
            1.0,      # progressive
            1.0,      # distance
            0.0,      # base
            10.0,   # reached goal
            10.0,    # agent collision
            10.0,    # obstacle collision
            1.0,      # v_lin
            1.0,      # v_ang
            1.0       # time
        ]

env=DiffDriveParallelEnvDone(
    v_ang_max=torch.pi/9,
    num_agents=6,
    num_obstacles=0
)

maddpg=IDDPGWithoutS(
    env,
    reward_scales=scale,
    batch_size=128,
    replay_buffer_size=50000,
)

maddpg.train_loop(
    start_training_after=500,
    train_each=100,
    patience=256,
    min_episodes_before_early_stop=10000,
    score_avg_window=256,
    max_steps=500,
    meta_extra={"seed": seed, "mode": "full"},
)

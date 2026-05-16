import torch, os

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, state_dim, num_agents, replay_buffer_size, batch_size, device):
        self.max_size = replay_buffer_size
        self.batch_size = batch_size
        self.device = device
        self.size = 0

        self.tagged_capacity = int(self.max_size * 0.25)
        self.notagged_capacity = self.max_size - self.tagged_capacity
        self.current_tagged_idx = 0
        self.current_notagged_idx = 0
        self.total_tagged = 0

        # Buffers
        self.obs_buf = torch.zeros((replay_buffer_size, num_agents, obs_dim))
        self.next_obs_buf = torch.zeros((replay_buffer_size, num_agents, obs_dim))
        self.act_buf = torch.zeros((replay_buffer_size, num_agents, action_dim))
        self.reward_buf = torch.zeros((replay_buffer_size, num_agents, 9))
        self.done_buf = torch.zeros((replay_buffer_size, num_agents), dtype=torch.bool)
        self.state_buf = torch.zeros((replay_buffer_size, state_dim))
        self.next_state_buf = torch.zeros((replay_buffer_size, state_dim))
        self.is_tagged_buf = torch.zeros((replay_buffer_size,), dtype=torch.bool)

    def add(self, state, observations, actions, rewards, next_state, next_observations, dones, tagged: bool = False):
        if tagged:
            idx = self.current_tagged_idx
            self.current_tagged_idx = (self.current_tagged_idx + 1) % self.tagged_capacity
            self.total_tagged = min(self.total_tagged + 1, self.tagged_capacity)
        else:
            idx = self.tagged_capacity + self.current_notagged_idx
            self.current_notagged_idx = (self.current_notagged_idx + 1) % self.notagged_capacity

        # Save transition
        self.obs_buf[idx] = observations.clone().detach().to(self.device)
        self.next_obs_buf[idx] = next_observations.clone().detach().to(self.device)
        self.act_buf[idx] = actions.clone().detach().to(self.device)
        self.reward_buf[idx] = rewards.clone().detach().to(self.device)
        self.done_buf[idx] = dones.clone().detach().bool().to(self.device)
        self.state_buf[idx] = state.clone().detach().to(self.device)
        self.next_state_buf[idx] = next_state.clone().detach().to(self.device)
        self.is_tagged_buf[idx] = tagged

        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        num_tagged_needed = int(batch_size * 0.25)
        num_tagged_available = min(self.total_tagged, self.tagged_capacity)
        num_untagged_available = min(self.size - self.total_tagged, self.notagged_capacity)

        num_tagged_actual = min(num_tagged_needed, num_tagged_available)
        num_untagged_actual = batch_size - num_tagged_actual

        # Adjust if not enough untagged to fill the rest
        if num_untagged_actual > num_untagged_available:
            num_untagged_actual = num_untagged_available
            num_tagged_actual = batch_size - num_untagged_actual

            # If still not enough tagged to compensate, shrink batch
            if num_tagged_actual > num_tagged_available:
                num_tagged_actual = num_tagged_available

        # Final fallback: if total < batch_size
        actual_total = num_tagged_actual + num_untagged_actual
        if actual_total < batch_size:
            batch_size = actual_total

        # Sample tagged
        if num_tagged_actual > 0:
            tagged_indices = torch.randint(0, num_tagged_available, (num_tagged_actual,))
        else:
            tagged_indices = torch.tensor([], dtype=torch.long)

        # Sample untagged
        if num_untagged_actual > 0:
            notagged_indices = torch.randint(0, num_untagged_available, (num_untagged_actual,)) + self.tagged_capacity
        else:
            notagged_indices = torch.tensor([], dtype=torch.long)

        idx = torch.cat([tagged_indices, notagged_indices], dim=0)

        return (
            self.obs_buf[idx].to(self.device),
            self.next_obs_buf[idx].to(self.device),
            self.state_buf[idx].to(self.device),
            self.next_state_buf[idx].to(self.device),
            self.act_buf[idx].to(self.device),
            self.reward_buf[idx].to(self.device),
            self.done_buf[idx].to(self.device),
        )

    def save(self, filepath: str):
        """
        Save full replay buffer contents and internal state to file.
        """
        data = {
            'obs_buf': self.obs_buf,
            'next_obs_buf': self.next_obs_buf,
            'act_buf': self.act_buf,
            'reward_buf': self.reward_buf,
            'done_buf': self.done_buf,
            'state_buf': self.state_buf,
            'next_state_buf': self.next_state_buf,
            'is_tagged_buf': self.is_tagged_buf,
            'size': self.size,
            'max_size': self.max_size,
            'tagged_capacity': self.tagged_capacity,
            'notagged_capacity': self.notagged_capacity,
            'current_tagged_idx': self.current_tagged_idx,
            'current_notagged_idx': self.current_notagged_idx,
            'total_tagged': self.total_tagged,
            'batch_size': self.batch_size,
        }
        tmp_filepath = f"{filepath}.tmp"
        torch.save(data, tmp_filepath)
        os.replace(tmp_filepath, filepath)

    def load(self, filepath: str):
        """
        Load full replay buffer contents and internal state from file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Replay buffer file not found: {filepath}")

        data = torch.load(filepath)

        self.obs_buf = data['obs_buf']
        self.next_obs_buf = data['next_obs_buf']
        self.act_buf = data['act_buf']
        self.reward_buf = data['reward_buf']
        self.done_buf = data['done_buf']
        self.state_buf = data['state_buf']
        self.next_state_buf = data['next_state_buf']
        self.is_tagged_buf = data['is_tagged_buf']

        self.size = data['size']
        self.max_size = data['max_size']
        self.tagged_capacity = data['tagged_capacity']
        self.notagged_capacity = data['notagged_capacity']
        self.current_tagged_idx = data['current_tagged_idx']
        self.current_notagged_idx = data['current_notagged_idx']
        self.total_tagged = data['total_tagged']
        self.batch_size = data['batch_size']


import torch
import torch.nn as nn

class MultiheadCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=3, hidden_dim=None, activation=nn.ReLU, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.device = torch.device(device)

        input_dim = state_dim + action_dim
        if hidden_dim is None:
            hidden_dim = max(128, input_dim)

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation()
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_heads)
        ])

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state, action, aggregate='mean'):
        # Ensure inputs are on the same device as model
        state = state.to(self.device)
        action = action.to(self.device)

        x = torch.cat([state, action], dim=-1)  # (B, state+action)
        features = self.shared(x)  # (B, hidden)
        q_values = torch.stack([head(features) for head in self.heads], dim=1)  # (B, H, 1)
        q_values = q_values.squeeze(-1)  # (B, H)

        if aggregate == 'mean':
            return q_values.mean(dim=1, keepdim=True)
        elif aggregate == 'min':
            return q_values.min(dim=1, keepdim=True).values
        elif aggregate == 'max':
            return q_values.max(dim=1, keepdim=True).values
        elif aggregate is None:
            return q_values  # (B, num_heads)
        else:
            raise ValueError(f"Invalid aggregation mode: {aggregate}")

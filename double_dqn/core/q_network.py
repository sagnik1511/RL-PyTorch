import torch
import torch.nn as nn


class DDQNetwork(nn.Module):

    def __init__(self, num_observations, num_actions):
        super().__init__()
        self.obs_n = num_observations
        self.out_n = num_actions
        self.net = self._define_network()

    def _define_network(self):
        model = nn.Sequential(
            nn.Linear(self.obs_n, 512),
            nn.Linear(512, 512),
            nn.Linear(512, self.out_n),
        )
        return model

    def forward(self, state):
        l_op = self.net(state)
        return torch.softmax(l_op, dim=1)

    def load_model(self, path):
        print(f"Loading model parameters from {path}")
        self.net._load_from_state_dict(torch.load(path))

    def save_model(self, path):
        print(f"Saving model parameters at {path}")
        torch.save(self.net.state_dict(), path)

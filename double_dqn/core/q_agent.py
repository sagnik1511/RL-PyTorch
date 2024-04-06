import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

from double_dqn.misc.replay_buffer import ReplayBuffer
from .q_network import DDQNetwork


class DDQNAgent:

    def __init__(
        self,
        memory_size,
        batch_size,
        observation_size,
        action_space_size,
        gamma,
        learning_rate,
        epsilon,
        epsilon_decay,
        epsilon_min,
        replace_counter=1000,
        checkpoint_directory="runs/artifacts/",
    ):
        self.batch_size = batch_size
        self.eps = epsilon
        self.eps_decay = epsilon_decay
        self.eps_min = epsilon_min
        self.gamma = gamma
        self.lr = learning_rate
        self.n_actions = action_space_size
        self.replace_counter = replace_counter
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(memory_size, batch_size)
        self.online_network = DDQNetwork(observation_size, action_space_size).to(
            self.device
        )
        self.target_network = DDQNetwork(observation_size, action_space_size).to(
            self.device
        )
        # self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate
        )
        self.chkpt_dir = Path(checkpoint_directory)
        self.iteration_counter = 0

    def _update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def get_action(self, state):
        # Updating epsilon (Epsilon Annealing)
        self.eps = max(self.eps_min, self.eps - self.eps_decay)

        if random.random() < self.eps:
            # Choosing Random Exploratory Action
            action = random.randint(0, self.n_actions - 1)
            return action
        else:
            # Moving the state to device as torch.tensor batch
            state = (
                torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            )
            self.online_network.eval()
            with torch.no_grad():
                # Generating Agent Response
                pred = self.online_network(state)
                action = torch.argmax(pred).squeeze(0)
            self.online_network.train()
            return action.item()

    def _update_q_values(self, states, actions, rewards, states_, dones):

        # Moving the data to corresponding device as torch.Tensor
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        )
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.long).to(self.device).unsqueeze(1)

        # Predicting actions from online network
        predicted_actions = self.online_network(states)
        q_values = predicted_actions.gather(1, actions)

        next_state_predicted_actions = self.target_network(states_)
        next_max_q_values = next_state_predicted_actions.detach().max(1)[0].unsqueeze(1)

        # We'll only the q values from the states where the episode didn't terminated
        next_q_values = rewards + self.gamma * next_max_q_values * (1 - dones)

        self.optimizer.zero_grad()
        loss = F.mse_loss(
            q_values, next_q_values
        )  # self.criterion(q_values, next_q_values)
        loss.backward()
        self.optimizer.step()

    def save_artifact(self):
        if not self.chkpt_dir.exists():
            print(f"Checkpoint Directory created at {self.chkpt_dir}")
            self.chkpt_dir.mkdir(parents=True, exist_ok=True)
        online_fpath = (
            str(self.chkpt_dir)
            + f"/online_model-{self.gamma}-{self.eps};{self.eps_min}-{self.lr}.pt"
        )
        target_fpath = online_fpath.replace("online_model", "target_model")
        print(f"Saving artifacts at {online_fpath} and {target_fpath}")
        self.online_network.save_model(online_fpath)
        self.target_network.save_model(target_fpath)

    def train(self):
        self.iteration_counter += 1
        if len(self.replay_buffer) < self.batch_size:
            # Not enough records to train the network
            return
        else:
            # Updating Target Network after a buffer time
            if (
                self.iteration_counter > 0
                and self.iteration_counter % self.replace_counter == 0
            ):
                self._update_target_network()
            states, actions, rewards, states_, dones = self.replay_buffer.sample()
            self._update_q_values(states, actions, rewards, states_, dones)

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.bs = batch_size

    def push(self, state, action, reward, state_, done):
        self.buffer.append([state, action, reward, state_, done])

    def sample(self):
        batch = random.sample(self.buffer, self.bs)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

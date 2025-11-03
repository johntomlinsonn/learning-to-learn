#Code from original OpenAI Gym CartPole example with minor modifications
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make("CartPole-v0")

def build_model():
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        #nueral network with 2 hidden layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #outputs the q value for each action, based on how well the action will do
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
        
    class ReplayBuffer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = []
            self.position = 0

        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (
                np.array(state, dtype=np.float32),
                action,
                np.float32(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
            )
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            batch = torch.rand.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done
        
        def __len__(self):
            return len(self.buffer)
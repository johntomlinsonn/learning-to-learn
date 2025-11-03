import math
import random as rand
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def build_model(state_dim, action_dim):
    class metaNetwork(nn.Module):
        def __init__(self, state_size, action_size, hidden_size=128):
            super(metaNetwork, self).__init__()
            # neural network with 2 hidden layers
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            # outputs the q value for each action, based on how well the action will do
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
            batch = rand.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done

        def __len__(self):
            return len(self.buffer)

    meta_network = metaNetwork(state_dim, action_dim)



    return meta_network, ReplayBuffer(10000)





def compute_reward(reward_list):

    reward_list = np.array(reward_list, dtype=float)
    N = len(reward_list)

    for i in range(N - 1):
        deltas = reward_list[i + 1:] - reward_list[i]
        total_deltas += deltas
    

    meta_reward = np.sum(total_deltas) / N

    return np.array(meta_reward,dtype=float)


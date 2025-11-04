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
            # outputs a single continuous action value between -1 and 1
            self.fc3 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            # Use tanh to bound the output between -1 and 1
            return torch.tanh(self.fc3(x))
        
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



def sliding_window(data, window_size, stride=1):
    data = np.array(data)
    
    # Calculate the nuber of windows convert to int
    n_samples = int((len(data) - window_size) // stride + 1)
    
    # Create start indices for each window
    start_indices = np.arange(n_samples, dtype=int) * stride
    
    # Create the windows
    windows = np.array([data[i:i + window_size] for i in start_indices])
    
    return windows


def sliding_window(data,window_size):
    new_list = []
    length = len(data)
    for i in range(window_size):
        new_list.append(data[(length - window_size) + i])

    return new_list
        



def compute_reward(reward_list):
    reward_list = np.array(reward_list, dtype=float)
    N = len(reward_list)

    for i in range(N - 1):
        deltas = reward_list[i + 1:] - reward_list[i]
        total_deltas += deltas
    
    meta_reward = np.sum(total_deltas) / N

    return np.array(meta_reward,dtype=float)

window = [1,2,3,4,5,6,7,8,9]
print(sliding_window(window,4))
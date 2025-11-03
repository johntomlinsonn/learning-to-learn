#Code from original OpenAI Gym CartPole example with minor modifications
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make("CartPole-v0")

def build_model():
    class Cartpole_Network(nn.Module):
        def __init__(self, state_size, action_size, hidden_size=128):
            super(Cartpole_Network, self).__init__()
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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    meta_network = Cartpole_Network(state_dim, action_dim)
    #target network learns more slowly to stabilize training
    target_network = Cartpole_Network(state_dim, action_dim)
    #copying the weights from the meta network to the target network
    target_network.load_state_dict(meta_network.state_dict())
    #moving the model to eval mode
    target_network.eval()

    return meta_network, target_network, ReplayBuffer(10000)



def compute_reward(raw_state):
    x, x_dot, theta, theta_dot = raw_state
    angle_error = abs(theta)
    position_error = abs(x)
    reward = 1.0
    reward -= 2.0 * (angle_error / THETA_LIMIT_RADIANS)
    reward -= 0.5 * (position_error / X_LIMIT)
    reward -= 0.01 * (abs(x_dot) + abs(theta_dot))
    return max(reward, -2.0)


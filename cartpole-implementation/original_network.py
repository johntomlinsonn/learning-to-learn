#Code from original OpenAI Gym CartPole example with minor modifications
import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


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

    Cartpole_Network = Cartpole_Network(state_dim, action_dim)
    #target network learns more slowly to stabilize training
    target_network = Cartpole_Network(state_dim, action_dim)
    #copying the weights from the Cartpole_Network to the target network
    target_network.load_state_dict(Cartpole_Network.state_dict())
    #moving the model to eval mode
    target_network.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Cartpole_Network.to(device)
    target_network.to(device)

    optimizer = optim.Adam(Cartpole_Network.parameters(), lr=5e-4)
    #storing the experiences
    memory = ReplayBuffer(10000)

    return Cartpole_Network, target_network, memory, optimizer, device

def build_env():
    env = gym.make("CartPole-v1")
    return env


#The enviorment outputs an array of 4 values: cart position, cart velocity, pole angle, pole angular velocity
def compute_reward(env):
    THETA_LIMIT_RADIANS = 12 * 2 * math.pi / 360  # 12 degrees to radians
    X_LIMIT = 1.0

    state = env.state
    x, x_dot, theta, theta_dot = state
    angle_error = abs(theta)
    position_error = abs(x)
    reward = 1.0
    reward -= 2.0 * (angle_error / THETA_LIMIT_RADIANS)
    reward -= 0.5 * (position_error / X_LIMIT)
    reward -= 0.01 * (abs(x_dot) + abs(theta_dot))
    return max(reward, -2.0)

def select_action(state, policy_net, device, epsilon, action_dim):
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.max(1)[1].item()

def update_target_network(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())


def optimize_model(policy_net, target_net, memory, optimizer, device, batch_size=64, gamma=0.99):
    if len(memory) < batch_size:
        return
    state, action, reward, next_state, done = memory.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.LongTensor(action).unsqueeze(1).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(done).unsqueeze(1).to(device)

    q_values = policy_net(state).gather(1, action)
    next_q_values = target_net(next_state).max(1)[0].unsqueeze(1)
    expected_q_values = reward + (gamma * next_q_values * (1 - done))

    loss = F.mse_loss(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
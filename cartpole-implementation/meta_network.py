import math
import random as rand
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def build_model(state_dim, lr=1e-3, device=None):
    class metaNetwork(nn.Module):
        def __init__(self, state_size, hidden_size=128):
            super(metaNetwork, self).__init__()
            # neural network with 2 hidden layers
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            # outputs a single continuous action value between -1 and 1
            self.fc3 = nn.Linear(hidden_size, 1)
            # training helpers (populated via configure_trainer)
            self._meta_memory = None
            self._meta_optimizer = None
            self._meta_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._meta_batch_size = 64
            self._meta_gamma = 0.99

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            # Use tanh to bound the output between -1 and 1
            return torch.tanh(self.fc3(x))

        def configure_trainer(self, memory, device=None, lr=1e-3, batch_size=64, gamma=0.99):
            self._meta_memory = memory
            self._meta_device = device or self._meta_device
            self._meta_batch_size = batch_size
            self._meta_gamma = gamma
            self._meta_optimizer = optim.Adam(self.parameters(), lr=lr)

        def optimize_model(self, memory=None, optimizer=None, device=None, batch_size=None, gamma=None):
            memory = memory or self._meta_memory
            optimizer = optimizer or self._meta_optimizer
            device = device or self._meta_device
            batch_size = batch_size or self._meta_batch_size
            gamma = gamma or self._meta_gamma

            if memory is None or optimizer is None:
                raise ValueError("Meta network trainer is not configured. Call configure_trainer or pass memory/optimizer explicitly.")

            _optimize_step(self, memory, optimizer, device, batch_size, gamma)
        
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

    meta_network = metaNetwork(state_dim, 1)
    meta_memory = ReplayBuffer(10000)
    trainer_device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta_network.configure_trainer(meta_memory, device=trainer_device, lr=lr)

    return meta_network, meta_memory


def sliding_window(data,window_size):
    new_list = []
    length = len(data)
    for i in range(min(window_size,length)):
        new_list.append(data[(length - window_size) + i])

    return new_list

def compute_reward(reward_list):
    reward_list = np.array(reward_list, dtype=float)
    N = len(reward_list)
    total_deltas = 0

    for i in range(N - 1):
        deltas = reward_list[i + 1:] - reward_list[i]
        total_deltas += deltas
    
    meta_reward = np.sum(total_deltas) / N

    return np.array(meta_reward,dtype=float)



def select_action(state, original_action, reward_given, meta_net, device, epsilon, input_dim):
    if np.random.rand() < epsilon:
        return float(np.random.uniform(-1.0, 1.0))

    features = np.asarray(state, dtype=np.float32).flatten()
    features = np.concatenate([
        features,
        np.array([float(original_action), float(reward_given)], dtype=np.float32),
    ])
    if features.shape[0] < input_dim:
        pad = np.zeros(input_dim - features.shape[0], dtype=np.float32)
        features = np.concatenate([features, pad])
    elif features.shape[0] > input_dim:
        features = features[:input_dim]

    state_t = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_t = meta_net(state_t)
    action = float(action_t.squeeze().item())
    return float(np.clip(action, -1.0, 1.0))


def _optimize_step(meta_net, memory, optimizer, device, batch_size=64, gamma=0.99):
    if len(memory) < batch_size:
        return
    state, action, reward, next_state, done = memory.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.LongTensor(action).unsqueeze(1).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(done).unsqueeze(1).to(device)

    q_values = meta_net(state).gather(1, action)
    next_q_values = meta_net(next_state).max(1)[0].unsqueeze(1)
    expected_q_values = reward + (gamma * next_q_values * (1 - done))

    loss = F.mse_loss(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def optimize_model(meta_net, memory, optimizer, device, batch_size=64, gamma=0.99):
    """Module-level helper retained for backward compatibility."""
    _optimize_step(meta_net, memory, optimizer, device, batch_size, gamma)


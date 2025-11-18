import math
import random as rand
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_DEFAULT_META_LR = 1e-3


def build_model(state_dim):
    class metaNetwork(nn.Module):
        def __init__(self, state_size, hidden_size=128):
            super(metaNetwork, self).__init__()
            # neural network with 2 hidden layers
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            # outputs a single continuous action value between -1 and 1
            self.fc3 = nn.Linear(hidden_size, 1)
            self._meta_optimizer = None

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            # Use tanh to bound the output between -1 and 1
            return torch.tanh(self.fc3(x))
        
    class ReplayBuffer:
        def __init__(self, capacity, feature_dim):
            self.capacity = capacity
            self.buffer = []
            self.position = 0
            self.feature_dim = feature_dim

        def _format_state(self, array_like):
            vector = np.asarray(array_like, dtype=np.float32).flatten()
            if vector.shape[0] < self.feature_dim:
                pad = np.zeros(self.feature_dim - vector.shape[0], dtype=np.float32)
                vector = np.concatenate([vector, pad])
            elif vector.shape[0] > self.feature_dim:
                vector = vector[: self.feature_dim]
            return vector

        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (
                self._format_state(state),
                action,
                np.float32(reward),
                self._format_state(next_state),
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
    meta_memory = ReplayBuffer(10000, state_dim)
    return meta_network, meta_memory


def sliding_window(data,window_size):
    return data[-(min(len(data),window_size)):]

def compute_reward(reward_list):
    reward_list = np.array(reward_list, dtype=float)
    N = len(reward_list)
    if N < 2:
        return np.array(0.0, dtype=float)

    deltas = np.diff(reward_list)
    meta_reward = np.mean(deltas)

    return np.array(meta_reward, dtype=float)



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


def _ensure_optimizer(meta_net, lr=_DEFAULT_META_LR):
    if getattr(meta_net, "_meta_optimizer", None) is None:
        meta_net._meta_optimizer = optim.Adam(meta_net.parameters(), lr=lr)
    return meta_net._meta_optimizer


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


def optimize_model(meta_net, memory, device, optimizer=None, batch_size=64, gamma=0.99, lr=_DEFAULT_META_LR):
    """Module-level helper for training the meta network."""
    optimizer = optimizer or _ensure_optimizer(meta_net, lr)
    _optimize_step(meta_net, memory, optimizer, device, batch_size, gamma)




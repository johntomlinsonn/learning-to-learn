import math
import random as rand
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

_DEFAULT_META_LR = 1e-3
_SHAPING_MIN = -1.0
_SHAPING_MAX = 1.0


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

        def push(self, state, target_value):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (
                self._format_state(state),
                np.float32(target_value),
            )
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            batch_size = min(batch_size, len(self.buffer))
            batch = rand.sample(self.buffer, batch_size)
            state, target = map(np.stack, zip(*batch))
            return state, target

        def __len__(self):
            return len(self.buffer)

    meta_network = metaNetwork(state_dim)
    meta_memory = ReplayBuffer(10000, state_dim)
    return meta_network, meta_memory


def sliding_window(data, window_size):
    return data[-(min(len(data), window_size)) :]

def compute_reward(reward_list):
    reward_list = np.array(reward_list, dtype=float)
    if reward_list.size < 2:
        return 0.0

    deltas = np.diff(reward_list)
    meta_reward = np.mean(deltas)

    return float(meta_reward)


def _prepare_features(state, original_action, reward_given, input_dim):
    features = np.asarray(state, dtype=np.float32).flatten()
    augments = np.array([float(original_action), float(reward_given)], dtype=np.float32)
    features = np.concatenate([features, augments])
    if features.shape[0] < input_dim:
        pad = np.zeros(input_dim - features.shape[0], dtype=np.float32)
        features = np.concatenate([features, pad])
    elif features.shape[0] > input_dim:
        features = features[:input_dim]
    return features


def select_action(state, original_action, reward_given, meta_net, device, epsilon, input_dim, return_features=False):
    features = _prepare_features(state, original_action, reward_given, input_dim)
    state_t = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        prediction = meta_net(state_t).squeeze().item()
    shaping_value = float(np.clip(prediction, _SHAPING_MIN, _SHAPING_MAX))
    if np.random.rand() < epsilon:
        shaping_value = float(np.random.uniform(_SHAPING_MIN, _SHAPING_MAX))
    if return_features:
        return shaping_value, features
    return shaping_value


def _ensure_optimizer(meta_net, lr=_DEFAULT_META_LR):
    if getattr(meta_net, "_meta_optimizer", None) is None:
        meta_net._meta_optimizer = optim.Adam(meta_net.parameters(), lr=lr)
    return meta_net._meta_optimizer


def _optimize_step(meta_net, memory, optimizer, device, batch_size=64):
    if len(memory) == 0:
        return
    state, target = memory.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    target = torch.FloatTensor(target).unsqueeze(1).to(device)

    prediction = meta_net(state)
    loss = F.mse_loss(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(meta_net.parameters(), 1.0)
    optimizer.step()


def optimize_model(meta_net, memory, device, optimizer=None, batch_size=64, lr=_DEFAULT_META_LR):
    """Module-level helper for training the meta network."""
    if len(memory) == 0:
        return
    optimizer = optimizer or _ensure_optimizer(meta_net, lr)
    _optimize_step(meta_net, memory, optimizer, device, batch_size)



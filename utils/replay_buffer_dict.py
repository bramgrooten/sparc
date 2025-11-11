import numpy as np
import torch


class ReplayBufferDict:
    def __init__(self, capacity, obs_shape, context_shape, history_shape,
                 action_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs_buf          = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.context_buf      = np.zeros((capacity, *context_shape), dtype=np.float32)
        self.history_buf      = np.zeros((capacity, *history_shape), dtype=np.float32)
        self.next_obs_buf     = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_context_buf = np.zeros((capacity, *context_shape), dtype=np.float32)
        self.actions_buf      = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards_buf      = np.zeros((capacity,), dtype=np.float32)
        self.dones_buf        = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, context, history, next_obs, next_context, action, reward, done):
        idx = self.ptr % self.capacity
        self.obs_buf[idx]          = obs
        self.context_buf[idx]      = context
        self.history_buf[idx]      = history
        self.next_obs_buf[idx]     = next_obs
        self.next_context_buf[idx] = next_context
        self.actions_buf[idx]      = action
        self.rewards_buf[idx]      = reward
        self.dones_buf[idx]        = float(done)
        self.ptr  += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs":          torch.tensor(self.obs_buf[idx], dtype=torch.float32, device=self.device),
            "context":      torch.tensor(self.context_buf[idx], dtype=torch.float32, device=self.device),
            "history":      torch.tensor(self.history_buf[idx], dtype=torch.float32, device=self.device),
            "next_obs":     torch.tensor(self.next_obs_buf[idx], dtype=torch.float32, device=self.device),
            "next_context": torch.tensor(self.next_context_buf[idx], dtype=torch.float32, device=self.device),
            "actions":      torch.tensor(self.actions_buf[idx], dtype=torch.float32, device=self.device),
            "rewards":      torch.tensor(self.rewards_buf[idx], dtype=torch.float32, device=self.device).view(-1, 1),
            "dones":        torch.tensor(self.dones_buf[idx], dtype=torch.float32, device=self.device).view(-1, 1),
        }
        return batch


class ReplayBufferExpert:
    """Simple replay buffer for expert training.

    Stores observations, contexts, next observations, next contexts, actions,
    rewards and done flags.  Does not store histories.
    """

    def __init__(self, capacity, obs_shape, context_shape, action_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.context_buf = np.zeros((capacity, *context_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_context_buf = np.zeros((capacity, *context_shape), dtype=np.float32)
        self.actions_buf = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards_buf = np.zeros((capacity,), dtype=np.float32)
        self.dones_buf = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, context, next_obs, next_context, action, reward, done):
        idx = self.ptr % self.capacity
        self.obs_buf[idx] = obs
        self.context_buf[idx] = context
        self.next_obs_buf[idx] = next_obs
        self.next_context_buf[idx] = next_context
        self.actions_buf[idx] = action
        self.rewards_buf[idx] = reward
        self.dones_buf[idx] = float(done)
        self.ptr += 1
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs": torch.tensor(self.obs_buf[idx], dtype=torch.float32, device=self.device),
            "context": torch.tensor(self.context_buf[idx], dtype=torch.float32, device=self.device),
            "next_obs": torch.tensor(self.next_obs_buf[idx], dtype=torch.float32, device=self.device),
            "next_context": torch.tensor(self.next_context_buf[idx], dtype=torch.float32, device=self.device),
            "actions": torch.tensor(self.actions_buf[idx], dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(self.rewards_buf[idx], dtype=torch.float32, device=self.device).view(-1, 1),
            "dones": torch.tensor(self.dones_buf[idx], dtype=torch.float32, device=self.device).view(-1, 1),
        }
        return batch

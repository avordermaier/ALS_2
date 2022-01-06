import torch
from collections import deque, namedtuple
import random
import numpy as np


class ExperienceReplay:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1, parallel_env=1):
        """Initialize a ExperienceReplay object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.parallel_env = parallel_env
        self.experience = namedtuple("Experience", field_names=["state", "next_state", "action", "reward", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0

    def add(self, state, next_state, action, reward, done):
        """Add a new experience to memory."""
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append((state, next_state, action, reward, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        self.iter_ += 1

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma ** idx * n_step_buffer[idx][2]

        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, next_states, actions, rewards, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# Init with
# self.memory = ExperienceReplay(BUFFER_SIZE, batch_size, device, seed, gamma, n_step)
# memory.add(state, action, reward, next_state, done)
from collections import deque, namedtuple
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, size, minimum, multi_step, gamma):
        self.size = size
        self.minimum = minimum
        # 'deque' is Doubly Ended Queuewhcih we use when we need quicker append and pop operations
        # from both the ends of container - https://docs.python.org/2.5/lib/deque-objects.html
        self.memory = deque(maxlen=size)
        # For multi_step we have to go multi_step number of transitions from one we decided to sample if it
        # is possible (if its not done). After iterating, we need to remember last state, total rewards
        self.multi_step = multi_step
        # We will calculate each reward as reward*gamma^i
        self.gamma = gamma
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def store(self, state, next_state, action, reward, done):
        transition = self.Transition(state=state, action=action, next_state=next_state, reward=reward, done=done)
        self.memory.append(transition)

    def sample(self, batch_size):
        chosen_transitions = random.sample(range(0, len(self.memory) - 1), batch_size)

        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        for transition in chosen_transitions:
            i = 0
            total_reward = 0
            new_done = self.memory[transition].done  # Test and Delete!
            new_next_state = self.memory[transition].next_state  # Test and Delete!

            for i in range(self.multi_step):
                if transition + i < len(self.memory):
                    total_reward += self.memory[transition + i].reward * (self.gamma ** i)
                    new_done = self.memory[transition + i].done
                    new_next_state = self.memory[transition + i].next_state
                    # If we reached end of game dont look for more look ahead states
                    if self.memory[transition + i].done:
                        i = self.multi_step

            states.append(self.memory[transition].state)
            actions.append(self.memory[transition].action)
            next_states.append(new_next_state)
            rewards.append(total_reward)
            dones.append(new_done)

        return np.array(states, dtype=np.float32), np.array(next_states, dtype=np.float32),np.array(actions, dtype=np.int64), \
               np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8)


class ExperienceReplayMemory(object):
    """
    Uniformly sampling experience replay memory
    """

    def __init__(self, capacity, gamma, n_step=1):
        self.memory = deque([], maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.experience = namedtuple("Experience", field_names=["state", "next_state", "action", "reward", "done"])
        self.n_step_buffer = deque([], maxlen=self.n_step)

    def __len__(self):
        return len(self.memory)

    def store(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()
        # without n_step
        # self.memory.append(self.experience(state, next_state, action, reward, done))

        # wit n_step
        self.n_step_buffer.append((state, next_state, action, reward, done))
        if len(self.n_step_buffer) == self.n_step:
            state, next_state, action, reward, done = self.calc_multistep_return()
            e = self.experience(state, next_state, action, reward, done)
            self.memory.append(e)

    def sample(self, batch_size):
        # TODO: uniformly sample batches of Tensors for: state, next_state, action, reward, done
        # ...
        transitions = random.sample(self.memory, batch_size)
        state = [t.state for t in transitions]
        next_state = [t.next_state for t in transitions]
        action = np.asarray([t.action for t in transitions])
        reward = np.asarray([t.reward for t in transitions])
        done = np.asarray([t.done for t in transitions])
        return np.array(state), np.array(next_state), np.array(action), np.array(reward), np.array(done).astype(
            np.uint8)

    def calc_multistep_return(self):
        n_step_return = 0
        for idx in range(self.n_step):
            n_step_return += self.gamma ** idx * self.n_step_buffer[idx][3]

        # state, next_state, action, reward, done
        return self.n_step_buffer[0][0], self.n_step_buffer[-1][1], self.n_step_buffer[0][2], n_step_return, \
               self.n_step_buffer[-1][4]


class PrioritizedReplayMemory(object):
    """
    Proportional prioritization sampling experience replay memory
    """

    def __init__(self, capacity, gamma, n_step=1, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha  # alpha determines how much prioritization is used, with α = 0 corresponding to the uniform case
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # for beta calculation
        self.capacity = capacity
        self.memory = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=self.n_step)
        self.gamma = gamma
        self.experience = namedtuple("Experience", field_names=["state", "next_state", "action", "reward", "done"])

    def __len__(self):
        return len(self.memory)

    def calc_multistep_return(self):
        n_step_return = 0
        for idx in range(self.n_step):
            n_step_return += self.gamma ** idx * self.n_step_buffer[idx][3]

        # state, next_state, action, reward, done
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[-1][2], n_step_return, \
               self.n_step_buffer[-1][4]

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def store(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()
        # without n_step
        # self.memory.append(self.experience(state, next_state, action, reward, done))

        # wit n_step & prioritized replay memory
        self.n_step_buffer.append((state, next_state, action, reward, done))
        if len(self.n_step_buffer) == self.n_step:
            state, next_state, action, reward, done = self.calc_multistep_return()

        max_prio = self.priorities.max() if self.memory else 1.0  # gives max priority if memory is not empty else 1

        if len(self.memory) < self.capacity:
            self.memory.append((state, next_state, action, reward, done))
        else:
            # puts the new data on the position of the oldest since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory
            self.memory[self.pos] = (state, next_state, action, reward, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity  # pos circle in the ranges of capacity if pos+1 > cap --> new posi=0

    def sample(self, batch_size):
        len_memory = len(self.memory)
        if len_memory == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        p = probs / probs.sum()

        # gets the indices depending on the probability p
        indices = np.random.choice(len_memory, batch_size, p=p)
        samples = [self.memory[i] for i in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (len_memory * p[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, next_states, actions, rewards, dones = zip(*samples)
        return np.array(states), np.array(next_states), np.array(actions), np.array(rewards), np.array(dones).astype(
            np.uint8), np.array(indices), np.array(weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for i, prio in zip(batch_indices, batch_priorities):
            i = int(i.item())
            prio = prio.item()
            self.priorities[i] = prio

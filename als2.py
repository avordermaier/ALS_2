import sys, os
import gym
import numpy as np
import random
import torch
from torch import nn
import torchvision
import copy
from gym.wrappers import FrameStack
from gym.spaces import Box
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import plot_results, createDir, save_weights, save_loss, save_rewards, plot_epsilon
from ExperienceReplay import ExperienceReplayMemory, PrioritizedReplayMemory


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        transform = torchvision.transforms.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(self.shape),
                                                     torchvision.transforms.Normalize(0, 255)])
        return transforms(observation).squeeze(0)


run_as_ddqn = False
device = None
path = os.path.join(os.getcwd(), 'Breakoutv4')
loss_function = 'huber'# 'mse'

env = gym.make("BreakoutNoFrameskip-v4")
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
image_stack, h, w = env.observation_space.shape
num_actions = env.action_space.n
action_list = np.arange(num_actions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 61
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Evaluating
epsilons = []
mean_training_rewards = []
mean_reward = 0
reward_threshold = 20
best_ep_avg = 0


# Parameters
batch_size = 32
alpha = 0.00025
gamma = 0.99
eps, eps_decay = 1.0, 0.999
max_train_episodes = 100
max_test_episodes = 100
max_train_frames = 1000
burn_in_phase = 200
sync_target = 100
curr_step = 0
n_step = 1
with_prio_replay = False
buffer_size = 500
# Replay memory
if with_prio_replay:
    buffer = PrioritizedReplayMemory(buffer_size, gamma=gamma, n_step=n_step)
else:
    buffer = ExperienceReplayMemory(buffer_size, gamma, n_step)


def convert(x):
    return torch.tensor(x.__array__()).float()


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        print('Input shape: {}'.format(input_shape))
        print('Num_actions: {}'.format(n_actions))
        print('Conv_out_size: {}'.format(conv_out_size))

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        print('O shape: {}'.format(o.shape))
        return int(np.prod(o.shape))

    def forward(self, x):
        conv_out = self.conv(x).view(x.shape[0], -1)
        return self.fc(conv_out)


# TODO: create an online and target DQN (Hint: Use copy.deepcopy() and requires_grad utilities!)

online_dqn = DQN(env.observation_space.shape, num_actions)
target_dqn = copy.deepcopy(online_dqn)
online_dqn = online_dqn.to(device)
target_dqn = target_dqn.to(device)

print('Model')
print(online_dqn)

# TODO: create the appropriate MSE criterion and Adam optimizer
# ...
optimizer = torch.optim.Adam(online_dqn.parameters(), lr=alpha)
criterion = F.mse_loss
huber_loss = torch.nn.SmoothL1Loss()


def policy(state, is_training):
    global eps
    state = convert(state).unsqueeze(0).to(device)

    # TODO: Implement an epsilon-greedy policy
    if not is_training:
        eps = 0

    if np.random.random() < eps:
        action = np.random.choice(action_list)
    else:
        qvals = online_dqn(state)
        action = torch.max(qvals, dim=-1)[1].item()

    return action


def compute_loss(state, action, reward, next_state, done):
    state = convert(state).type(torch.float32).to(device)
    next_state = convert(next_state).type(torch.float32).to(device)
    action = torch.tensor(action, dtype=torch.int64).to(device).unsqueeze(-1)
    reward = torch.tensor(reward, dtype=torch.float32).to(device).unsqueeze(-1)
    done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(-1)

    # Get max predicted Q values (for next states) from target model
    q_targets_next = target_dqn(next_state).detach()
    if run_as_ddqn:
        best_action = torch.argmax(online_dqn(next_state), dim=-1)
        q_targets_next = q_targets_next.gather(1, torch.tensor(best_action, dtype=torch.int64, device=device).unsqueeze(1))

    else:
        q_targets_next = q_targets_next.max(1)[0]
        q_targets_next = q_targets_next.unsqueeze(1)

    # Compute Q targets for current states
    # note n_step for multistep learning
    q_targets = reward + (gamma ** n_step * q_targets_next * (1 - done))
    # Get expected Q values from local model

    q_expected = online_dqn(state).gather(1, action.type(torch.int64))
    # action_q_values = torch.gather(online_dqn(state), dim=1, index=action)

    if loss_function == 'mse':
        loss = criterion(q_expected, q_targets)
    elif loss_function == 'huber':
        loss = huber_loss(q_expected, q_targets)
    else: raise Exception(loss_function)

    # TODO: Return the loss computed using the criterion.
    return loss


def compute_loss_prio(state, action, reward, next_state, done, indices, weights):
    state = convert(state).type(torch.float32).to(device)
    next_state = convert(next_state).type(torch.float32).to(device)
    action = torch.tensor(action, dtype=torch.int64).to(device).unsqueeze(-1)
    reward = torch.tensor(reward, dtype=torch.float32).to(device).unsqueeze(-1)
    done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(-1)
    indices = torch.tensor(indices, dtype=torch.float32).to(device).unsqueeze(-1)
    weights = torch.tensor(weights, dtype=torch.float32).to(device).unsqueeze(-1)

    # Get max predicted Q values (for next states) from target model
    q_targets_next = target_dqn(next_state).detach()
    if run_as_ddqn:
        best_action = torch.argmax(online_dqn(next_state), dim=-1)
        q_targets_next = q_targets_next.gather(1, torch.tensor(best_action, dtype=torch.int64, device=device).unsqueeze(1))

    else:
        q_targets_next = q_targets_next.max(1)[0]
        q_targets_next = q_targets_next.unsqueeze(1)

    # Compute Q targets for current states
    # note n_step for multistep learning
    q_targets = reward + (gamma ** n_step * q_targets_next * (1 - done))
    # Get expected Q values from local model

    q_expected = online_dqn(state).gather(1, action.type(torch.int64))
    # action_q_values = torch.gather(online_dqn(state), dim=1, index=action)

    td_error = q_targets - q_expected
    #mse
    loss = (td_error.pow(2) * weights).mean().to(device)
    buffer.update_priorities(indices, abs(td_error.detach().numpy()))

    # TODO: Return the loss computed using the criterion.
    return loss


def run_episode(curr_step, buffer, is_training, is_rendering=False):
    global eps
    episode_reward, episode_loss = 0, 0.
    state = env.reset()
    if is_rendering:
        env.render("rgb_array")

    for t in range(max_train_frames):
        action = policy(state, is_training)
        curr_step += 1

        next_state, reward, done, _ = env.step(action)
        if is_rendering:
            env.render("rgb_array")

        episode_reward += reward

        if is_training:
            buffer.store(state, next_state, action, reward, done)

            if curr_step > burn_in_phase:
                if with_prio_replay:
                    state_batch, next_state_batch, action_batch, reward_batch, done_batch, indices, weights = buffer.sample(batch_size)
                else:
                    state_batch, next_state_batch, action_batch, reward_batch, done_batch = buffer.sample(batch_size)

                if curr_step % sync_target == 0:
                    # TODO: Periodically update your target_dqn at each sync_target frames
                    target_dqn.load_state_dict(online_dqn.state_dict())

                if with_prio_replay:
                    loss = compute_loss_prio(state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                                             indices, weights)
                else:
                    loss = compute_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()
        else:
            with torch.no_grad():
                episode_loss += compute_loss(np.array([state]), np.array([action]), reward, np.array([next_state]),
                                             done).item()

        state = next_state

        if done:
            break

    return dict(reward=episode_reward, loss=episode_loss / t)


def update_metrics(metrics, episode):
    for k, v in episode.items():
        metrics[k].append(v)


def print_metrics(it, metrics, is_training, window=100):
    global best_ep_avg
    reward_mean = np.mean(metrics['reward'][-window:])
    loss_mean = np.mean(metrics['loss'][-window:])
    mode = "train" if is_training else "test"
    print(f"Episode {it:4d} | {mode:5s} | reward {reward_mean:5.5f} | loss {loss_mean:5.5f}")

    mean_training_rewards.append(reward_mean)
    if is_training and max(mean_training_rewards) > best_ep_avg:
        best_ep_avg = max(mean_training_rewards)
        print("\nNew best {} episode average: {:.2f}".format(window, best_ep_avg))
        save_weights(path, online_dqn, file_name="ep_{}_dqn_weights".format(it))

    return reward_mean


createDir(path)
train_metrics = dict(reward=[], loss=[])
for it in range(max_train_episodes):
    episode_metrics = run_episode(curr_step, buffer, is_training=True)
    update_metrics(train_metrics, episode_metrics)
    if it % 10 == 0:
        mean_reward = print_metrics(it, train_metrics, is_training=True)
    eps *= eps_decay
    epsilons.append(eps)
    if mean_reward >= reward_threshold:
        print('\nEnvironment solved in {} steps!'.format(it))
        break

save_loss(path, 'ddqn' if run_as_ddqn else 'dqn', train_metrics['loss'])
save_rewards(path, 'ddqn' if run_as_ddqn else 'dqn', train_metrics['reward'])


plt.plot(train_metrics['reward'])
x = [i for i in range(len(train_metrics['reward']))]
plt.ylabel('Training Reward')
plt.xlabel('Number of Episodes')
amax = np.argmax(train_metrics['reward'])
xlim, ylim = plt.xlim(), plt.ylim()
plt.plot([x[amax], x[amax], xlim[0]], [xlim[0], train_metrics['reward'][amax], train_metrics['reward'][amax]],
          linestyle="--")
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

plt.plot(train_metrics['loss'])
plt.ylabel('Training Loss')
plt.xlabel('Number of Episodes')
plt.show()

plot_epsilon(epsilons)

test_metrics = dict(reward=[], loss=[])
curr_step = 0
for it in range(max_test_episodes):
    episode_metrics = run_episode(curr_step, buffer, is_training=False)
    update_metrics(test_metrics, episode_metrics)
    print_metrics(it + 1, test_metrics, is_training=False)

# TODO: Plot your train_metrics and test_metrics
# ...


plt.plot(test_metrics['reward'])
plt.ylabel('Testing Reward')
plt.xlabel('Number of Episodes')
plt.show()

plt.plot(test_metrics['loss'])
plt.ylabel('Testing Reward')
plt.xlabel('Number of Episodes')
plt.show()

plot_results(train_metrics, test_metrics)

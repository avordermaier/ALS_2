import torch
import torch.nn as nn

import numpy as np


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

# class DQN(nn.Module):
#     # nature paper architecture
#
#     def __init__(self, in_channels, num_actions):
#         super().__init__()
#
#         network = [
#             torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_actions)
#         ]
#
#         self.network = nn.Sequential(*network)
#
#     def forward(self, x):
#         actions = self.network(x)
#         return actions


class NatureCNN(nn.Module):
    # nature paper architecture

    def __init__(self, env):
        super().__init__()

        observation_space = env.observation_space
        num_actions = env.action_space.n
        image_stack = observation_space.shape[0]

        cnn = nn.Sequential(
            torch.nn.Conv2d(image_stack, 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).floaat()).shape[1]

        self.net = nn.Sequential(cnn, nn.Linear(n_flatten, 512), nn.ReLU(), nn.Linear(512, num_actions))

    def forward(self, x):
        actions = self.net(x)
        return actions

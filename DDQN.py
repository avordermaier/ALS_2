import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class DDQN(nn.Module):
    def __init__(self, image_stack, state_size, num_actions, layer_size, seed):
        super(DDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.num_actions = num_actions
        self.state_dim = len(state_size)
        if self.state_dim == 3:
            self.cnn_1 = nn.Conv2d(image_stack, out_channels=32, kernel_size=8, stride=4)
            self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            weight_init([self.cnn_1, self.cnn_2, self.cnn_3])

            self.ff_1 = nn.Linear(self.calc_input_layer(), layer_size)
            self.ff_2 = nn.Linear(layer_size, num_actions)
            weight_init([self.ff_1])
        elif self.state_dim == 1:

            self.head_1 = nn.Linear(self.input_shape[0], layer_size)
            self.ff_1 = nn.Linear(layer_size, layer_size)
            self.ff_2 = nn.Linear(layer_size, num_actions)
            weight_init([self.head_1, self.ff_1])
        else:
            print("Unknown input dimension!")

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]

    def forward(self, input):
        """

        """
        if self.state_dim == 3:
            x = torch.relu(self.cnn_1(input))
            x = torch.relu(self.cnn_2(x))
            x = torch.relu(self.cnn_3(x))
            x = x.view(input.size(0), -1)
        else:
            x = torch.relu(self.head_1(input))

        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out


class TestingDQN(nn.Module):
    def __init__(self, action_size, input_shape=(4, 84, 84)):
        super(TestingDQN, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
        self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.ff_1 = nn.Linear(self.calc_input_layer(), 512)
        self.ff_2 = nn.Linear(512, action_size)

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]

    def forward(self, x):
        """

        """
        x = torch.relu(self.cnn_1(x))
        x = torch.relu(self.cnn_2(x))
        x = torch.relu(self.cnn_3(x))
        x = x.flatten()
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out
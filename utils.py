import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
import torch
from datetime import datetime


def plot_results(train_metrics, test_metrics, window=100):
    fig, ax_list = plt.subplots(2, 2)
    ax_list[0, 0].set_title('Training')
    ax_list[0, 0].plot(train_metrics['reward'], lw=2, color='green')
    ax_list[0, 0].set_ylabel('Acc. Reward')
    # ax_list[1,0].plot(time_list[:N_trial], lw=2, color='blue')
    # ax_list[1,0].set_ylabel('Trial duration')
    ax_list[1, 0].plot(train_metrics['loss'], lw=2, color='red')
    # ax_list[2,0].set_ylabel('Bellman error')
    ax_list[1, 0].set_xlabel('Episode')
    ax_list[1, 0].set_ylabel('Loss')
    ax_list[1, 0].set_yscale('log')

    ax_list[0, 1].set_title('Testing')
    ax_list[0, 1].plot(test_metrics['reward'], lw=2, color='green')
    ax_list[0, 1].set_ylabel('Acc. Reward')
    ax_list[0, 1].axhline(np.mean(test_metrics['reward']), color='black', linestyle='dashed', lw=4)
    # ax_list[0,1].set_ylim(ax_list[0,0].get_ylim())

    ax_list[1, 1].plot(test_metrics['loss'], lw=2, color='red')
    ax_list[1, 1].axhline(np.mean(test_metrics['loss']), lw=4, color='black', linestyle='dashed')
    # ax_list[2,0].set_ylabel('Bellman error')
    ax_list[1, 1].set_xlabel('Episode')
    ax_list[1, 1].set_yscale('log')
    ax_list[1, 1].set_ylabel('Loss')

    # ax_list[1,1].plot(time_list[N_trial:], lw=2, color='blue')
    # ax_list[1,1].set_ylabel('Trial duration')
    # ax_list[1,1].axhline(np.mean(time_list[N_trial:]),color='black',linestyle='dashed',lw=4)
    # ax_list[1,1].set_ylim(ax_list[1,0].get_ylim())

    # ax_list[2,1].plot(err_list[N_trial:], lw=2, color='red')
    # ax_list[2,1].set_ylabel('Bellman error')
    # ax_list[2,1].axhline(np.mean(err_list[N_trial:]),color='black',linestyle='dashed',lw=4)
    # ax_list[2,1].set_ylim(ax_list[2,0].get_ylim())
    # ax_list[2,1].set_yscale('log')

    # ax_list[2,1].set_xlabel('Trial number')
    plt.show()


def plot_epsilon(eps):
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.plot(eps)
    x = [i for i in range(len(eps))]
    amin = np.argmin(eps)
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.plot([x[amin], x[amin], xlim[0]], [xlim[0], eps[amin], eps[amin]],
             linestyle="--")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()



def createDir(path):
    if os.path.exists(path) is False:
        os.makedirs(path, exist_ok=True)


def save_rewards(path, model, rewards):
    time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    file_name = f'{model}_training_rewards_{time}.txt'
    file_name = os.path.join(path, file_name)
    if os.path.exists(file_name):
        file = open(file_name, 'a')
    else:
        file = open(file_name, 'w')
        file.write('reward\n')
    for x in rewards:
        file.write('{}\n'.format(x))
    file.close()

def save_loss(path, model, loss):
    time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    file_name = f'{model}_training_loss_{time}.txt'
    file_name = os.path.join(path, file_name)
    if os.path.exists(file_name):
        file = open(file_name, 'a')
    else:
        file = open(file_name, 'w')
        file.write('loss\n')
    for x in loss:
        file.write('{}\n'.format(x))
    file.close()


def save_weights(path, online_dqn, file_name=None):
    time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    file_name = f'{file_name}_{time}.pt'
    if file_name is None:
        file_name = 'dqn_weights.pt'
    weights_path = os.path.join(path, file_name)
    torch.save(online_dqn.state_dict(), weights_path)



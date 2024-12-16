import gym
import torch
from utils import moving_average
import matplotlib.pyplot as plt
from gym.spaces import Discrete, Box
import random

def create_env(env_name):
    env = gym.make(env_name)
    #seed=random.randint(0,10000)
    env.reset()
    torch.manual_seed(0)
    return env

def show(return_list,env_name,policy_name):
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(policy_name+' on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(policy_name+' on {}'.format(env_name))
    plt.show()

def check_environment_type(space):
    # 检查动作空间类型
    if isinstance(space, Discrete):
        action_type = 'discrete'
    elif isinstance(space, Box):
        action_type = 'continuous'
    else:
        action_type = 'unknown'
    # 检查观察空间类型
    if isinstance(space, Discrete):
        observation_type = 'discrete'
    elif isinstance(space, Box):
        observation_type = 'continuous'
    else:
        observation_type = 'unknown'
    return action_type, observation_type
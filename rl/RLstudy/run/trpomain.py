from utils import train_on_policy_agent
import torch
from policy.trpo import TRPO
from env import create_env,show

#离散动作空间参数
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
critic_lr = 1e-2
kl_constraint = 0.0005
alpha = 0.5

# #连续动作空间参数定义
# num_episodes = 2000
# hidden_dim = 128
# gamma = 0.9
# lmbda = 0.9
# critic_lr = 1e-2
# kl_constraint = 0.00005
# alpha = 0.5

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    env_name='CartPole-v0'
    env=create_env(env_name)
    agent = TRPO(hidden_dim, env.observation_space, env.action_space,
                           lmbda, kl_constraint, alpha, critic_lr, gamma, device)
    return_list = train_on_policy_agent(env, agent, num_episodes)
    show(return_list,env_name,'TRPO')

import torch
from policy.ppo import PPO
from utils import train_on_policy_agent
from env import create_env,show

# #离散动作空间参数
# actor_lr = 1e-3
# critic_lr = 1e-2
# num_episodes = 500
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# epochs = 10
# eps = 0.2
# env_name='CartPole-v0'

#连续动作空间参数
actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
env_name='Pendulum-v0'

#if __name__=='main':
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
"cpu")
env=create_env(env_name)

agent = PPO( hidden_dim,env.observation_space, env.action_space, actor_lr, critic_lr, lmbda,
        epochs, eps, gamma, device)
return_list = train_on_policy_agent(env, agent, num_episodes)

show(return_list,env_name,'PPO')
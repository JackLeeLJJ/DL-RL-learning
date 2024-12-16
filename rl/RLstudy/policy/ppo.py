import torch
import torch.nn.functional as F
from utils import compute_advantage
from env import check_environment_type

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self,hidden_dim, state_space, action_space, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        state_dim = state_space.shape[0]
        action_type, observation_type = check_environment_type(action_space)
        if action_type == 'continuous':
            action_dim = action_space.shape[0]
            self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                             action_dim).to(device)
        elif action_type == 'discrete':
            action_dim = action_space.n
            self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)

        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        output = self.actor(state)
        if isinstance(output, tuple):  # 如果是元组，通常是 (mu, std)
            mu, sigma = output
            action_dist = torch.distributions.Normal(mu, sigma)
            action = action_dist.sample()
            return [action.item()]
        else:
            probs = output
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)

        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        output=self.actor(states)
        if isinstance(output,tuple):
            mu,std =output
            action_dists=torch.distributions.Normal(mu.detach(),std.detach())
            old_log_probs=action_dists.log_prob(actions)
        else:
            old_log_probs = torch.log(output.gather(1,actions)).detach()

        for _ in range(self.epochs):
            output = self.actor(states)
            if isinstance(output, tuple):
                mu, std = output
                action_dists = torch.distributions.Normal(mu, std)
                log_probs = action_dists.log_prob(actions)
            else:
                log_probs = torch.log(output.gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
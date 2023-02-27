import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from config import *


__all__ = [
    'DQN',
    'PPO'
]


# --------------------------------------
# Deep Q-Network
# --------------------------------------
class DQNReplayBuffer:
    def __init__(self, max_size=np.infty):
        self.max_size = max_size
        self.data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        self.size = 0

    def add_experience(self, **kwargs):
        if self.size == self.max_size:
            for key, value in kwargs.items():
                self.data[key].pop(0)
                self.data[key].append(value)
        else:
            for key, value in kwargs.items():
                self.data[key].append(value)
            self.size += 1

    def random_sample(self, batch_size):
        samples = {}

        index = np.random.choice(self.size, batch_size)
        for key in self.data.keys():
            samples[key] = [self.data[key][i] for i in index]
        return samples


class Q(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer_n, action_n):
        super(Q, self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for l_idx in range(hidden_layer_n):
            self.fn.append(nn.Linear(hidden_dim, hidden_dim))
            self.fn.append(nn.ReLU())
        self.fn.append(nn.Linear(hidden_dim, action_n))

    def forward(self, x):
        return self.fn(x)


class DQN:
    def __init__(self, input_dim, hidden_dim, hidden_layer_n, action_n, lr,
                 discount, greedy, update_target_freq, buffer_size, batch_size):
        self.action_n = action_n
        self.device = CONFIG.DEVICE

        self.buffer = DQNReplayBuffer(buffer_size)

        self.q = Q(input_dim, hidden_dim, hidden_layer_n, action_n).to(self.device)
        self.q_optim = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.q_target = Q(input_dim, hidden_dim, hidden_layer_n, action_n).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.discount = discount
        self.greedy = greedy
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq

        self.time = 0

    def act(self, state):
        if np.random.uniform() < self.greedy:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            return self.q(state).argmax().detach().item()
        else:
            return np.random.choice(self.action_n)

    def update(self):
        samples = self.buffer.random_sample(self.batch_size)

        states = torch.squeeze(torch.stack(samples.get('states'), dim=0)).detach().to(self.device)
        actions = torch.squeeze(torch.stack(samples.get('actions'), dim=0)).detach().to(self.device)
        rewards = torch.squeeze(torch.stack(samples.get('rewards'), dim=0)).detach().to(self.device)
        next_states = torch.squeeze(torch.stack(samples.get('next_states'), dim=0)).detach().to(self.device)
        dones = torch.squeeze(torch.stack(samples.get('dones'), dim=0)).detach().to(self.device)

        q_pred = self.q(states).gather(1, actions.unsqueeze(1))
        q_target = rewards.detach()
        q_target += (1 - dones) * self.discount * self.q_target(next_states).detach().max(dim=1)[0]
        q_target = q_target.unsqueeze(1).detach()

        self.q_optim.zero_grad()
        loss = self.loss_fn(q_pred, q_target)
        loss.backward()
        self.q_optim.step()

        self.time += 1
        if self.time % self.update_target_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def is_able_to_update(self):
        return self.buffer.size > self.batch_size


# --------------------------------------
# Proximal Policy Optimization
# --------------------------------------
class PPOReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_policies = []
        self.state_values = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_policies[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.dones[:]


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer_n, action_n):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for l_idx in range(hidden_layer_n):
            self.actor.append(nn.Linear(hidden_dim, hidden_dim))
            self.actor.append(nn.ReLU())
        self.actor.append(nn.Linear(hidden_dim, action_n))
        self.actor.append(nn.Softmax(dim=-1))

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for l_idx in range(hidden_layer_n):
            self.critic.append(nn.Linear(hidden_dim, hidden_dim))
            self.critic.append(nn.ReLU())
        self.critic.append(nn.Linear(hidden_dim, 1))

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        policy = self.actor(state)
        dist = Categorical(policy)
        action = dist.sample()
        log_policy = dist.log_prob(action)
        state_value = self.critic(state)
        return action.detach(), log_policy.detach(), state_value.detach()

    def evaluate(self, state, action):
        policy = self.actor(state)
        if True in torch.isnan(policy):
            pass
        dist = Categorical(policy)
        log_policy = dist.log_prob(action)
        state_value = self.critic(state)
        dist_entropy = dist.entropy()

        return log_policy, state_value, dist_entropy


class PPO:
    def __init__(self, input_dim, hidden_dim, hidden_layer_n, action_n, lr_actor, lr_critic,
                 gamma,  epsilon, c_critic, c_entropy, k_epochs):
        self.device = CONFIG.DEVICE
        self.gamma = gamma
        self.epsilon = epsilon
        self.c_critic = c_critic
        self.c_entropy = c_entropy
        self.k_epochs = k_epochs

        self.fn = ActorCritic(input_dim, hidden_dim, hidden_layer_n, action_n).to(self.device)
        self.fn_optim = optim.Adam([
                        {'params': self.fn.actor.parameters(), 'lr': lr_actor},
                        {'params': self.fn.critic.parameters(), 'lr': lr_critic}
                    ])

        self.old_fn = ActorCritic(input_dim, hidden_dim, hidden_layer_n, action_n).to(self.device)
        self.old_fn.load_state_dict(self.fn.state_dict())

        self.mse_loss = nn.MSELoss()

        self.buffer = PPOReplayBuffer()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action, log_policy, state_value = self.old_fn.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_policies.append(log_policy)
        self.buffer.state_values.append(state_value)

        return action.item()

    def update(self):
        # Monte Carlo estimation of expectations
        expectations = []
        expectation = 0
        for history_idx, (state, state_value, reward, done) in enumerate(zip(
            reversed(self.buffer.states),
            reversed(self.buffer.state_values),
            reversed(self.buffer.rewards),
            reversed(self.buffer.dones)
        )):
            if done:
                expectation = 0
            expectation = reward + self.gamma * expectation
            expectations.insert(0, expectation)
        # Normalizing expectations
        expectations = torch.tensor(expectations, dtype=torch.float32).to(self.device)
        expectations = (expectations - expectations.mean()) / (expectations.std() + CONFIG.VAREPSILON)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_log_policies = torch.squeeze(torch.stack(self.buffer.log_policies, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # Calculate advantages
        advantages = expectations.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            log_policies, state_values, dist_entropies = self.fn.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(log_policies - old_log_policies.detach())

            actor_loss = -torch.min(
                ratios * advantages.detach(),
                torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            )
            critic_loss = self.mse_loss(state_values, expectations)
            loss = actor_loss + self.c_critic * critic_loss - self.c_entropy * dist_entropies

            self.fn_optim.zero_grad()
            loss.mean().backward()
            self.fn_optim.step()

        self.old_fn.load_state_dict(self.fn.state_dict())
        self.buffer.clear()

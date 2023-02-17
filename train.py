import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

from config import *
from agent import *

np.random.seed(CONFIG.RANDOM_SEED)
torch.manual_seed(CONFIG.RANDOM_SEED)


output_dir = 'outputs/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def train_dqn(env_name, hidden_dim, hidden_layer_n, lr, updates, fig_update_freq):
    env = gym.make(env_name)

    agent = DQN(input_dim=env.observation_space.shape[0],
                hidden_dim=hidden_dim,
                hidden_layer_n=hidden_layer_n,
                action_n=env.action_space.n,
                lr=lr,
                discount=CONFIG.DQN.DISCOUNT,
                greedy=CONFIG.DQN.GREEDY,
                update_target_freq=CONFIG.DQN.UPDATE_TARGET_FREQ,
                buffer_size=CONFIG.DQN.REPLAY_MEMORY_SIZE,
                batch_size=CONFIG.DQN.BATCH_SIZE)

    reward_rec = []
    fig, axs = plt.subplots(1, 1)
    for update_idx in range(1, updates):
        trial_reward = 0
        state, *_ = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, *_ = env.step(action)
            done = terminated

            trial_reward += reward
            agent.buffer.add_experience(
                states=torch.tensor(state, dtype=torch.float32),
                actions=torch.tensor(action, dtype=torch.int64),
                rewards=torch.tensor(reward, dtype=torch.float32),
                next_states=torch.tensor(next_state, dtype=torch.float32),
                dones=torch.tensor(done, dtype=torch.int64)
            )

            if agent.is_able_to_update():
                agent.update()

            if done:
                print('Update: {}/{} Trial Reward: {}'.format(
                    update_idx, updates, trial_reward), end='\r')
                reward_rec.append(trial_reward)
                if update_idx % fig_update_freq == 0:
                    axs.clear()
                    axs.plot(reward_rec, c='k', alpha=0.2)
                    axs.plot(np.convolve(reward_rec,
                                         np.ones(CONFIG.WINDOW_SIZE) / CONFIG.WINDOW_SIZE,
                                         'valid'), c='r')
                    axs.set_xlabel('#Trials')
                    axs.set_ylabel('Trial Reward')
                    if np.max(reward_rec) - np.min(reward_rec) > 1e3:
                        axs.set_yscale('symlog')
                    title = '{}-{}'.format('DQN', env_name)
                    axs.set_title(title)
                    plt.tight_layout()
                    fig.savefig(os.path.join(output_dir, title + '.png'))
                break
            else:
                state = next_state
    env.close()


def train_ppo(env_name, hidden_dim, hidden_layer_n, lr_actor, lr_critic, updates, fig_update_freq):
    env = gym.make(env_name)

    agent = PPO(input_dim=env.observation_space.shape[0],
                hidden_dim=hidden_dim,
                hidden_layer_n=hidden_layer_n,
                action_n=env.action_space.n,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=CONFIG.PPO.GAMMA,
                epsilon=CONFIG.PPO.EPSILON,
                c_critic=CONFIG.PPO.C_CRITIC,
                c_entropy=CONFIG.PPO.C_ENTROPY,
                k_epochs=CONFIG.PPO.K_EPOCHS)
    reward_rec = []
    fig, axs = plt.subplots(1, 1)
    for update_idx in range(1, updates):
        trial_reward = 0
        state, *_ = env.reset()

        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, *_ = env.step(action)
            done = terminated

            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)

            trial_reward += reward

            if done:
                agent.update()
                print('Update: {}/{} Trial Reward: {}'.format(
                    update_idx, updates, trial_reward), end='\r')
                reward_rec.append(trial_reward)
                if update_idx % fig_update_freq == 0:
                    axs.clear()
                    axs.plot(reward_rec, c='k', alpha=0.2)
                    axs.plot(np.convolve(reward_rec,
                                         np.ones(CONFIG.WINDOW_SIZE) / CONFIG.WINDOW_SIZE,
                                         'valid'), c='r')
                    axs.set_xlabel('#Trials')
                    axs.set_ylabel('Trial Reward')
                    if np.max(reward_rec) - np.min(reward_rec) > 1e3:
                        axs.set_yscale('symlog')
                    title = '{}-{}'.format('PPO', env_name)
                    axs.set_title(title)
                    plt.tight_layout()
                    fig.savefig(os.path.join(output_dir, title + '.png'))
                break
            else:
                state = next_state
    env.close()


if __name__ == '__main__':
    train_dqn(env_name='CartPole-v1',
              hidden_dim=64,
              hidden_layer_n=1,
              lr=0.00005,
              updates=2000,
              fig_update_freq=10)

    train_dqn(env_name='MountainCar-v0',
              hidden_dim=64,
              hidden_layer_n=3,
              lr=0.00001,
              updates=5000,
              fig_update_freq=10)

    train_ppo(env_name='CartPole-v1',
              hidden_dim=64,
              hidden_layer_n=1,
              lr_actor=0.00001,
              lr_critic=0.0001,
              updates=2000,
              fig_update_freq=50)

    train_ppo(env_name='MountainCar-v0',
              hidden_dim=64,
              hidden_layer_n=3,
              lr_actor=0.00001,
              lr_critic=0.0001,
              updates=5000,
              fig_update_freq=10)

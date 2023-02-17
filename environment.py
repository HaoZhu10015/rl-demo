import gymnasium as gym
import numpy as np

__all__ = [
    'make_env'
]


def make_env(env_name):
    if env_name == 'MountainCar-v0':
        class MountainCarWithRewardWrapper(gym.RewardWrapper):
            def __init__(self):
                super(MountainCarWithRewardWrapper, self).__init__(gym.make('MountainCar-v0'))

            def reward(self, reward):
                position = self.env.state[0]
                return max(0.5, np.abs(position)) + reward

        env = MountainCarWithRewardWrapper()
        return env
    else:
        return gym.make(env_name)

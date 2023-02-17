import torch


__all__ = [
    'CONFIG'
]


class CONFIG:
    RANDOM_SEED = 10015
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VAREPSILON = 1e-7  # small value for numerical stability

    WINDOW_SIZE = 30  # moving average window size for plot performance

    # --------------------------------------
    # DQN
    # --------------------------------------
    class DQN:
        REPLAY_MEMORY_SIZE = 10000
        BATCH_SIZE = 32
        DISCOUNT = 0.99
        GREEDY = 0.9
        UPDATE_TARGET_FREQ = 20

    # --------------------------------------
    # PPO
    # --------------------------------------
    class PPO:
        GAMMA = 0.99  # discount
        EPSILON = 0.2  # for clipped surrogate objective
        C_CRITIC = 0.5  # critic coefficient
        C_ENTROPY = 0.01  # entropy coefficient

        K_EPOCHS = 80

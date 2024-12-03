import numpy as np


def estimate_reward(beliefs: np.ndarray, action: np.ndarray):
    team_interaction = action[:, None] == action
    np.fill_diagonal(team_interaction, False)

    return np.sum(beliefs * team_interaction) / 2


def calculate_distance(preferences: np.ndarray, beliefs: np.ndarray):
    return np.mean((preferences - beliefs) ** 2)

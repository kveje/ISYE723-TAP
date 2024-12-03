import numpy as np


class Individuals:
    def __init__(self, num_individuals, sigma_w=0.1, sigma_f=0.1, sigma_p=1):
        self.num_individuals = num_individuals
        self.preferences = np.random.normal(0, 1, (num_individuals, num_individuals))
        np.fill_diagonal(self.preferences, 0)
        self.sigma_w = sigma_w
        self.sigma_f = sigma_f
        self.sigma_p = sigma_p

    def reset_preferences(self, i):
        self.preferences[i, :] = np.random.normal(0, 1, self.num_individuals)
        self.preferences[:, i] = np.random.normal(0, 1, self.num_individuals)
        np.fill_diagonal(self.preferences, 0)

    def reset(self):
        self.preferences = np.random.normal(
            0, self.sigma_p, (self.num_individuals, self.num_individuals)
        )
        np.fill_diagonal(self.preferences, 0)

    def update_preferences(self):
        self.preferences += np.random.normal(
            0, self.sigma_w, (self.num_individuals, self.num_individuals)
        )
        np.fill_diagonal(self.preferences, 0)

    def collect_feedback(self, team_interaction: np.ndarray):
        # Generate feedback where individuals are in the same team
        feedback = np.where(
            team_interaction,
            self.preferences
            + np.random.normal(
                0, self.sigma_f, size=(self.num_individuals, self.num_individuals)
            ),
            np.nan,
        )

        return feedback

    def calculate_reward(self, team_interaction: np.ndarray):
        return np.sum(team_interaction * self.preferences) / 2

    def get_true_preferences(self):
        return self.preferences


if __name__ == "__main__":
    individuals = Individuals(5)
    action = np.array([0, 1, 0, 1, 0])
    print(individuals.collect_feedback(action))
    print(individuals.get_true_preferences())

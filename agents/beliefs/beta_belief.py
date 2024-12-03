import numpy as np


class BetaBelief:
    def __init__(self, num_individuals: int):
        # Number of individuals
        self.num_individuals = num_individuals

        # Initialize alpha and beta matrices with ones (Beta(1,1) uniform prior)
        self.alpha = np.ones((num_individuals, num_individuals))
        self.beta = np.ones((num_individuals, num_individuals))

    def update(self, feedback_matrix: np.ndarray):
        """
        Update Beta parameters based on feedback.

        Args:
            feedback_matrix (dict): Nested dictionary with feedback values.
                                     feedback_matrix[i][j] = feedback from i about j.
        """
        self.alpha += feedback_matrix
        self.beta += 1 - feedback_matrix

    def get_means(self):
        """
        Retrieve the current mean estimates.

        Returns:
            np.ndarray: n x n matrix of mean estimates.
        """
        return self.alpha / (self.alpha + self.beta)

    def get_variances(self):
        """
        Retrieve the current variance estimates.

        Returns:
            np.ndarray: n x n matrix of variance estimates.
        """
        return (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )

    def reset(self):
        """
        Reset the Beta parameters to the initial state.
        """
        self.alpha.fill(1.0)
        self.beta.fill(1.0)

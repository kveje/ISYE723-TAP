# agents/learners/bayesian_learner.py

from ..beliefs.normal_belief import NormalBelief


class KalmanLearner:
    def __init__(self, env_config):
        """
        Initialize the Bayesian Learner with BetaBelief.

        Args:
            num_individuals (int): Total number of individuals.
        """
        num_individuals = env_config.num_individuals
        self.normal_belief = NormalBelief(num_individuals)

    def update(self, experience):
        """
        Update beliefs based on the latest experience.

        Args:
            experience (tuple): Contains (observation, action, reward, next_observation, done).
        """
        # Unpack the experience tuple
        feedback, _, _, _ = experience
        self.normal_belief.update(feedback)

    def get_means(self):
        """
        Retrieve the current mean estimates.

        Returns:
            np.ndarray: n x n matrix of mean estimates.
        """
        return self.normal_belief.get_means()

    def get_variances(self):
        """
        Retrieve the current variance estimates.

        Returns:
            np.ndarray: n x n matrix of variance estimates.
        """
        return self.normal_belief.get_variances()

    def reset(self):
        """
        Reset the learner's beliefs.
        """
        raise NotImplementedError("Reset method is not implemented yet.")

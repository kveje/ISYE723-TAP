# agents/learners/bayesian_learner.py

from beliefs.beta_belief import BetaBelief


class BayesianBetaLearner:
    def __init__(self, num_individuals):
        """
        Initialize the Bayesian Learner with BetaBelief.

        Args:
            num_individuals (int): Total number of individuals.
        """
        self.beta_belief = BetaBelief(num_individuals)

    def update(self, experience):
        """
        Update beliefs based on the latest experience.

        Args:
            experience (tuple): Contains (observation, action, reward, next_observation, done).
        """
        # Unpack the experience tuple
        _, action, reward, next_observation, done = experience
        feedback_matrix = next_observation["feedback"]  # Assuming feedback is a dict
        self.beta_belief.update(feedback_matrix)

    def get_means(self):
        """
        Retrieve the current mean estimates.

        Returns:
            np.ndarray: n x n matrix of mean estimates.
        """
        return self.beta_belief.get_means()
    
    def get_variances(self):
        """
        Retrieve the current variance estimates.

        Returns:
            np.ndarray: n x n matrix of variance estimates.
        """
        return self.beta_belief.get_variances()

    def reset(self):
        """
        Reset the learner's beliefs.
        """
        self.beta_belief.reset()

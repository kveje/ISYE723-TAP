# agents/learners/base_learner.py

from abc import ABC, abstractmethod
from envs.env_config import EnvConfig


class BaseLearner(ABC):
    def __init__(self, env_config: EnvConfig):
        """
        Initialize the Learner with the environment configuration.

        Args:
            env_config (EnvConfig): The environment configuration.
        """
        self.env_config = env_config
        self.num_individuals = env_config.num_individuals

    @abstractmethod
    def update(self, experience):
        """
        Update the learner based on new experience.

        Args:
            experience (tuple): A tuple containing (observation, action, reward, next_observation, done).
        """
        pass

    @abstractmethod
    def get_means(self):
        """
        Retrieve the current mean estimates.

        Returns:
            np.ndarray: n x n matrix of mean estimates.
        """
        pass

    @abstractmethod
    def get_variances(self):
        """
        Retrieve the current variance estimates.

        Returns:
            np.ndarray: n x n matrix of variance estimates.
        """
        pass

# agents/actors/base_actor.py

from abc import ABC, abstractmethod
from envs.env_config import EnvConfig


class BaseActor(ABC):
    def __init__(self, env_config: EnvConfig):
        """
        Initialize the Actor with the environment configuration.

        Args:
            env_config (EnvConfig): The environment configuration.
        """
        self.env_config = env_config

    @abstractmethod
    def act(self, learner: object):
        """
        Assign individuals to teams based on the actor's strategy.

        Args:
            beliefs (dict): Current beliefs about preferences.

        Returns:
            action: Team assignments (e.g., list mapping individuals to teams).
        """
        pass

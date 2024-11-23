from abc import ABC, abstractmethod
from envs.env_config import EnvConfig
from gym import spaces


class Agent(ABC):
    def __init__(self, action_space: spaces, env_config: EnvConfig):
        self.action_space = action_space
        self.env_config = env_config

    @abstractmethod
    def act(self, observation):
        pass

    def remember(self, observation, action, reward, next_observation, done):
        # Optional: Store experiences for learning agents
        pass

    def learn(self):
        # Optional: Implement learning mechanism
        pass

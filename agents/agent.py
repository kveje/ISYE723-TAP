# agents/agent.py

from abc import ABC, abstractmethod
from agents.actors.base_actor import BaseActor
from agents.learners.base_learner import BaseLearner
import matplotlib.pyplot as plt


class Agent(ABC):
    def __init__(self, actor: BaseActor, learner: BaseLearner):
        """
        Initialize the Agent with an Actor and a Learner.

        Args:
            actor (BaseActor): The actor component responsible for assignments.
            learner (BaseLearner): The learner component responsible for learning preferences.
        """
        self.actor = actor
        self.learner = learner

    def act(self):
        """
        Decide on an action based on the current observation.

        Args:
            individuals (list): List of Individual objects.
            beliefs (dict): Current beliefs about preferences.

        Returns:
            action: The action to take (e.g., team assignments).
        """
        action = self.actor.act(self.learner)
        # self.actions.append(action)
        return self.actor.act(self.learner)

    def learn(self, experience):
        """
        Update the learner with new experiences.

        Args:
            experience (tuple): A tuple containing (observation, action, reward, next_observation, done).
        """
        observation, _, reward, _ = experience
        self.learner.update(experience)

    def get_beliefs(self):
        return self.learner.get_means(), self.learner.get_variances()

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

        self.actions = []
        self.rewards = []
        self.beliefs = []
        self.feedback = []

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
        self.actions.append(action)
        return self.actor.act(self.learner)

    def learn(self, experience):
        """
        Update the learner with new experiences.

        Args:
            experience (tuple): A tuple containing (observation, action, reward, next_observation, done).
        """
        observation, _, reward, _ = experience
        self.learner.update(experience)
        # self.rewards.append(reward)
        # self.beliefs.append(
        #     {
        #         "means": self.learner.get_means(),
        #         "variances": self.learner.get_variances(),
        #     }
        # )
        # self.feedback.append(observation)

    def get_beliefs(self):
        return self.learner.get_means(), self.learner.get_variances()

    def visualize(self, person1=0, person2=1):
        """
        Visualize the agent's performance.
        """
        # Rewards over time
        plt.plot(self.rewards)
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.title("Agent Performance")
        plt.show()

        # Beliefs over time
        means = [belief["means"] for belief in self.beliefs]
        variances = [belief["variances"] for belief in self.beliefs]

        plt.plot([mean[person1, person2] for mean in means], label=f"Person {person1}")
        # Add the variance as a shaded region
        plt.fill_between(
            range(len(variances)),
            [
                mean[person1, person2] - var[person1, person2]
                for mean, var in zip(means, variances)
            ],
            [
                mean[person1, person2] + var[person1, person2]
                for mean, var in zip(means, variances)
            ],
            alpha=0.2,
        )
        # Plot the true preference
        plt.plot(
            [feedback[person1, person2] for feedback in self.feedback],
            label=f"Feedback",
        )
        # Add labels
        plt.xlabel("Time")
        plt.ylabel("Mean Preference")
        plt.title(f"Preference of Person {person1} over Person {person2}")
        plt.legend()
        plt.show()

import random
from .base_agent import Agent


class RandomAgent(Agent):
    def act(self, observation):
        # Randomly assign individuals to teams
        num_individuals = len(observation["previous_assignments"])
        num_teams = self.env_config.number_teams
        max_team_size = self.env_config.max_team_size

        # Randomly assign individuals to teams
        action = [team_id for team_id in range(num_teams) for _ in range(max_team_size)]
        random.shuffle(action)
        action = action[
            :num_individuals
        ]  # Trim the action to the number of individuals

        return action

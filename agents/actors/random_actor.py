import numpy as np
from .base_actor import BaseActor


class RandomActor(BaseActor):
    def act(self, beliefs):
        # Get parameters
        num_individuals = self.env_config.num_individuals
        num_teams = self.env_config.number_teams
        max_team_size = self.env_config.max_team_size

        # Generate team assignments
        action = np.repeat(np.arange(num_teams), max_team_size)

        # Shuffle the array
        np.random.shuffle(action)

        # Truncate to the number of individuals
        return action[:num_individuals]

import numpy as np
from .base_actor import BaseActor
from gurobipy import Model, GRB, quicksum


class ThompsonActor(BaseActor):
    def __init__(self, env_config):
        """
        Initialize the Thompson Sampling Actor with the environment configuration.

        Args:
            env_config (EnvConfig): The environment configuration.
            beta (float): The beta parameter for the softmax function.
        """
        super().__init__(env_config)

    def act(self, learner: object):
        # Get parameters
        num_individuals = self.env_config.num_individuals
        num_teams = self.env_config.number_teams
        max_team_size = self.env_config.max_team_size

        mean = learner.get_means()
        variance = learner.get_variances()
        # Thompson sampling
        score = (
            np.random.randn(num_individuals, num_individuals) * variance + mean
        ) / 2

        # Print the input parameters
        # print(f"Number of individuals: {num_individuals}")
        # print(f"Number of teams: {num_teams}")
        # print()
        # print(f"Scores: {score}")

        # Create a new model
        model = Model("team-formation")
        model.setParam("OutputFlag", 0)

        # Create variables
        x = {
            (i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            for i in range(num_individuals)
            for j in range(num_teams)
        }

        # Constraints
        for i in range(num_individuals):
            model.addConstr(quicksum(x[i, j] for j in range(num_teams)) == 1)

        for j in range(num_teams):
            model.addConstr(
                quicksum(x[i, j] for i in range(num_individuals)) <= max_team_size
            )

        # Objective
        model.setObjective(
            quicksum(
                x[i, k] * x[j, k] * score[i, j]
                for i in range(num_individuals)
                for j in range(num_individuals)
                for k in range(num_teams)
            ),
            GRB.MAXIMIZE,
        )

        # Solve
        model.optimize()

        # Get team assignments
        action = [
            j
            for i in range(num_individuals)
            for j in range(num_teams)
            if x[i, j].x > 0.5
        ]

        # Erase model
        del model

        return np.array(action)
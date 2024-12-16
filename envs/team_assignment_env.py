import gym
from gym import spaces
import numpy as np
from models.individuals import Individuals
from envs.env_config import EnvConfig


class TeamAssignmentEnv(gym.Env):
    """
    A gym-like environment for team assignment problems.
    """

    def __init__(self, env_config: EnvConfig):
        super(TeamAssignmentEnv, self).__init__()
        self.num_individuals = env_config.num_individuals
        self.teams = list(range(env_config.number_teams))
        self.max_team_size = env_config.max_team_size
        self.num_periods = env_config.num_periods
        self.random_substitution = env_config.random_substitution

        self.observation_space = (
            spaces.Box(
                low=-100,
                high=100,
                shape=(self.num_individuals, self.num_individuals),
                dtype=np.float32,
            ),
        )

        self.current_period = 0
        self.individuals = Individuals(
            self.num_individuals,
            env_config.sigma_w,
            env_config.sigma_f,
            env_config.sigma_p,
        )
        self.state = np.zeros((self.num_individuals, self.num_individuals))

    def reset(self):
        self.current_period = 0
        # Reset individuals' true preferences and belief states
        self.individuals.reset()

        # Feedback is not available at the beginning
        self.state = np.zeros((self.num_individuals, self.num_individuals))

        return self.state

    def step(self, action: np.ndarray):
        # Update period
        self.current_period += 1

        # TODO: Check if the action is valid

        # Evolve preferences
        self.individuals.update_preferences()

        team_interaction = self._get_team_interaction(action)

        # Get feedback
        feedback = self.individuals.collect_feedback(team_interaction)

        # Calculate reward
        reward = self.individuals.calculate_reward(team_interaction)

        # Check if the episode is done
        done = self.current_period >= self.num_periods

        # Random substitution
        random_probs = np.random.rand(self.num_individuals)
        reset = random_probs < self.random_substitution
        individuals_to_reset = np.where(reset)[0].tolist()
        for i in individuals_to_reset:
            self.individuals.reset_preferences(i)

        # Optional info dictionary # TODO: Add more info?
        info = {"reset": individuals_to_reset}

        return feedback, reward, done, info

    def render(self, mode="human"):
        # Optional: Implement visualization
        pass

    def _get_team_interaction(self, action: np.ndarray):
        team_interaction = action[:, None] == action
        np.fill_diagonal(team_interaction, False)
        return team_interaction

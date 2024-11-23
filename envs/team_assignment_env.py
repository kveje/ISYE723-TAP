import gym
from gym import spaces
import numpy as np
from models.individual import Individual
from models.team import Team
from utils.feedback import generate_feedback
from utils.preference_evolution import evolve_preference
from utils.metrics import calculate_overall_performance
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
        self.phi = env_config.phi
        self.sigma_eta = env_config.sigma_eta
        self.sigma_feedback = env_config.sigma_feedback

        # Action space: List of team numbers (a team number for each individual)
        self.action_space = spaces.Discrete(self.num_individuals)

        self.observation_space = spaces.Dict(
            {
                "feedback": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_individuals, self.num_individuals),
                    dtype=np.float32,
                ),
                "previous_assignments": spaces.Box(
                    shape=(self.num_individuals, self.num_individuals),
                    low=0,
                    high=self.num_periods,
                    dtype=np.int32,
                ),
            }
        )
        self.previous_assignments = np.zeros(
            shape=(self.num_individuals, self.num_individuals), dtype=int
        )
        self.current_period = 0
        self.individuals = {i: Individual(id=i) for i in range(self.num_individuals)}
        self.current_teams = None
        self.state = None
        self.reset()

    def reset(self):
        self.current_period = 0
        # Reset individuals' true preferences and belief states
        for ind in self.individuals.values():
            ind.reset_preferences()
        # Reset teams
        self.current_teams = None

        self.state = self._get_observation()
        return self.state

    def step(self, action: list[int]):
        # Action: A list where the index is the individual ID and the value is the team assignment
        self.current_period += 1

        # Apply the action: Assign individuals to teams
        self.current_teams = self._assign_teams(action)

        # Evolve preferences for the next period
        self._evolve_preferences()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if the episode is done
        done = self.current_period >= self.num_periods

        # Get next observation
        self.state = self._get_observation()

        # Optional info dictionary
        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        # Optional: Implement visualization
        pass

    def _get_observation(self):
        # Collect feedback
        if self.current_teams is None:
            feedback = np.zeros((self.num_individuals, self.num_individuals))
        else:
            feedback = self._collect_feedback(self.current_teams)

        # Construct the observation
        observation = {
            "feedback": feedback,
            "previous_assignments": self.previous_assignments,
        }
        return observation

    def _assign_teams(self, action: list[int]):
        # Convert action into team assignments
        team_assignments = {}
        for i in range(self.num_individuals):
            team_id = action[i]
            if team_id not in team_assignments:
                team_assignments[team_id] = []
            team_assignments[team_id].append(self.individuals[i])

        # Increment the previous assignments
        for team_id in team_assignments:
            for i in team_assignments[team_id]:
                for j in team_assignments[team_id]:
                    self.previous_assignments[i.id, j.id] += 1

        return [Team(members) for members in team_assignments.values()]

    def _collect_feedback(self, teams: list[Team]):
        feedback = np.zeros((self.num_individuals, self.num_individuals))
        for team in teams:
            temp = team.collect_feedback(self.sigma_feedback)
            for i in temp:
                for j in temp[i]:
                    feedback[i][j] = temp[i][j]
        return feedback

    def _evolve_preferences(self):
        for team in self.current_teams:
            team_lst = [ind.id for ind in team.members]
            for ind in team.members:
                ind.update_true_preferences(team_lst, self.phi, self.sigma_eta)

    def _calculate_reward(self):
        # The reward can be the total team performance
        total_performance = calculate_overall_performance(
            self.current_teams, metric="L1"
        )
        return total_performance

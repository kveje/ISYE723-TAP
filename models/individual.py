import random


class Individual:
    def __init__(self, id):
        self.id = id
        self.true_preferences = {}

    def get_preferences(self, id):
        if id in self.true_preferences:
            return self.true_preferences[id]
        else:
            raise ValueError(
                f"Individual {self.id} does not have preferences for individual {id}"
            )

    def reset_preferences(self, num_individuals=10):
        # Reset true preferences
        self.true_preferences = {
            i: random.random() if i != self.id else 1 for i in range(num_individuals)
        }

    def collect_feedback(self, j, sigma_feedback):
        # Collect feedback from individual j and clip it between 0 and 1
        return max(
            0, min(1, self.true_preferences[j] + random.gauss(0, sigma_feedback))
        )

    def update_true_preferences(
        self, team_members: list[int], phi: float, sigma_eta: float
    ):
        """
        Update true preferences over time

        Args:
            team_members (list[int]): List of team members
            phi (float): Evolution parameter
            sigma_eta (float): Noise parameter
        """
        # Update true preferences over time
        for j, w_ij in self.true_preferences.items():
            # Update preferences only for team members
            if j in team_members and j != self.id:
                eta = random.gauss(0, sigma_eta)
                self.true_preferences[j] = (
                    max(0, min(1, phi * w_ij + eta)) if j != self.id else 1
                )

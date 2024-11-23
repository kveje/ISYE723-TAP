from models.individual import Individual


# team.py
class Team:
    def __init__(self, members: list[Individual]):
        self.members = members

    def collect_feedback(self, sigma_feedback):
        # Collect feedback from team members
        feedback = {}
        for ind1 in self.members:
            feedback[ind1.id] = {}
            for ind2 in self.members:
                feedback[ind1.id][ind2.id] = ind1.collect_feedback(
                    ind2.id, sigma_feedback
                )
        return feedback

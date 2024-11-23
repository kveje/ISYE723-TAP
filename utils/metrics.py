from models.team import Team
from models.individual import Individual


def calculate_team_performance(team: Team, metric="L1"):
    if metric == "L1":
        # Sum of preferences among team members
        performance = sum(
            team_member.get_preferences(other_member.id)
            for team_member in team.members
            for other_member in team.members
            if team_member.id != other_member.id
        )
    elif metric == "L2":
        # Euclidean norm of preferences
        performance = sum(
            team_member.get_preferences(other_member.id) ** 2
            for team_member in team.members
            for other_member in team.members
            if team_member.id != other_member.id
        )
    elif metric == "Linf":
        # Minimum preference among team members
        performance = min(
            team_member.get_preferences(other_member.id)
            for team_member in team.members
            for other_member in team.members
            if team_member.id != other_member.id
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return performance


def calculate_overall_performance(teams: list[Team], metric="L1"):
    if metric in ["L1", "L2", "Linf"]:
        total_performance = sum(
            calculate_team_performance(team, metric) for team in teams
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return total_performance

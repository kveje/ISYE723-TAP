from dataclasses import dataclass
from math import ceil


@dataclass
class EnvConfig:
    num_individuals: int = 100
    number_teams: int = 10
    max_team_size: int = ceil(num_individuals / number_teams)
    num_periods: int = 1000
    phi: float = 0.5
    sigma_eta: float = 0.1
    sigma_feedback: float = 0.1
    metric: str = "L1"  # L1, L2 or Linf

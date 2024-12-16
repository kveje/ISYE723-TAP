PATH = "results_4"
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import estimate_reward, calculate_distance
from utils.visualization import aggregate, plot, cummulative_aggregate, create_color_map
import json

# Open simulation configuration
with open(f"{PATH}/config.json") as f:
    config = json.load(f)

ACTORS = config["ACTORS"]
NUM_INDIVIDUALS = config["NUM_INDIVIDUALS"]
NUM_TEAMS = config["NUM_TEAMS"]
MAX_TEAM_SIZE = config["MAX_TEAM_SIZE"]
SIGMA_F = config["SIGMA_F"]
SIGMA_W = config["SIGMA_W"]
SIGMA_P = config["SIGMA_P"]
NUM_SIMULATIONS = config["NUM_SIMULATIONS"]
NUM_PERIODS = config["NUM_PERIODS"]
RANDOM_SUBSTITUTION = config.get("RANDOM_SUBSTITUTION", 0)

# Initialize variables
actual_rewards = {
    actor_name: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for actor_name in ACTORS
}
estimated_rewards = {
    actor_name: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for actor_name in ACTORS
}
preference_distances = {
    actor_name: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for actor_name in ACTORS
}
times = {
    actor_name: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for actor_name in ACTORS
}

# Load results
for actor_name in ACTORS:
    # Load using numpy (PATH/actor_name/rewards.npy)
    actual_rewards[actor_name] = np.load(f"{PATH}/{actor_name}/actual_rewards.npy")
    estimated_rewards[actor_name] = np.load(
        f"{PATH}/{actor_name}/estimated_rewards.npy"
    )
    preference_distances[actor_name] = np.load(
        f"{PATH}/{actor_name}/preference_distances.npy"
    )
    times[actor_name] = np.load(f"{PATH}/{actor_name}/times.npy")

# Aggregate results
agg_actual_rewards = aggregate(actual_rewards)
agg_estimated_rewards = aggregate(estimated_rewards)
agg_distances = aggregate(preference_distances)
agg_times = aggregate(times)
cum_agg_actual_rewards = cummulative_aggregate(actual_rewards)
cum_agg_estimated_rewards = cummulative_aggregate(estimated_rewards)
cum_agg_times = cummulative_aggregate(times)

# Prepare labels
x_label = f"""Period
#Individuals: {NUM_INDIVIDUALS},  #Teams: {NUM_TEAMS},  $alpha=${MAX_TEAM_SIZE},  $\sigma_f=${SIGMA_F},  $\sigma_w=${SIGMA_W},  $\sigma_p=${SIGMA_P}, $\gamma=${RANDOM_SUBSTITUTION}"""
title = "of Different Actors"
color_map = create_color_map(ACTORS)

# Plot results
plot(
    agg_actual_rewards,
    x_label,
    "Reward",
    f"Reward {title}",
    f"{PATH}/rewards.png",
    color_map,
)
plot(
    agg_estimated_rewards,
    x_label,
    "Estimated Reward",
    f"Estimated Reward {title}",
    f"{PATH}/estimated_rewards.png",
    color_map,
)
plot(
    agg_distances,
    x_label,
    "Preference Distance",
    f"Preference Distance {title}",
    f"{PATH}/preference_distance.png",
    color_map,
)

# Plot cumulative results
plot(
    cum_agg_actual_rewards,
    x_label,
    "Cumulative Reward",
    f"Cumulative Reward {title}",
    f"{PATH}/cumulative_rewards.png",
    color_map,
)

plot(
    cum_agg_estimated_rewards,
    x_label,
    "Cumulative Estimated Reward",
    f"Cumulative Estimated {title}",
    f"{PATH}/cumulative_estimated_rewards.png",
    color_map,
)

# Plot time results
plot(
    agg_times,
    x_label,
    "Time (s)",
    f"Time {title}",
    f"{PATH}/times.png",
    color_map,
)

plot(
    cum_agg_times,
    x_label,
    "Cumulative Time (s)",
    f"Cumulative Time {title}",
    f"{PATH}/cumulative_times.png",
    color_map,
)

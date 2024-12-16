import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import estimate_reward, calculate_distance
from utils.visualization import aggregate, plot, cummulative_aggregate, create_color_map
import json
import os

ACTORS = ["UCB_0", "UCB_0.1", "UCB_0.5", "Thompson"]


for ACTOR in ACTORS:
    # Make directories
    os.makedirs(f"plots/{ACTOR}", exist_ok=True)
    PATHS = ["results_1", "results_2", "results_3", "results_4"]
    HYPER = [0, 0.01, 0.05, 0.1]
    NUM_PERIODS = 100
    NUM_SIMULATIONS = 10
    NUM_INDIVIDUALS = 10
    NUM_TEAMS = 4
    MAX_TEAM_SIZE = 3
    SIGMA_F = 0.1
    SIGMA_W = 0.1
    SIGMA_P = 1

    # Initialize variables
    actual_rewards = {h: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for h in HYPER}
    estimated_rewards = {
        h: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for h in HYPER
    }
    preference_distances = {
        h: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for h in HYPER
    }
    times = {h: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for h in HYPER}

    # Load results
    i = 0
    for path in PATHS:
        # Load using numpy (PATH/actor_name/rewards.npy)
        actual_rewards[HYPER[i]] = np.load(f"{path}/{ACTOR}/actual_rewards.npy")
        estimated_rewards[HYPER[i]] = np.load(f"{path}/{ACTOR}/estimated_rewards.npy")
        preference_distances[HYPER[i]] = np.load(
            f"{path}/{ACTOR}/preference_distances.npy"
        )
        times[HYPER[i]] = np.load(f"{path}/{ACTOR}/times.npy")
        i += 1

    # Aggregate results
    agg_actual_rewards = aggregate(actual_rewards)
    agg_estimated_rewards = aggregate(estimated_rewards)
    agg_distances = aggregate(preference_distances)
    agg_times = aggregate(times)
    cum_agg_actual_rewards = cummulative_aggregate(actual_rewards)
    cum_agg_estimated_rewards = cummulative_aggregate(estimated_rewards)
    cum_agg_times = cummulative_aggregate(times)

    x_label = f"""Period
    #Individuals: {NUM_INDIVIDUALS},  #Teams: {NUM_TEAMS},  $alpha=${MAX_TEAM_SIZE},  $\sigma_f=${SIGMA_F},  $\sigma_w=${SIGMA_W},  $\sigma_p=${SIGMA_P}"""

    title = f"of {ACTOR} actor"
    color_map = create_color_map(HYPER)

    # Plot
    plot(
        agg_actual_rewards,
        xlabel=x_label,
        ylabel="Reward",
        title=f"Reward {title}",
        path=f"plots/{ACTOR}_rewards.png",
        color_map=color_map,
    )
    plot(
        agg_estimated_rewards,
        xlabel=x_label,
        ylabel="Estimated Reward",
        title=f"Estimated Reward {title}",
        path=f"plots/{ACTOR}_estimated_rewards.png",
        color_map=color_map,
    )
    plot(
        agg_distances,
        xlabel=x_label,
        ylabel="Distance",
        title=f"Preference Distance {title}",
        path=f"plots/{ACTOR}_preference_distances.png",
        color_map=color_map,
    )
    plot(
        agg_times,
        xlabel=x_label,
        ylabel="Time",
        title=f"Time {title}",
        path=f"plots/{ACTOR}_times.png",
        color_map=color_map,
    )
    plot(
        cum_agg_actual_rewards,
        xlabel=x_label,
        ylabel="Cumulative Reward",
        title=f"Cumulative Reward {title}",
        path=f"plots/{ACTOR}_cumulative_rewards.png",
        color_map=color_map,
    )

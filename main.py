from envs.team_assignment_env import TeamAssignmentEnv
from agents.actors.random_actor import RandomActor
from agents.actors.ucb_actor import UCBActor
from agents.actors.thompson_actor import ThompsonActor
from agents.agent import Agent
from agents.learners.kalman_learner import KalmanLearner
from envs.env_config import EnvConfig
from utils.metrics import estimate_reward, calculate_distance
from utils.visualization import aggregate, plot
import numpy as np
import os
import json
import time


def test(env_config: EnvConfig, actor: object, learner: object):
    # Create the environment
    env = TeamAssignmentEnv(env_config)

    # Create the agent
    agent = Agent(actor=actor, learner=learner)

    # Initialize variables
    total_reward = 0
    done = False
    rewards = np.zeros(env_config.num_periods)
    estimated_rewards = np.zeros(env_config.num_periods)
    period = 0
    preference_distances = np.zeros(env_config.num_periods)
    times = np.zeros(env_config.num_periods)

    # Reset the environment
    observation = env.reset()

    while not done:
        start = time.time()
        # Decide on an action
        action = agent.act()

        # Take a step in the environment
        observation, reward, done, info = env.step(action)

        # Learn from the experience
        agent.learn((observation, action, reward, done, info))

        # Update the total reward
        total_reward += reward
        rewards[period] = reward

        # Estimate the reward
        estimated_rewards[period] = estimate_reward(agent.get_beliefs()[0], action)

        # Calculate the distance between true preferences and estimated preferences
        preference_distances[period] = calculate_distance(
            env.individuals.get_true_preferences(),
            agent.learner.get_means(),
        )
        end = time.time()
        times[period] = end - start
        # Update period
        period += 1

        if period % 20 == 0:
            print(f"  Period {period} / {env_config.num_periods}")

    return rewards, estimated_rewards, preference_distances, times


if __name__ == "__main__":
    NUM_SIMULATIONS = 10
    ACTORS = [
        "UCB_0",
        "UCB_0.1",
        "UCB_0.5",
        "Random",
        "Thompson",
    ]
    LEARNERS = ["Kalman"]
    NUM_INDIVIDUALS = 10
    NUM_TEAMS = 4
    MAX_TEAM_SIZE = 3
    NUM_PERIODS = 100
    SIGMA_F = 0.1
    SIGMA_W = 0.1
    SIGMA_P = 1
    RANDOM_SUBSTITUTION = 0.1

    # Create folders for storing results
    if not os.path.exists("results"):
        os.makedirs("results")

    for actor_type in ACTORS:
        if not os.path.exists(f"results/{actor_type}"):
            os.makedirs(f"results/{actor_type}")

    # Save Simulation Config
    config = {
        "NUM_SIMULATIONS": NUM_SIMULATIONS,
        "ACTORS": ACTORS,
        "LEARNERS": LEARNERS,
        "NUM_INDIVIDUALS": NUM_INDIVIDUALS,
        "NUM_TEAMS": NUM_TEAMS,
        "MAX_TEAM_SIZE": MAX_TEAM_SIZE,
        "NUM_PERIODS": NUM_PERIODS,
        "SIGMA_F": SIGMA_F,
        "SIGMA_W": SIGMA_W,
        "SIGMA_P": SIGMA_P,
        "RANDOM_SUBSTITUTION": RANDOM_SUBSTITUTION,
    }

    # Save the config as json
    json.dump(config, open("results/config.json", "w"))

    # Simulations
    for actor_type in ACTORS:
        actual_rewards = np.zeros(
            shape=(NUM_SIMULATIONS, NUM_PERIODS), dtype=np.float32
        )
        estimated_rewards = np.zeros(
            shape=(NUM_SIMULATIONS, NUM_PERIODS), dtype=np.float32
        )
        preference_distances = np.zeros(
            shape=(NUM_SIMULATIONS, NUM_PERIODS), dtype=np.float32
        )
        times = np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS), dtype=np.float32)
        print(f"Starting simlatuion with actor: {actor_type} and learner: Kalman")
        for i in range(NUM_SIMULATIONS):
            # Config
            env_config = EnvConfig(
                num_individuals=NUM_INDIVIDUALS,
                number_teams=NUM_TEAMS,
                max_team_size=MAX_TEAM_SIZE,
                num_periods=NUM_PERIODS,
                sigma_f=SIGMA_F,
                sigma_w=SIGMA_W,
                sigma_p=SIGMA_P,
                random_substitution=RANDOM_SUBSTITUTION,
            )

            # Create the agent
            if actor_type.lower()[0:3] == "ucb":
                beta = float(actor_type.split("_")[1])
                actor = UCBActor(env_config=env_config, beta=beta)
            elif actor_type.lower() == "random":
                actor = RandomActor(env_config=env_config)
            elif actor_type.lower() == "thompson":
                actor = ThompsonActor(env_config=env_config)
            else:
                raise ValueError("Invalid actor type")

            learner = KalmanLearner(env_config=env_config)

            # Test the agent
            act_rewards, est_rewards, pref_dist, t = test(env_config, actor, learner)

            # Store the results
            actual_rewards[i, :] = act_rewards
            estimated_rewards[i, :] = est_rewards
            preference_distances[i, :] = pref_dist
            times[i, :] = t

            print(
                f" Agent: {actor_type}, Simulation {i+1} / {NUM_SIMULATIONS} completed"
            )

        # Save the results
        np.save(f"results/{actor_type}/actual_rewards.npy", actual_rewards)
        np.save(f"results/{actor_type}/estimated_rewards.npy", estimated_rewards)
        np.save(f"results/{actor_type}/preference_distances.npy", preference_distances)
        np.save(f"results/{actor_type}/times.npy", times)

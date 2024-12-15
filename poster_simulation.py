from envs.team_assignment_env import TeamAssignmentEnv
from agents.actors.random_actor import RandomActor
from agents.actors.ucb_actor import UCBActor
from agents.agent import Agent
from agents.learners.kalman_learner import KalmanLearner
from envs.env_config import EnvConfig
import matplotlib.pyplot as plt
from statistics import mean, stdev
from utils.metrics import estimate_reward, calculate_distance
from utils.visualization import aggregate, plot
import numpy as np


def test(env_config: EnvConfig, actor: object, learner: object):
    # Create the environment
    env = TeamAssignmentEnv(env_config)

    # Create the agent
    agent = Agent(actor=actor, learner=learner)

    # Initialize variables
    total_reward = 0
    done = False
    rewards = []
    estimated_rewards = []
    period = 0
    preference_distances = []

    # Reset the environment
    observation = env.reset()

    random_person_1 = np.random.randint(0, env_config.num_individuals)
    random_person_2 = np.random.randint(0, env_config.num_individuals)
    while random_person_1 == random_person_2:
        random_person_2 = np.random.randint(0, env_config.num_individuals)

    true_preferences = []
    estimated_preferences = []
    estimated_preferences_variances = []

    while not done:
        # Decide on an action
        action = agent.act()

        # Take a step in the environment
        observation, reward, done, _ = env.step(action)

        # Learn from the experience
        agent.learn((observation, action, reward, done))

        # Update the total reward
        total_reward += reward
        rewards.append(reward)

        # Estimate the reward
        estimated_rewards.append(estimate_reward(agent.get_beliefs()[0], action))

        # Calculate the distance between true preferences and estimated preferences
        preference_distances.append(
            calculate_distance(
                env.individuals.get_true_preferences(),
                agent.learner.get_means(),
            )
        )

        # Update period
        period += 1

        # Save preferences for visualization
        true_preferences.append(
            env.individuals.get_true_preferences()[random_person_1, random_person_2]
        )
        estimated_preferences.append(
            agent.learner.get_means()[random_person_1, random_person_2]
        )
        estimated_preferences_variances.append(
            agent.learner.get_variances()[random_person_1, random_person_2]
        )

        if period % 20 == 0:
            print(f"  Period {period} / {env_config.num_periods}")

    return (
        agent,
        np.array(rewards),
        np.array(estimated_rewards),
        np.array(preference_distances),
        np.array(true_preferences) - np.array(estimated_preferences),
    )


if __name__ == "__main__":
    NUM_SIMULATIONS = 10
    BETAS = [0, 0.1, 0.5, 1, 2, 5]
    NUM_INDIVIDUALS = 10
    NUM_TEAMS = 3
    MAX_TEAM_SIZE = 4
    NUM_PERIODS = 100
    SIGMA_F = 0.1
    SIGMA_W = 0.1
    SIGMA_P = 1

    # Initialize dictionaries to store rewards
    rewards = {beta: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for beta in BETAS}
    estimated_rewards = {
        beta: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for beta in BETAS
    }
    preference_distances = {
        beta: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for beta in BETAS
    }
    prefs = {beta: np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS)) for beta in BETAS}

    # Beta simulation
    for beta in BETAS:
        print(f"Starting beta: {beta}")
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
            )

            # Create the agent
            actor = UCBActor(env_config=env_config, beta=beta)
            learner = KalmanLearner(env_config=env_config)

            # Test the agent
            agent, act_rewards, est_rewards, pref_dist, pref = test(
                env_config, actor, learner
            )

            # Store the results
            rewards[beta][i, :] = act_rewards
            estimated_rewards[beta][i, :] = est_rewards
            preference_distances[beta][i, :] = pref_dist
            prefs[beta][i, :] = pref

            print(f" Beta: {beta}, Simulation {i+1} / {NUM_SIMULATIONS} completed")

    # Random simulation
    # Initialize dictionaries to store rewards
    rewards["Random"] = np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS))
    estimated_rewards["Random"] = np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS))
    preference_distances["Random"] = np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS))
    prefs["Random"] = np.zeros(shape=(NUM_SIMULATIONS, NUM_PERIODS))

    print(f"Starting Random")
    # Run the simulation
    for _ in range(NUM_SIMULATIONS):
        # Config
        env_config = EnvConfig(
            num_individuals=NUM_INDIVIDUALS,
            number_teams=NUM_TEAMS,
            max_team_size=MAX_TEAM_SIZE,
            num_periods=NUM_PERIODS,
        )

        # Create the agent
        actor = RandomActor(env_config=env_config)
        learner = KalmanLearner(env_config=env_config)

        # Test the agent
        agent, act_rewards, est_rewards, pref_dist, pref = test(
            env_config, actor, learner
        )

        # Store the results
        rewards["Random"][i, :] = act_rewards
        estimated_rewards["Random"][i, :] = est_rewards
        preference_distances["Random"][i, :] = pref_dist
        prefs["Random"][i, :] = pref

        print(f" Random, Simulation {i+1} / {NUM_SIMULATIONS} completed")

    # Aggregate results
    mean_rewards = aggregate(rewards)
    mean_estimated_rewards = aggregate(estimated_rewards)
    mean_distances = aggregate(preference_distances)
    mean_prefs = aggregate(prefs)

    x_label = f"""Period
               No. of Individuals: {NUM_INDIVIDUALS},   No. of Teams: {NUM_TEAMS},   Max Team Size: {MAX_TEAM_SIZE},   $\sigma_f=${SIGMA_F},   $\sigma_w=${SIGMA_W},   $\sigma_p=${SIGMA_P}"""
    title = "Performance of Different Actors"

    # Plot results
    plot(mean_rewards, x_label, "Reward", title, "results/rewards.png")
    plot(
        mean_estimated_rewards,
        x_label,
        "Estimated Reward",
        title,
        "results/estimated_rewards.png",
    )
    plot(
        mean_distances,
        x_label,
        "Preference Distance",
        title,
        "results/preference_distance.png",
    )
    plot(
        mean_prefs,
        x_label,
        "Preference Difference (two random individuals)",
        title,
        "results/preference_difference.png",
    )

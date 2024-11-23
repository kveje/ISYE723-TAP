from envs.team_assignment_env import TeamAssignmentEnv
from agents.random_agent import RandomAgent
from envs.env_config import EnvConfig

# from agents.heuristic_agent import HeuristicAgent
# from agents.rl_agent import RLAgent


def main():
    # Config
    env_config = EnvConfig(
        num_individuals=10, number_teams=2, max_team_size=5, num_periods=100
    )
    env = TeamAssignmentEnv(env_config)

    # Create the agent
    agent = RandomAgent(action_space=env.action_space, env_config=env_config)

    # Run the simulation
    observation = env.reset()
    total_reward = 0
    done = False

    action = agent.act(observation)
    while not done:
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        action = agent.act(observation)

    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    main()

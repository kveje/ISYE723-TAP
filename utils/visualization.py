import matplotlib.pyplot as plt


def plot_performance_over_time(performance_history):
    plt.figure(figsize=(10, 6))
    plt.plot(performance_history, marker="o")
    plt.title("Team Performance Over Time")
    plt.xlabel("Period")
    plt.ylabel("Total Performance")
    plt.grid(True)
    plt.show()


def plot_individual_preferences(individuals):
    # Visualization code for individual preferences
    pass

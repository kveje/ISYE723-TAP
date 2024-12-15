import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.misc import show_name


def create_color_map(actors: list):
    """
    Create a color map for the actors.

    Args:
        actors (list): List of actor names.

    Returns:
        dict: Dictionary containing the color map.
    """
    palette = sns.color_palette("Set1", n_colors=len(actors))
    color_map = {actor: palette[i] for i, actor in enumerate(actors)}

    return color_map


def aggregate(data: dict):
    """
    Aggregate the data for visualization.

    Args:
        data (dict): Dictionary containing the data to aggregate.

    Returns:
        dict: Dictionary containing the aggregated data.
    """
    aggregated_data = {}
    for key, values in data.items():
        aggregated_data[key] = {
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0),
        }
    return aggregated_data


def cummulative_aggregate(data: dict):
    """
    Aggregate the data for visualization.

    Args:
        data (dict): Dictionary containing the data to aggregate.

    Returns:
        dict: Dictionary containing the aggregated data.
    """
    aggregated_data = {}
    for key, values in data.items():
        aggregated_data[key] = {
            "mean": np.cumsum(np.mean(values, axis=0)),
            "std": np.cumsum(np.std(values, axis=0)),
        }
    return aggregated_data


def plot(
    aggregated_data: dict,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    path: str = None,
    color_map: dict = None,
):
    """
    Plot the aggregated data.

    Args:
        aggregated_data (dict): Dictionary containing the aggregated data on the form {agent_name: {"mean": [], "std": []}}.
    """
    plt.figure(figsize=(7, 4), constrained_layout=True)
    plt.style.use("ggplot")
    for key, values in aggregated_data.items():
        plt.plot(
            values["mean"], label=show_name(key), color=color_map[key], linewidth=2
        )
        plt.fill_between(
            range(len(values["mean"])),
            values["mean"] - values["std"],
            values["mean"] + values["std"],
            alpha=0.2,
            color=color_map[key],
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if path:
        plt.savefig(path)
        plt.savefig(path[:-3] + "pdf", format="pdf", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

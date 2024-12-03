import matplotlib.pyplot as plt
import numpy as np


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


def plot(
    aggregated_data: dict,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    path: str = None,
):
    """
    Plot the aggregated data.

    Args:
        aggregated_data (dict): Dictionary containing the aggregated data.
    """
    plt.figure(figsize=(8, 4))
    for key, values in aggregated_data.items():
        plt.plot(values["mean"], label=key)
        plt.fill_between(
            range(len(values["mean"])),
            values["mean"] - values["std"],
            values["mean"] + values["std"],
            alpha=0.2,
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

import matplotlib.pyplot as plt
import numpy as np


def plot_costs(costs_obj, trials, title=""):
    total_costs = []
    for t in range(trials):
        costs = np.array(costs_obj.get_separate_costs(trial=t))
        total_costs.append(costs)

    total_costs = np.stack(total_costs)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    for i in range(costs_obj.feature_vector_size):
        cs = total_costs[:, i]
        ax.plot(range(trials), cs, label=f"Feature {i}")
        ax.set_title(title)
        ax.legend()

    plt.close()

    return fig

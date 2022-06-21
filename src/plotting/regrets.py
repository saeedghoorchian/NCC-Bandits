import matplotlib.pyplot as plt


def plot_regrets(regret_dict, reward_change_points=(), cost_change_points=(), all_change_point_values=()):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8));

    ind_to_params = {
        # ind: (mark, color)
        0: ('x', 'blue'),
        1: ('>', 'g'),
        2: ('d', 'm'),
        3: ('<', 'goldenrod'),
        4: ('s', 'r'),
        5: ('o', 'dodgerblue'),
        6: ('^', 'k'),
    }

    sorted_regret_dict = {
        k: v for k, v in sorted(regret_dict.items(), key=lambda x: x[1][-1], reverse=True)
    }

    # Find out largest value. Do this first so change points appear first in the legend.
    max_vline = 0
    for regret in sorted_regret_dict.values():
        max_vline = max(max_vline, max(regret))

    plt.vlines(
        reward_change_points, ymin=0, ymax=max_vline, linestyle=':', alpha=0.4, label='Change Points'
    )

    for ind, (label, regret) in enumerate(sorted_regret_dict.items()):

        if label == 'SimOOS':
            label_new = 'Sim-OOS'
        else:
            label_new = label

        mark, color = ind_to_params[ind % 7]
        line, = ax.plot(regret, label=label_new, linestyle=':',
                        marker=mark, markevery=750, markersize=11, markeredgewidth=1.5,
                        fillstyle='none', color=color,
                        linewidth=3,
                        )

    extraticks = all_change_point_values
    plt.xticks(extraticks, rotation=35, fontsize=25)
    plt.yticks(fontsize=25)

    ax.set_xlabel("Time Step", fontsize=38)
    ax.set_ylabel('Regret', fontsize=38)

    plt.legend(prop={'size': 22.5}, ncol=1);

    plt.close()

    return fig
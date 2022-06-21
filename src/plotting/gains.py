import matplotlib.pyplot as plt


def plot_gains(gain_dict, reward_change_points=(), cost_change_points=(), all_change_point_values=()):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8));

    ind_to_params = {
        # ind: (mark, color)
        0: ('x', 'k'),
        1: ('o', 'g'),
        2: ('s', 'r'),
        3: ('^', 'blue'),
        4: ('d', 'olive'),
        5: ('>', 'm'),
        6: ('*', 'c'),
        7: ('<', 'y'),
    }

    sorted_gain_dict = {
        k: v for k, v in sorted(gain_dict.items(), key=lambda x: x[1][-1], reverse=True)
    }

    max_vline = 0

    for ind, (label, gain) in enumerate(sorted_gain_dict.items()):
        mark, color = ind_to_params[ind % 8]
        ax.plot(gain, label=label, linestyle=':',
                marker=mark, markevery=750, markersize=10,
                fillstyle='none', color=color,
                linewidth=3,
                )
        max_vline = max(max_vline, max(gain))

    plt.vlines(
        reward_change_points, ymin=0, ymax=max_vline, linestyle=':', alpha=0.4, label='Change Points'
    )

    extraticks = all_change_point_values
    plt.xticks(extraticks, rotation=35, fontsize=20)
    plt.yticks(fontsize=20)

    ax.set_xlabel('Time Step', fontsize=26)
    ax.set_ylabel('Cumulative Gain', fontsize=26)

    plt.legend(prop={'size': 20});
    plt.close()

    return fig
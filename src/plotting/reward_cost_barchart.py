import matplotlib.pyplot as plt
import numpy as np


def plot_reward_cost_barchart(labels, final_rewards, final_costs, final_gains):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    colors = plt.cm.Dark2(np.linspace(0, 1, 8))
    ax.set_prop_cycle('color', colors[:2][::-1])

    # Make stacked bar charts
    rects_cost = ax.bar(labels, final_costs, bottom=final_gains,
                        label='Total Paid Cost',
                        )

    rects_gain = ax.bar(labels, final_gains,
                        label='Total Gain'
                        )

    # Write values for gains and rewards
    for i, rect in enumerate(rects_gain):
        gain_height = 2000  # rect.get_height()

        #     if labels[i] == 'Oracle':
        #         gain_height = rects_gain[i+1].get_height()

        ax.text(rect.get_x() + rect.get_width() / 2.0, gain_height - 670,
                '%.1d' % final_gains[i],
                ha='center', va='bottom',
                fontsize=22,
                color='w',
                fontweight='bold',
                )

        reward_height = rect.get_height() + rects_cost[i].get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, reward_height + 100,
                '%.1d' % final_rewards[i],
                ha='center', va='bottom',
                fontsize=22,
                fontweight='bold',
                )

    # Write values for costs
    for i, rect in enumerate(rects_cost):

        if final_costs[i] == 0:
            continue

        if labels[i] == 'Oracle':
            continue

        height = rect.get_height() + rects_gain[i].get_height()

        ax.text(rect.get_x() + rect.get_width() / 2.0, height - 685,
                '%.1d' % final_costs[i],
                ha='center', va='bottom',
                fontsize=22,
                color='w',
                fontweight='bold',
                )

    # Write value for oracle cost that did not fit inside rectangle
    oracle_index = labels.index('Oracle')
    oracle_gain_height = rects_gain[oracle_index].get_height() - 700
    oracle_gain_loc = rects_gain[oracle_index].get_x() + rects_gain[oracle_index].get_width() / 2.0 - 0.2
    oracle_cost = final_costs[oracle_index]

    xytext = (oracle_gain_loc, oracle_gain_height)
    xy = (xytext[0] + 0.215, xytext[1] + 1020)

    ax.annotate(
        text=f"{int(oracle_cost)}",
        xy=xy,
        xytext=xytext,
        arrowprops=dict(arrowstyle="simple", color='w', linewidth=0.5),
        fontsize=22, fontweight='bold',
        color='w',
    )

    plt.xticks(rotation=20, fontsize=25)
    extraticks = [10000]
    plt.yticks(list(ax.get_yticks()) + extraticks, fontsize=25)

    ax.set_xlabel('Policy', fontsize=38)
    ax.set_ylabel('Total Reward', fontsize=38)

    plt.legend(prop={'size': 28})

    plt.tight_layout()

    plt.close()

    return fig
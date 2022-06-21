import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_accuracy(accuracies):
    fig, ax = plt.subplots(1, figsize=(10, 5))
    ind_to_params = {
        # ind: (mark, color)
        0: ('d' ,'red'), 1 :('s', 'darkblue') , 2 :('o' ,'deepskyblue') , 3: ('>' ,'orange'),
        4: ('^', 'm'), 5: ('<', 'k'), 6: ('x' ,'green'),
    }

    for ind, (algorithm_name, accuracy_for_alg) in enumerate(accuracies.items()):

        mark, color = ind_to_params[ind]
        ax.plot(
            range(len(accuracy_for_alg)),
            accuracy_for_alg,
            label=algorithm_name,
            marker=mark, markersize=8, fillstyle='none',
            c=color, linestyle=':', linewidth=3,
        )

    ax.set_xlabel('Number Of Observations', fontsize=26)
    ax.set_ylabel('Accuracy', fontsize=26)
    ax.set_ylim(0.0, 1.0)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Only integer labels

    plt.legend(prop={'size': 24}, loc='lower right')

    plt.close()

    return fig
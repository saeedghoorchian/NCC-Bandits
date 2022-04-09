import numpy as np


def discretize(array, num_bins):
    x_min = min(array)
    x_max = max(array)
    bins = np.linspace(x_min, x_max, num_bins+1)
    new_array = array.copy()
    for i in range(len(array)):
        x = array[i]
        if x >= x_max:
            new_x = x_max
        elif x <= x_min:
            new_x = x_min
        else:
            new_x = bins[np.digitize(x, bins)]
        new_array[i] = new_x
    return new_array

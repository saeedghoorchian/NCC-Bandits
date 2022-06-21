import numpy as np


def exchange_rows(array: np.array, rows1: list, rows2: list):
    """Exchange specified rows in the given two-dimensional array.

     Parameters
    ----------
    array:
        numpy array to exchange rows in. Not mutated.
    rows1, rows2:
        lists of row indexes of rows to exchange, must have the same length and no intersection in indexes.

     Returns:
    ----------
    result_array:
        copy of the original array with exchanged rows
    """
    result_array = array.copy()
    assert len(rows1) == len(rows2)
    assert len(set(rows1)) == len(rows1)
    assert len(set(rows2)) == len(rows2)
    assert set(rows1).intersection(set(rows2)) == set()
    temp1 = array[rows1].copy()
    temp2 = array[rows2].copy()

    result_array[rows1] = temp2
    result_array[rows2] = temp1
    return result_array


def rebalance_arms_between_regions(
        contexts: np.array,
        rewards: np.array,
        stationarity_regions: list,
        arm_1: int,
        arm_2: int,
        region_1: int,
        region_2: int,
        percentage: float,
):
    """Exchange rows in both context and reward arrays in such a way that rows
    for arm_1 in region_1 are exchanged with arm_2 in region_2. Only do this for a percentage of rows.

     Parameters
    ----------
    array:
        numpy array to exchange rows in. Not mutated.
    rows1, rows2:
        lists of row indexes of rows to exchange, must have the same length and no intersection in indexes.

     Returns:
    ----------
    result_array:
        copy of the original array with exchanged rows
    """
    assert arm_1 != arm_2
    assert region_1 != region_2

    region_1_slice = stationarity_regions[region_1]
    reg_1_mask = np.zeros(contexts.shape[0], dtype=bool)
    reg_1_mask[region_1_slice] = True

    region_2_slice = stationarity_regions[region_2]
    reg_2_mask = np.zeros(contexts.shape[0], dtype=bool)
    reg_2_mask[region_2_slice] = True

    arm_1_mask = np.array(rewards[:, arm_1], dtype=bool)
    arm_2_mask = np.array(rewards[:, arm_2], dtype=bool)

    mask1 = arm_1_mask & reg_1_mask
    mask2 = arm_2_mask & reg_2_mask

    rows1 = np.where(mask1)[0]
    rows2 = np.where(mask2)[0]

    num_rows_to_exchange = int(
        percentage * min(len(rows1), len(rows2))
    )

    rows1 = list(
        np.random.choice(
            rows1,
            replace=False,
            size=num_rows_to_exchange,
        )
    )
    rows2 = list(
        np.random.choice(
            rows2,
            replace=False,
            size=num_rows_to_exchange,
        )
    )

    if len(rows1) > len(rows2):
        rows1 = rows1[:len(rows2)]
    else:
        rows2 = rows2[:len(rows1)]

    contexts = exchange_rows(contexts, rows1, rows2)
    rewards = exchange_rows(rewards, rows1, rows2)

    return contexts, rewards
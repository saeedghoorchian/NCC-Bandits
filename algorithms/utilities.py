# This file contains utility functions for SimOOS and similar algorithms. Most of them have to do with
# states (partial vectors) and observations (feature subsets).
from collections import defaultdict
import itertools
import numpy as np


def full_perm_construct(size: int) -> np.array:
    """Constructs all possible observation actions up to length 'size'.

    Observation actions are just binary vectors. For size = 3 the result is:
    array([[0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        ...
        [1., 1., 0.],
        [1., 1., 1.]])
    """
    all_perms = np.zeros((2 ** size, size))

    for i in range(2 ** size):
        bin_str = np.binary_repr(i, width=size)
        bin_arr = np.fromstring(bin_str, 'u1') - ord('0')
        all_perms[i, :] = bin_arr

    return all_perms


def perm_construct(org_dim_context: int, max_no_red_context: int) -> np.array:
    """Computes all possible observation actions(permutations) for org_dim_context sources(features).

     max_no_red_context is the maximum number of sources that can be selected.
    """

    # If all permutations are needed - a quicker procedure is available.
    if org_dim_context == max_no_red_context:
        return full_perm_construct(org_dim_context)

    for i in range(0, max_no_red_context + 1):

        temp1 = np.array([1 for j in range(i)])

        temp2 = np.array([0 for j in range(org_dim_context - i)])

        temp_together = np.concatenate((temp1, temp2))

        temp = list(itertools.permutations(temp_together))

        p = np.unique(temp, axis=0)

        if i == 0:
            all_perms = p
        else:
            all_perms = np.concatenate((all_perms, p), axis=0)

    return all_perms


def save_feature_values(all_contexts) -> tuple:
    """Save unique values for each feature and their count.

    This is used in SimOOS and similar algorithms to enumerate all possible states and get index of state from
    observed context (state = partial vector in paper).
    """
    org_dim_context = all_contexts.shape[1]
    feature_values = defaultdict(list)
    feature_counts = np.zeros(org_dim_context)

    for i in range(org_dim_context):
        unique_features = np.unique(all_contexts[:, i])
        # None represents not observed feature
        values = [None] + sorted(list(unique_features))
        feature_values[i] = values

        feature_counts[i] = len(values)

    return feature_values, feature_counts


def state_construct(all_feature_counts: np.array, all_contexts: np.array, one_perm: np.array) -> tuple:
    """Count the number of partial state vectors (psi[i]) and size of state array (s_o[i]) for given observation.

    all_feature_counts: number of unique values for each feature
    all_contexts : context matrix
    one_perm : observation action
    all_feature_counts[i] : number of possible i-type context values
    """
    org_dim_context = all_contexts.shape[1]
    number_of_observation_action = np.dot(one_perm,
                                          np.ones(org_dim_context))  # How many features observed in one_perm.

    s_o = 1
    psi = 1

    # TODO Why is .item(0) here if its just a scalar?
    if number_of_observation_action.item(0) > 0:
        for i in range(org_dim_context):
            # For each observed feature multiply psi by the number of its values.
            # So psi stores how many partial vectors with support given by one_perm there can be.
            # all_feature_counts includes None, but partial vectors can't have None as values
            psi = psi * (all_feature_counts[i] - 1) ** (one_perm[i])

            # s_o is the size of state_array for a given observation. This array is larger, as it considers
            # None as a value for each feature.
            s_o = s_o * (all_feature_counts[i]) ** (one_perm[i])

    psi = int(psi)
    s_o = int(s_o)

    return psi, s_o


def state_extract(feature_values, all_feature_counts, context_at_t, observation_action_at_t) -> int:
    """Return the state number by context and observation.

    Let psi_i be the set of values of feature i (stored in feature_values[i]). psi_i includes None
    to incorporate the possibility of not observed feature.
    Then let Psi_total denote the cartesian product of psi_i for i in 1 to num_features.
    There are |Psi_total| possible partial vectors in total. Each partial context vector then corresponds
    to an index from 1 to |Psi_total|. This function returns this index.
    |Psi_total| = product(all_feature_counts[i] for i = 1 to num_features)
    The idea for this algorithm comes from positional number systems.

    context_at_t contains None values at those indices where observation_action_at_t is 0 - these are
    not observed features.
    """
    org_dim_context = context_at_t.shape[0]
    for feature, observation in zip(context_at_t, observation_action_at_t):
        # Sanity check
        if observation == 1:
            assert feature is not None
        else:
            assert feature is None

    # index of given context out of all possible partial vectors
    state_index = 0
    for i in range(org_dim_context):
        state_index *= all_feature_counts[i]
        # index of feature value of all possible values for this feature
        feature_index = feature_values[i].index(context_at_t[i])
        state_index += feature_index

    return int(state_index)


def state_create(state_index, feature_values):
    """Create partial vector from state_index.

    Not used in algorithms, only here for testing purposes.
    """
    org_dim_context = len(feature_values)

    context = np.zeros(org_dim_context)
    for i in reversed(range(org_dim_context)):
        feature_count = len(feature_values[i])
        feature_index = state_index % feature_count
        context[i] = feature_values[i][feature_index]
        state_index -= feature_index
        state_index = int(state_index / feature_count)

    return np.array([c if not np.isnan(c) else None for c in context])


def is_round_over(N_old, N):
    """Checks whether the round is ended.

    flag = 1 - round is over, flag = 0 - round is not over.
    """
    # Old, unvectorized version
    # flag = 0
    #
    # for j in range(N_old.shape[2]):  # if for a perm j=o
    #     for i in range(N_old.shape[1]):  # if for a state s=i
    #         for k in range(N_old.shape[0]):  # if for an action a=k
    #
    #             if (int(N[k, i, j].item(0)) <= N_old[k, i, j]) and (int(N_old[k, i, j].item(0)) > 0):
    #                 flag = 1

    flag = int(((N <= N_old) & (N_old > 0)).any())

    return flag

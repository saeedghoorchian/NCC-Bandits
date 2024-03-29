import datetime
import numpy as np

from src.algorithms import utilities



def get_ind_of_stationarity_period_by_t(t, stationarity_periods):
    if t < stationarity_periods[0][0]:
        return 0
    if t > stationarity_periods[-1][-1]:
        return len(stationarity_periods)-1
    for i, period in enumerate(stationarity_periods):
        if period[0] <= t <= period[1]:
            return i
    else:
        raise ValueError(f"t={t} not found")

def general(all_contexts, all_rewards, stationarity_periods, max_num_observations, s_o_max_general):
    time_horizon = all_rewards.shape[0]
    number_of_actions = all_rewards.shape[1]
    context_dimensionality = all_contexts.shape[1]
    num_stationarity_periods = len(stationarity_periods)

    all_perms = utilities.perm_construct(context_dimensionality, max_num_observations)
    number_of_perms_general = all_perms.shape[0]

    selected_context_general = np.zeros((time_horizon, context_dimensionality))
    selected_action_general = np.zeros(time_horizon)
    all_gain_general = np.zeros(time_horizon + 1)

    number_of_unique_state_s_when_applying_perm_i = np.zeros((number_of_perms_general, s_o_max_general))

    # d_os = np.zeros((number_of_perms_general, s_o_max_general)) #Definition: N_os / N_o
    true_prob_so = np.zeros((number_of_perms_general,
                             s_o_max_general))  # Definition: number_of_unique_state_s_when_applying_perm_i / time_horizon

    S_Size = np.zeros(number_of_perms_general)

    true_average_reward = np.zeros((num_stationarity_periods, number_of_perms_general, s_o_max_general, number_of_actions))

    number_of_visits_for_average_reward = np.zeros((num_stationarity_periods, number_of_perms_general, s_o_max_general, number_of_actions))

    sum_of_rewards = np.zeros((num_stationarity_periods, number_of_perms_general, s_o_max_general, number_of_actions))

    for i in range(number_of_perms_general):

        one_perm = all_perms[i]

        O_temp = np.tile(one_perm, [time_horizon, 1])

        all_contexts_temp = all_contexts * O_temp  # a*b element.wise multiplication, np.multiply(a,b) is also correct

        # S_i is the matrix whose rows are the unique(between already existing contexts) state realizations of all of the contexts with observation action one_perm[i]
        S_i = np.unique(all_contexts_temp, axis=0)

        # number of unique state realizations for each domain
        # it has a subtle difference with s_o[i] = number of different states(realizations) with the same observation action all_perms[i]
        S_Size[i] = S_i.shape[0]

        for j in range(int(S_Size[i])):

            all_unique_state_s_when_applying_perm_i = (all_contexts_temp == S_i[j, :]).all(1)
            number_of_unique_state_s_when_applying_perm_i[i, j] = number_of_unique_state_s_when_applying_perm_i[
                                                                      i, j] + np.sum(
                all_unique_state_s_when_applying_perm_i)

            true_prob_so[i, j] = number_of_unique_state_s_when_applying_perm_i[i, j] / time_horizon

            for per_num, period in enumerate(stationarity_periods):
                period_slice = slice(*period)
                all_unique_states_period = all_unique_state_s_when_applying_perm_i[period_slice]
                all_rewards_temp = all_rewards[period_slice][all_unique_states_period, :]

                for k in range(number_of_actions):
                    number_of_visits_for_average_reward[per_num, i, j, k] = all_rewards_temp[:, k].shape[0]
                    sum_of_rewards[per_num, i, j, k] = np.sum(all_rewards_temp[:, k])

                    true_average_reward[per_num, i, j, k] = (
                        sum_of_rewards[per_num, i, j, k] / (number_of_visits_for_average_reward[per_num, i, j, k] or 1)
                    )

    return [true_prob_so, true_average_reward, S_Size]


class Algorithm1_Oracle:
    """Oracle policy

    This Oracle is different from SimOOS_Oracle in that it is designed to work in a piece-wise stationary
    context. This oracle has access to true expected reward values which are piece-wise constant.

    """
    def __init__(self,
                 all_contexts: np.array,
                 all_rewards: np.array,
                 cost_means: np.array,
                 stationarity_periods: list,
                 number_of_actions: int,
                 max_num_observations: int,
                 beta: float = 1.0,
                 ):

        self.name = f"Algorithm1-Oracle (beta={beta})"

        self.time_horizon = all_contexts.shape[0]
        self.context_dimensionality = all_contexts.shape[1]
        self.max_num_observations = max_num_observations
        self.number_of_actions = number_of_actions
        self.beta = beta

        self.validate_stationarity_periods(stationarity_periods)
        self.stationarity_periods = stationarity_periods
        self.num_stationarity_periods = len(stationarity_periods)

        # All possible subsets of features (I in paper)
        self.all_perms = utilities.perm_construct(self.context_dimensionality, self.max_num_observations)
        self.perm_to_index = {}
        for i, perm in enumerate(self.all_perms):
            self.perm_to_index[tuple(perm)] = i

        self.number_of_perms = self.all_perms.shape[0]
        self.s_o = np.zeros(self.number_of_perms)

        self.selected_context = np.zeros((self.time_horizon, self.context_dimensionality))
        self.selected_action = np.zeros(self.time_horizon)
        self.all_gain = np.zeros(self.time_horizon + 1)

        self.collected_gains = np.zeros(self.time_horizon)
        self.collected_rewards = np.zeros(self.time_horizon)
        self.collected_costs = np.zeros(self.time_horizon)

        # debug variables
        self.states = np.zeros(self.time_horizon)
        self.true_average_rewards = np.zeros((self.time_horizon, self.number_of_actions))

        ########################################################################################
        self.feature_values, self.all_feature_counts = utilities.save_feature_values(all_contexts)

        for i in range(self.number_of_perms):
            # psi[i] = number of different partial vectors(realizations) with given observation action self.all_perms[i]
            # How many partial vectors with support given by self.all_perms[i].
            # Equal to cardinality of Psi(I) in the paper. Used to determine Psi_total, for confidence bounds.

            # s_o[i] - size of state array for a given observation action. It is bigger than psi[i] because
            # it also includes states which have None for observed features (although they are unreachable).
            self.s_o[i] = utilities.state_construct(self.all_feature_counts, all_contexts, self.all_perms[i])

        # s_o_max - the largest state vector for all observations, needed to create arrays.
        self.s_o_max = int(np.amax(self.s_o))
        self.Psi_total = int(np.sum(self.s_o))

        # Oracle variables (not present in SimOOS or Alg1)
        self.true_prob_so, self.true_average_reward, self.S_Size = general(
            all_contexts, all_rewards, self.stationarity_periods, self.max_num_observations, self.s_o_max
        )

        self.r_star = np.zeros((self.num_stationarity_periods, self.number_of_perms, self.s_o_max))
        self.action_star = np.zeros((self.num_stationarity_periods, self.number_of_perms, self.s_o_max))

        for num_per in range(self.num_stationarity_periods):
            for i in range(self.number_of_perms):

                for j in range(int(self.S_Size[i])):
                    self.r_star[num_per, i, j] = np.max(self.true_average_reward[num_per, i, j, :])

                    self.action_star[num_per, i, j] = np.argmax(self.true_average_reward[num_per, i, j, :])

        self.value_o = np.zeros((self.num_stationarity_periods, self.number_of_perms))
        self.index_of_observation_action_star = np.zeros(self.num_stationarity_periods, dtype=int)
        self.selected_observation_action_at_t = np.zeros((self.num_stationarity_periods, self.context_dimensionality))


        for num_per in range(self.num_stationarity_periods):
            for i in range(self.number_of_perms):
                self.value_o[num_per, i] = self.beta * np.dot(
                    self.true_prob_so[i, :], self.r_star[num_per, i, :]
                ) - np.dot(self.all_perms[i], cost_means[num_per])

            self.index_of_observation_action_star[num_per] = int(np.argmax(self.value_o[num_per, :]))
            self.selected_observation_action_at_t[num_per, :] = self.all_perms[self.index_of_observation_action_star[num_per]]

    def validate_stationarity_periods(self, stationarity_periods):
        assert stationarity_periods == sorted(stationarity_periods)

        for l, r in zip(stationarity_periods, stationarity_periods[1:]):
            assert l[1] == r[0]

    def choose_features_to_observe(self, t, feature_indices, cost_vector):
        num_per = get_ind_of_stationarity_period_by_t(t, self.stationarity_periods)
        self.observation_action_at_t = self.selected_observation_action_at_t[num_per]
        return [ind for ind, value in enumerate(self.observation_action_at_t) if value]

    def choose_arm(self, t: int, context_at_t: np.array, pool_indices: list):
        """Return best arm's index relative to the pool of available arms.

        Args:
            t: number of trial
            context_at_t: user context at time t. One for each trial, arms don't have contexts.
            pool_indices: indices of arms available at time t.
        """
        num_per = get_ind_of_stationarity_period_by_t(t, self.stationarity_periods)
        s_t = utilities.state_extract(self.feature_values, self.all_feature_counts, context_at_t, self.observation_action_at_t)
        self.states[t] = s_t
        self.true_average_rewards[t, :] = self.true_average_reward[num_per, self.index_of_observation_action_star[num_per], s_t]

        self.action_at_t = int(self.action_star[num_per, self.index_of_observation_action_star[num_per], s_t])
        return pool_indices.index(self.action_at_t)

    def update(self, t, action_index_at_t, reward_at_t, cost_vector_at_t, context_at_t, pool_indices):

        num_per = get_ind_of_stationarity_period_by_t(t, self.stationarity_periods)

        cost_at_t = np.dot(cost_vector_at_t, self.observation_action_at_t)

        action_at_t = pool_indices[action_index_at_t]

        self.selected_context[t, :] = self.selected_observation_action_at_t[num_per]

        self.all_gain[t + 1] = self.all_gain[t] + reward_at_t - cost_at_t

        self.selected_action[t] = action_at_t

        self.collected_gains[t] = reward_at_t - cost_at_t
        self.collected_rewards[t] = reward_at_t
        self.collected_costs[t] = cost_at_t

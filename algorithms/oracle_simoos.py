import math
import datetime
import numpy as np
import cvxpy as cp

import algorithms.utilities as utilities


def general(all_contexts, all_rewards, max_no_red_context, s_o_max_general):
    time_horizon = all_rewards.shape[0]
    number_of_actions = all_rewards.shape[1]
    org_dim_context = all_contexts.shape[1]

    all_perms = utilities.perm_construct(org_dim_context, max_no_red_context)
    number_of_perms_general = all_perms.shape[0]

    selected_context_general = np.zeros((time_horizon, org_dim_context))
    selected_action_general = np.zeros(time_horizon)
    all_gain_general = np.zeros(time_horizon + 1)

    number_of_unique_state_s_when_applying_perm_i = np.zeros((number_of_perms_general, s_o_max_general))

    # d_os = np.zeros((number_of_perms_general, s_o_max_general)) #Definition: N_os / N_o
    true_prob_so = np.zeros((number_of_perms_general,
                             s_o_max_general))  # Definition: number_of_unique_state_s_when_applying_perm_i / time_horizon

    S_Size = np.zeros(number_of_perms_general)

    true_average_reward = np.zeros((number_of_perms_general, s_o_max_general, number_of_actions))

    number_of_visits_for_average_reward = np.zeros((number_of_perms_general, s_o_max_general, number_of_actions))

    sum_of_rewards = np.zeros((number_of_perms_general, s_o_max_general, number_of_actions))

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

            all_rewards_temp = all_rewards[all_unique_state_s_when_applying_perm_i, :]

            for k in range(number_of_actions):
                number_of_visits_for_average_reward[i, j, k] = all_rewards_temp[:, k].shape[0]
                sum_of_rewards[i, j, k] = np.sum(all_rewards_temp[:, k])

                true_average_reward[i, j, k] = sum_of_rewards[i, j, k] / number_of_visits_for_average_reward[i, j, k]

    return [true_prob_so, true_average_reward, S_Size]


class SimOOS_Oracle:
    def __init__(self,
                 all_contexts: np.array,
                 all_rewards: np.array,
                 cost_vector: np.array,
                 number_of_actions: int,
                 max_no_red_context: int,
                 beta_SimOOS: float,
                 ):

        self.name = f"SimOOS-Oracle (beta={beta_SimOOS})"

        self.time_horizon = all_contexts.shape[0]
        self.org_dim_context = all_contexts.shape[1]
        self.max_no_red_context = max_no_red_context
        self.number_of_actions = number_of_actions
        self.beta_SimOOS = beta_SimOOS

        # All possible subsets of features (I in paper)
        self.all_perms = utilities.perm_construct(self.org_dim_context, self.max_no_red_context)
        self.perm_to_index = {}
        for i, perm in enumerate(self.all_perms):
            self.perm_to_index[tuple(perm)] = i

        self.number_of_perms_SimOOS = self.all_perms.shape[0]
        self.s_o = np.zeros(self.number_of_perms_SimOOS)

        self.selected_context_SimOOS = np.zeros((self.time_horizon, self.org_dim_context))
        self.selected_action_SimOOS = np.zeros(self.time_horizon)
        self.all_gain_SimOOS = np.zeros(self.time_horizon + 1)

        self.collected_gains_SimOOS = np.zeros(self.time_horizon)
        self.collected_rewards_SimOOS = np.zeros(self.time_horizon)
        self.collected_costs_SimOOS = np.zeros(self.time_horizon)

        # debug variables
        self.states = np.zeros(self.time_horizon)
        self.true_average_rewards = np.zeros((self.time_horizon, self.number_of_actions))

        ########################################################################################
        self.feature_values, self.all_feature_counts = utilities.save_feature_values(all_contexts)

        for i in range(self.number_of_perms_SimOOS):
            # psi[i] = number of different partial vectors(realizations) with given observation action self.all_perms[i]
            # How many partial vectors with support given by self.all_perms[i].
            # Equal to cardinality of Psi(I) in the paper. Used to determine Psi_total, for confidence bounds.

            # s_o[i] - size of state array for a given observation action. It is bigger than psi[i] because
            # it also includes states which have None for observed features (although they are unreachable).
            self.s_o[i] = utilities.state_construct(self.all_feature_counts, all_contexts, self.all_perms[i])

        # s_o_max_SimOOS - the largest state vector for all observations, needed to create arrays.
        self.s_o_max_SimOOS = int(np.amax(self.s_o))
        self.Psi_total = int(np.sum(self.s_o))

        # Oracle variables (not present in SimOOS)
        self.true_prob_so, self.true_average_reward, self.S_Size = general(
            all_contexts, all_rewards, self.max_no_red_context, self.s_o_max_SimOOS
        )

        self.r_star = np.zeros((self.number_of_perms_SimOOS, self.s_o_max_SimOOS))
        self.action_star = np.zeros((self.number_of_perms_SimOOS, self.s_o_max_SimOOS))

        for i in range(self.number_of_perms_SimOOS):

            for j in range(int(self.S_Size[i])):
                self.r_star[i, j] = np.max(self.true_average_reward[i, j, :])

                self.action_star[i, j] = np.argmax(self.true_average_reward[i, j, :])

        self.value_o = np.zeros(self.number_of_perms_SimOOS)

        for i in range(self.number_of_perms_SimOOS):
            self.value_o[i] = self.beta_SimOOS * np.dot(self.true_prob_so[i, :], self.r_star[i, :]) - np.dot(self.all_perms[i],
                                                                                                cost_vector)

        self.index_of_observation_action_star = np.argmax(self.value_o)

        self.selected_observation_action_at_t = self.all_perms[self.index_of_observation_action_star]

    def choose_features_to_observe(self, t, feature_indices, cost_vector):
        self.observation_action_at_t = self.selected_observation_action_at_t
        return [ind for ind, value in enumerate(self.observation_action_at_t) if value]

    def choose_arm(self, t: int, context_at_t: np.array, pool_indices: list):
        """Return best arm's index relative to the pool of available arms.

        Args:
            t: number of trial
            context_at_t: user context at time t. One for each trial, arms don't have contexts.
            pool_indices: indices of arms available at time t.
        """
        s_t = utilities.state_extract(self.feature_values, self.all_feature_counts, context_at_t, self.observation_action_at_t)
        self.states[t] = s_t
        self.true_average_rewards[t, :] = self.true_average_reward[self.index_of_observation_action_star, s_t]

        self.action_at_t = int(self.action_star[self.index_of_observation_action_star, s_t])
        return pool_indices.index(self.action_at_t)

    def update(self, t, action_index_at_t, reward_at_t, cost_vector_at_t, context_at_t, pool_indices):
        if t % 500 == 0:
            print(f"Trial {t}, time {datetime.datetime.now()}")

        cost_at_t = np.dot(cost_vector_at_t, self.observation_action_at_t)

        action_at_t = pool_indices[action_index_at_t]

        self.selected_context_SimOOS[t, :] = self.selected_observation_action_at_t

        self.all_gain_SimOOS[t + 1] = self.all_gain_SimOOS[t] + reward_at_t - cost_at_t

        self.selected_action_SimOOS[t] = action_at_t

        self.collected_gains_SimOOS[t] = reward_at_t - cost_at_t
        self.collected_rewards_SimOOS[t] = reward_at_t
        self.collected_costs_SimOOS[t] = cost_at_t

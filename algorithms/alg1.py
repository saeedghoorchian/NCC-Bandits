import collections
import math
import datetime
import numpy as np
import cvxpy as cp

import algorithms.utilities as utilities


class Algorithm1:

    def __init__(self,
                 all_contexts: np.array,
                 number_of_actions: int,
                 max_no_red_context: int,
                 beta: float,
                 delta: float,
                 window_length: int,
                 feature_flag: bool=False,
                 oracle_costs: bool=False,
                 costs_range: float=None,
                 ):

        self.feature_flag = feature_flag
        self.oracle_costs = oracle_costs
        self.costs_range = costs_range
        self.name = f"Algorithm1 (beta={beta}, delta={delta}, w={window_length})"

        self.time_horizon = all_contexts.shape[0]
        self.org_dim_context = all_contexts.shape[1]
        self.max_no_red_context = max_no_red_context
        self.number_of_actions = number_of_actions
        self.beta = beta
        self.delta = delta
        self.w = window_length

        # All possible subsets of features (I in paper)
        self.all_perms = utilities.perm_construct(self.org_dim_context, self.max_no_red_context)
        self.perm_to_index = {}
        for i, perm in enumerate(self.all_perms):
            self.perm_to_index[tuple(perm)] = i

        self.number_of_perms_SimOOS = self.all_perms.shape[0]
        self.s_o = np.zeros(self.number_of_perms_SimOOS)
        self.psi = np.zeros(self.number_of_perms_SimOOS)

        self.selected_context_SimOOS = np.zeros((self.time_horizon, self.org_dim_context))
        self.selected_action_SimOOS = np.zeros(self.time_horizon)
        self.all_gain_SimOOS = np.zeros(self.time_horizon + 1)

        self.collected_gains_SimOOS = np.zeros(self.time_horizon)
        self.collected_rewards_SimOOS = np.zeros(self.time_horizon)
        self.collected_costs_SimOOS = np.zeros(self.time_horizon)

        ########################################################################################
        self.feature_values, self.all_feature_counts = utilities.save_feature_values(all_contexts)

        for i in range(self.number_of_perms_SimOOS):
            # psi[i] = number of different partial vectors(realizations) with given observation action self.all_perms[i]
            # How many partial vectors with support given by self.all_perms[i].
            # Equal to cardinality of Psi(I) in the paper. Used to determine Psi_total, for confidence bounds.

            # s_o[i] - size of state array for a given observation action. It is bigger than psi[i] because
            # it also includes states which have None for observed features (although they are unreachable).
            self.psi[i], self.s_o[i] = utilities.state_construct(self.all_feature_counts, all_contexts,
                                                                self.all_perms[i])

        # s_o = contains the number of all different states(reaqlizations) with the same observation action
        # up to "max_no_red_context" number of permitted observations.
        self.s_o_max_SimOOS = int(np.amax(self.s_o))
        self.Psi_total = int(np.sum(self.psi))

        # Sliding windows and counters
        # Different Tau variables are implemented as binary vectors of len=window, for easy dot products with window.
        self.reward_window = collections.deque(maxlen=self.w)
        self.Tau_aso = collections.deque(maxlen=self.w)
        # # Old unvectorized version.
        # self.Tau_aso = []
        # for a in range(self.number_of_actions):
        #     self.Tau_aso.append([])
        #     for s in range(self.s_o_max_SimOOS):
        #         self.Tau_aso[-1].append([])
        #         for o in range(self.number_of_perms_SimOOS):
        #             self.Tau_aso[-1][-1].append(collections.deque(maxlen=self.w))

        self.cost_window = []
        self.Tau_f = []
        for f in range(self.org_dim_context):
            self.cost_window.append(collections.deque(maxlen=self.w))
            self.Tau_f.append(collections.deque(maxlen=self.w))

        # a - action, s - state (partial vector), o - observation (subset of features), f - feature
        self.r_hat_t = np.zeros((self.number_of_actions, self.s_o_max_SimOOS, self.number_of_perms_SimOOS))
        self.c_hat_t = np.zeros(self.org_dim_context)
        self.conf1_t = np.zeros((self.number_of_actions, self.s_o_max_SimOOS, self.number_of_perms_SimOOS))
        self.N_t_aso = np.zeros((self.number_of_actions, self.s_o_max_SimOOS, self.number_of_perms_SimOOS))
        self.N_t_f = np.zeros(self.org_dim_context)
        self.N_t_o = np.zeros(self.number_of_perms_SimOOS)
        self.N_t_os = np.zeros((self.number_of_perms_SimOOS, self.s_o_max_SimOOS))
        self.d_t_os = np.zeros((self.number_of_perms_SimOOS, self.s_o_max_SimOOS))  # Definition: N_t_os / N_t_o

        # Values needed for history.
        # Last observed feature subset.
        self.observation_action_at_t = None
        # Last chosen action
        self.action_at_t = None

        # Important Variables
        # Optimistic reward estimates, r_hat + conf_1 for all actions, realizations (states), and permutations.
        self.upsilon_t = np.zeros((self.number_of_actions, self.s_o_max_SimOOS, self.number_of_perms_SimOOS))

        # optimistic action policy. This is actually the h_hat in the paper
        # For each state and observation we store not the optimal action, but rather all actions sorted
        # by their optimistic reward estimates. This is needed so that at each trial we can choose
        # the best arm from the pool of available arms.
        self.a_hat_t = np.zeros((self.s_o_max_SimOOS, self.number_of_perms_SimOOS, self.number_of_actions))

        # optimistic value of observation. This is actually the V_hat in the paper
        self.nu_t = np.zeros(self.number_of_perms_SimOOS)

        self.index_of_observation_action_at_t = 0

        self.selected_observation_action_at_t = np.zeros(self.org_dim_context)

        # Debug variables
        self.ucbs = np.zeros((self.time_horizon + 1, self.number_of_actions))
        self.rewards = np.zeros((self.time_horizon + 1, self.number_of_actions))
        self.confidences = np.zeros((self.time_horizon + 1, self.number_of_actions))

        self.costs = np.zeros((self.time_horizon + 1, self.org_dim_context))
        self.costs_plus_bound = np.zeros((self.time_horizon + 1, self.org_dim_context))
        self.costs_minus_bound = np.zeros((self.time_horizon + 1, self.org_dim_context))
        self.c_hats = np.zeros((self.time_horizon + 1, self.org_dim_context))
        self.cost_conf_int = np.zeros((self.time_horizon + 1, self.org_dim_context))
        self.N_t_fs = np.zeros((self.time_horizon + 1, self.org_dim_context))

        self.r_stars = np.zeros((self.time_horizon + 1, self.s_o_max_SimOOS))
        self.nus = np.zeros((self.time_horizon + 1, self.number_of_perms_SimOOS))
        self.rounds = 0
        self.random_actions = 0

    def find_optimal_policy(self, t, cost_vector):
        self.rounds += 1

        # Optimistic cost value c_tilde is same for all observations.
        c_tilde = np.zeros(self.org_dim_context)
        for f in range(self.org_dim_context):
            conf_int_before_min = math.sqrt(
                math.log((self.org_dim_context * self.w * self.time_horizon) / self.delta) / (
                        2 * self.N_t_f[f]
                )
            )
            if self.costs_range is not None:
                confidence_interval_cost_f = min(self.costs_range, conf_int_before_min)
            else:
                confidence_interval_cost_f = min(1, conf_int_before_min)
            self.cost_conf_int[t, f] = conf_int_before_min

            self.costs_plus_bound[t, f] = self.c_hat_t[f] + confidence_interval_cost_f
            self.costs_minus_bound[t, f] = self.c_hat_t[f] - confidence_interval_cost_f

            c_tilde[f] = self.c_hat_t[f] - confidence_interval_cost_f

            if self.oracle_costs:
                c_tilde[f] = cost_vector[f]

        self.c_hats[t, :] = self.c_hat_t
        self.costs[t, :] = c_tilde
        self.N_t_fs[t, :] = self.N_t_f

        for i in range(self.number_of_perms_SimOOS):

            if i == 0:
                # r_star[observation][state]
                r_star = [[] for i in range(self.number_of_perms_SimOOS)]  # the r_star values will be stored here

            z = int(self.s_o[i])
            for j in range(z):

                for k in range(self.number_of_actions):

                    if self.N_t_aso[k, j, i] == 0:
                        self.upsilon_t[k, j, i] = 1  # min(1, !) = 1
                        confidence_interval_reward = 0
                    else:
                        confidence_interval_reward = min(1, math.sqrt(
                            math.log((self.number_of_actions * self.Psi_total * self.w * self.time_horizon) / self.delta) / (
                                    2 * self.N_t_aso[k, j, i])))
                        self.upsilon_t[k, j, i] = self.r_hat_t[k, j, i] + confidence_interval_reward

                    self.conf1_t[k, j, i] = confidence_interval_reward

                r_star[i].append(np.max(self.upsilon_t[:, j, i]))

                # which action has the highest UCB. This is actually the h_hat in the paper
                self.a_hat_t[j, i] = np.argsort(self.upsilon_t[:, j, i])[::-1]  # descending sort

                # Tie breaking rule - random.
                if np.all(np.isclose(self.upsilon_t[:, j, i], self.upsilon_t[0, j, i])):
                    np.random.shuffle(self.a_hat_t[j, i])

            prob_hat = self.d_t_os[i, :z]

            # number_of_perms_SimOOS = |P(D)| in paper
            confidence_interval_prob = min(
                1,
                math.sqrt(
                    2 * self.Psi_total * math.log((2 * self.time_horizon * self.number_of_perms_SimOOS) / self.delta)
                    / (2 * self.N_t_o[i])
                )
           )

            observation_action_in_optimization = self.all_perms[i]

            # Construct the problem, Equation (15) in the paper
            prob_tilde = cp.Variable(z)

            objective = cp.Maximize(
                (self.beta * (np.array(r_star[i]) * prob_tilde))
                - np.dot(observation_action_in_optimization, c_tilde)
            )

            r_star_array = np.zeros(self.s_o_max_SimOOS)
            for ind in range(len(r_star[i])):
                r_star_array[ind] += r_star[i][ind]

            self.r_stars[t, :] = r_star_array

            constraints = [cp.norm((prob_tilde - prob_hat), 1) <= confidence_interval_prob, cp.sum(prob_tilde) == 1]

            prob = cp.Problem(objective, constraints)

            prob.solve()

            # Similar to paper, set V_hat[i] = nu_t[i] as the maximizer
            self.nu_t[i] = prob.value

        self.nus[t, :] = self.nu_t
        self.index_of_observation_action_at_t = np.argmax(
            self.nu_t)  # Find which all_perms[i](= index_of_observation_action_at_t) gives the highest prob_tilde

        self.selected_observation_action_at_t = self.all_perms[self.index_of_observation_action_at_t]

    def choose_features_to_observe(self, t, feature_indices, cost_vector):
        if t < self.number_of_perms_SimOOS:
            # First go through all subsets of features once.
            self.selected_observation_action_at_t = self.all_perms[t]
        else:
            # Optimistic Policy Optimization
            self.find_optimal_policy(t, cost_vector)

        self.observation_action_at_t = self.selected_observation_action_at_t
        return [ind for ind, value in enumerate(self.observation_action_at_t) if value]

    def choose_arm(self, t: int, context_at_t: np.array, pool_indices: list):
        """Return best arm's index relative to the pool of available arms.

        Args:
            t: number of trial
            context_at_t: user context at time t. One for each trial, arms don't have contexts.
            pool_indices: indices of arms available at time t.
        """
        if t < self.number_of_perms_SimOOS:
            # Random Source Selection Part
            self.action_at_t = np.random.choice(pool_indices)
        else:
            # Optimistic Policy Optimization
            s_t = utilities.state_extract(self.feature_values, self.all_feature_counts, context_at_t,
                                         self.observation_action_at_t)

            # Only actions choose from the available pool of actions. This is needed for yahoo r6a/b experiments.
            pool_indices_set = set(pool_indices)

            # a_hat_t has actions sorted by their optimistic reward estimates.
            for action in self.a_hat_t[s_t, self.index_of_observation_action_at_t]:
                if action in pool_indices_set:
                    action_at_t = action
                    break
            else:
                raise ValueError(f"No action found at time {t}, something went wrong.")

            self.action_at_t = int(action_at_t)
            self.ucbs[t] = self.upsilon_t[:, s_t, self.index_of_observation_action_at_t]
            self.rewards[t] = self.r_hat_t[:, s_t, self.index_of_observation_action_at_t]
            self.confidences[t] = self.conf1_t[:, s_t, self.index_of_observation_action_at_t]

        return pool_indices.index(self.action_at_t)

    def update_tau_aso(self, t, action_at_t, s_t, o_t):
        update_tensor = np.zeros((self.number_of_actions, self.s_o_max_SimOOS, self.number_of_perms_SimOOS))
        update_tensor[action_at_t, s_t, o_t] = 1
        if t > self.w:
            self.Tau_aso.popleft()
        self.Tau_aso.append(update_tensor)

        tau_aso_array = np.array(self.Tau_aso)
        # There is not max(1, count) for N_t_aso because in the places where it is used the case when it is 0 is
        # considered. This is bad design, but I tried to make this consistent with SimOOS code in those parts.
        self.N_t_aso = np.count_nonzero(tau_aso_array, axis=0)
        sum_of_window_rewards = np.tensordot(np.array(self.reward_window), tau_aso_array, axes=1)
        self.r_hat_t = sum_of_window_rewards / np.maximum(1, self.N_t_aso)

        # # Old unvectorized version
        # for obs in range(self.number_of_perms_SimOOS):
        #     for state in range(int(self.s_o[obs])):
        #         for a in range(self.number_of_actions):
        #             if t > self.w:
        #                 self.Tau_aso[a][state][obs].popleft()
        #             if a == action_at_t and state == s_t and obs == o_t:
        #                 self.Tau_aso[a][state][obs].append(1)
        #             else:
        #                 self.Tau_aso[a][state][obs].append(0)
        #
        #             tau_aso_array = np.array(self.Tau_aso[a][state][obs])
        #             self.N_t_aso[a, state, obs] = np.count_nonzero(tau_aso_array)
        #
        #             sum_of_window_rewards = np.dot(np.array(self.reward_window), tau_aso_array)
        #             # r_hat_t is the empirical average reward.
        #             self.r_hat_t[a, state, obs] = sum_of_window_rewards / self.N_t_aso[a][state][obs]

    def update(self, t, action_index_at_t, reward_at_t, cost_vector_at_t, context_at_t, pool_indices):
        if t % 500 == 0:
            print(f"Round {t}, time {datetime.datetime.now()}")

        cost_at_t = np.dot(cost_vector_at_t, self.observation_action_at_t)

        action_at_t = pool_indices[action_index_at_t]

        s_t = utilities.state_extract(self.feature_values, self.all_feature_counts, context_at_t,
                                     self.observation_action_at_t)
        # Sanity check
        # created_context = utilities.state_create(s_t, self.feature_values)
        # assert np.all(context_at_t == created_context)

        # observation at time t
        # first we try all possible observations once
        o_t = t if t < self.number_of_perms_SimOOS else self.index_of_observation_action_at_t

        # Update all counters (move the window)
        # Unlike SimOOS, here when window moves - counters update for all action-state-observation tuples,
        # not just the ones chosen in this trial.

        # Reward window and relevant counters
        if t > self.w:
            self.reward_window.popleft()
        self.reward_window.append(reward_at_t)

        self.update_tau_aso(t, action_at_t, s_t, o_t)

        # Cost window and counters
        for f in range(self.org_dim_context):
            if t > self.w:
                self.cost_window[f].popleft()
                self.Tau_f[f].popleft()

            self.cost_window[f].append(cost_vector_at_t[f])
            self.Tau_f[f].append(self.all_perms[o_t][f])

            tau_f_array = np.array(self.Tau_f[f])

            self.N_t_f[f] = max(1, np.count_nonzero(tau_f_array))
            sum_of_window_costs_one_feature = np.dot(np.array(self.cost_window[f]), tau_f_array)
            self.c_hat_t[f] = sum_of_window_costs_one_feature / self.N_t_f[f]

        self.selected_context_SimOOS[t, :] = self.selected_observation_action_at_t
        if t < self.number_of_perms_SimOOS:
            # This makes sense, since in this for loop, both N_t_os and N_t_o are being updated at every time
            # and each observation is seen only once.
            self.N_t_o[o_t] += 1
            self.N_t_os[o_t, s_t] += 1
            self.d_t_os[o_t, s_t] = 1

        else:
            # Optimistic Policy Optimization
            # Update counters for all substates of the seen state.
            substates, substate_observations = utilities.generate_substates(
                context_at_t, self.observation_action_at_t
            )
            for sub, sub_obs in zip(substates, substate_observations):
                sub_s_t = utilities.state_extract(self.feature_values, self.all_feature_counts, sub, sub_obs)
                sub_obs_index = self.perm_to_index[tuple(sub_obs)]
                self.N_t_o[sub_obs_index] += 1
                self.N_t_os[sub_obs_index, sub_s_t] += 1
                self.d_t_os[sub_obs_index, :] = self.N_t_os[sub_obs_index, :] / self.N_t_o[sub_obs_index]

        self.all_gain_SimOOS[t + 1] = self.all_gain_SimOOS[t] + reward_at_t - cost_at_t

        self.selected_action_SimOOS[t] = action_at_t

        self.collected_gains_SimOOS[t] = reward_at_t - cost_at_t
        self.collected_rewards_SimOOS[t] = reward_at_t
        self.collected_costs_SimOOS[t] = cost_at_t


def run_algorithm1(all_contexts, all_rewards, max_no_red_context, beta, cost_vector, delta, window_length):

    alg = Algorithm1(
        all_contexts=all_contexts,
        max_no_red_context=max_no_red_context,
        number_of_actions=all_rewards.shape[1],
        beta=beta,
        delta=delta,
        window_length=window_length,
    )

    pool_indices = list(range(1, all_rewards.shape[1]))

    for t in range(alg.time_horizon):
        context_at_t = all_contexts[t, :]
        feature_indices = [ind for ind, c in enumerate(context_at_t)]
        features_to_observe = alg.choose_features_to_observe(t, feature_indices, cost_vector)

        observed_context_at_t = np.array(
            [
                feature if index in features_to_observe else None
                for index, feature in enumerate(context_at_t)
            ]
        )

        action_index_at_t = alg.choose_arm(t, observed_context_at_t, pool_indices)
        action_at_t = pool_indices[action_index_at_t]
        reward_at_t = all_rewards[t, action_at_t]
        cost_at_t = np.sum(c for ind, c in enumerate(cost_vector) if ind in features_to_observe)

        alg.update(t, action_index_at_t, reward_at_t, cost_at_t, observed_context_at_t, pool_indices)

    average_gain_SimOOS = alg.all_gain_SimOOS[1:alg.time_horizon + 1] / np.arange(1, alg.time_horizon + 1)

    return [alg.selected_context_SimOOS, alg.selected_action_SimOOS, average_gain_SimOOS, alg.all_gain_SimOOS,
            alg.number_of_perms_SimOOS, alg.s_o_max_SimOOS, alg.Psi_total, alg.collected_gains_SimOOS, alg.collected_rewards_SimOOS,
            alg.collected_costs_SimOOS]

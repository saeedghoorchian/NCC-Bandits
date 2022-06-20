import math
import datetime
import numpy as np
import cvxpy as cp

from src.algorithms import utilities


class SimOOSAlgorithm:
    def __init__(self,
                 all_contexts: np.array,
                 number_of_actions: int,
                 max_no_red_context: int,
                 beta_SimOOS: float,
                 delta_SimOOS: float,
                 ):

        self.name = f"SimOOS (beta={beta_SimOOS}, delta={delta_SimOOS})"

        self.time_horizon = all_contexts.shape[0]
        self.org_dim_context = all_contexts.shape[1]
        self.max_no_red_context = max_no_red_context
        self.number_of_actions = number_of_actions
        self.beta_SimOOS = beta_SimOOS
        self.delta_SimOOS = delta_SimOOS

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
        self.states = np.zeros(self.time_horizon)

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

        # Define the counters and variables
        # a - action, s - state (partial vector), o - observation (subset of features)
        self.r_hat_t = np.zeros((self.number_of_actions, self.s_o_max_SimOOS, self.number_of_perms_SimOOS))
        self.conf1_t = np.zeros((self.number_of_actions, self.s_o_max_SimOOS, self.number_of_perms_SimOOS))
        self.N_t_aso = np.zeros((self.number_of_actions, self.s_o_max_SimOOS, self.number_of_perms_SimOOS))
        self.N_t_o = np.zeros(self.number_of_perms_SimOOS)
        self.N_t_os = np.zeros((self.number_of_perms_SimOOS, self.s_o_max_SimOOS))
        self.d_t_os = np.zeros((self.number_of_perms_SimOOS, self.s_o_max_SimOOS))  # Definition: N_t_os / N_t_o
        self.N_t_as = np.zeros((self.number_of_actions, self.s_o_max_SimOOS))

        # N_old_os = np.zeros((self.number_of_perms_SimOOS, self.s_o_max_SimOOS))
        # N_old_as = np.zeros((self.number_of_actions, self.s_o_max_SimOOS))
        self.N_old_aso = np.zeros((self.number_of_actions, self.s_o_max_SimOOS, self.number_of_perms_SimOOS))

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

        self.new_round = 1  # so that it can run at the beginning

        self.ucbs = np.zeros((self.time_horizon + 1, self.number_of_actions))
        self.rewards = np.zeros((self.time_horizon + 1, self.number_of_actions))
        self.confidences = np.zeros((self.time_horizon + 1, self.number_of_actions))

        self.costs = np.zeros((self.time_horizon + 1, self.org_dim_context))

        self.nus = np.zeros((self.time_horizon + 1, self.number_of_perms_SimOOS))
        self.rounds = 0

    def initialize_new_round(self, t, cost_vector):
        self.rounds += 1

        self.costs[t, :] = cost_vector
        for i in range(self.number_of_perms_SimOOS):

            if i == 0:
                F = [[] for i in range(self.number_of_perms_SimOOS)]  # the r_star values will be stored here

            z = int(self.s_o[i])
            for j in range(z):

                for k in range(self.number_of_actions):

                    if self.N_t_aso[k, j, i] == 0:
                        self.upsilon_t[k, j, i] = 1  # min(1, !) = 1
                        confidence_interval_1 = 0
                    else:
                        confidence_interval_1 = min(1, math.sqrt(
                            math.log((20 * self.Psi_total * self.number_of_actions * (t ** 5)) / self.delta_SimOOS) / (
                                    2 * self.N_t_aso[k, j, i])))
                        self.upsilon_t[k, j, i] = self.r_hat_t[k, j, i] + confidence_interval_1
                    # if t == 106:
                    #     print(f"s_t: {j} obs: {i}, confidence: {confidence_interval_1}")
                    self.conf1_t[k, j, i] = confidence_interval_1

                F[i].append(np.max(self.upsilon_t[:, j, i]))

                # which action has the highest UCB. This is actually the h_hat in the paper
                self.a_hat_t[j, i] = np.argsort(self.upsilon_t[:, j, i])[::-1]  # descending sort

            prob_hat = self.d_t_os[i, :z]

            confidence_interval_2 = min(1, math.sqrt(
                (10 * self.Psi_total * math.log(4 * t / self.delta_SimOOS)) / self.N_t_o[i]))

            observation_action_in_optimization = self.all_perms[i]

            # Construct the problem, Equation (3) in the paper
            prob_tilde = cp.Variable(z)

            objective = cp.Maximize(
                (self.beta_SimOOS * (np.array(F[i]) * prob_tilde))
                - np.dot(observation_action_in_optimization, cost_vector)
            )

            constraints = [cp.norm((prob_tilde - prob_hat), 1) <= confidence_interval_2, cp.sum(prob_tilde) == 1]

            prob = cp.Problem(objective, constraints)

            prob.solve()

            # Similar to paper, set V_hat[i] = nu_t[i] as the maximizer
            self.nu_t[i] = prob.value

        self.nus[t, :] = self.nu_t
        self.index_of_observation_action_at_t = np.argmax(
            self.nu_t)  # Find which all_perms[i](= index_of_observation_action_at_t) gives the highest prob_tilde

        self.selected_observation_action_at_t = self.all_perms[self.index_of_observation_action_at_t]

        self.N_old_aso = np.copy(self.N_t_aso)

        self.new_round = 0  # New round initialized, no new round needed.

    def choose_features_to_observe(self, t, feature_indices, cost_vector):
        if t < self.number_of_perms_SimOOS:
            # First go through all subsets of features once.
            self.observation_action_at_t = self.all_perms[t]
        else:
            # Optimistic Policy Optimization
            if self.new_round == 1:
                self.initialize_new_round(t, cost_vector)
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
            s_t = utilities.state_extract(self.feature_values, self.all_feature_counts, context_at_t, self.observation_action_at_t)
            # Sanity check
            # created_context = utilities.state_create(s_t, self.feature_values)
            # assert np.all(context_at_t == created_context)

            # If some of the actions have not been chosen yet - choose them.
            action_at_t_temp = np.argwhere(
                self.N_t_aso[:, s_t, self.index_of_observation_action_at_t] == 0
            )
            # But only choose from the available pool of actions.
            pool_indices_set = set(pool_indices)
            action_at_t_temp_set = set(action_at_t_temp.T[0]).intersection(pool_indices_set)

            if len(action_at_t_temp_set) > 0:
                action_at_t = next(iter(action_at_t_temp_set))
            else:
                # If all actions have been tried - choose best possible action of all available ones.
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
            self.states[t] = s_t


        return pool_indices.index(self.action_at_t)

    def update(self, t, action_index_at_t, reward_at_t, cost_vector_at_t, context_at_t, pool_indices):

        if t % 500 == 0:
            print(f"Trial {t}, time {datetime.datetime.now()}")

        cost_at_t = np.dot(cost_vector_at_t, self.observation_action_at_t)

        action_at_t = pool_indices[action_index_at_t]

        s_t = utilities.state_extract(self.feature_values, self.all_feature_counts, context_at_t, self.observation_action_at_t)
        # Sanity check
        # created_context = utilities.state_create(s_t, self.feature_values)
        # assert np.all(context_at_t == created_context)

        if t < self.number_of_perms_SimOOS:
            # Random Source Selection Part
            # r_hat_t is basically the empirical average reward.
            self.r_hat_t[action_at_t, s_t, t] = (self.N_t_aso[action_at_t, s_t, t] * self.r_hat_t[action_at_t, s_t, t] + reward_at_t) / (
                        self.N_t_aso[action_at_t, s_t, t] + 1)

            self.N_t_aso[action_at_t, s_t, t] += 1
            self.N_t_o[t] += 1
            self.N_t_os[t, s_t] += 1
            self.d_t_os[
                t, s_t
            ] = 1  # This makes sense, since in this for loop, both N_t_os and N_t_o are being updated at every time.
            # Moreover, each observation is seen only once.
            self.N_t_as[action_at_t, s_t] += 1

            self.selected_context_SimOOS[t, :] = np.array([c if c is not None else 0 for c in context_at_t])

        else:
            # Optimistic Policy Optimization
            self.r_hat_t[action_at_t, s_t, self.index_of_observation_action_at_t] = (self.N_t_aso[
                                                                               action_at_t, s_t, self.index_of_observation_action_at_t] *
                                                                           self.r_hat_t[
                                                                               action_at_t, s_t, self.index_of_observation_action_at_t] + reward_at_t) / (
                                                                                  self.N_t_aso[
                                                                                      action_at_t, s_t, self.index_of_observation_action_at_t] + 1)
            self.N_t_aso[action_at_t, s_t, self.index_of_observation_action_at_t] = self.N_t_aso[
                                                                              action_at_t, s_t, self.index_of_observation_action_at_t] + 1
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

            self.N_t_as[action_at_t, s_t] += 1

            self.new_round = utilities.is_round_over(self.N_old_aso, self.N_t_aso)

            self.selected_context_SimOOS[t, :] = self.selected_observation_action_at_t

        # This part is common both for random observation selection part and for optimistic policy optimization.
        self.all_gain_SimOOS[t + 1] = self.all_gain_SimOOS[t] + reward_at_t - cost_at_t

        self.selected_action_SimOOS[t] = action_at_t

        self.collected_gains_SimOOS[t] = reward_at_t - cost_at_t
        self.collected_rewards_SimOOS[t] = reward_at_t
        self.collected_costs_SimOOS[t] = cost_at_t


def run_new_SimOOS(all_contexts, all_rewards, max_no_red_context, beta_SimOOS, cost_vector, delta_SimOOS):

    alg = SimOOSAlgorithm(
        all_contexts=all_contexts,
        max_no_red_context=max_no_red_context,
        number_of_actions=all_rewards.shape[1],
        beta_SimOOS=beta_SimOOS,
        delta_SimOOS=delta_SimOOS,
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
            alg.number_of_perms_SimOOS, alg.s_o_max_SimOOS, alg.s_o_total_SimOOS, alg.collected_gains_SimOOS, alg.collected_rewards_SimOOS,
            alg.collected_costs_SimOOS]

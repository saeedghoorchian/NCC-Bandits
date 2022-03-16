import collections
import itertools
import math
import datetime
from collections import defaultdict

import numpy as np
import cvxpy as cp


class Algorithm1:

    def full_perm_construct(self, size: int) -> np.array:
        all_perms = np.zeros((2 ** size, size))

        for i in range(2 ** size):
            bin_str = np.binary_repr(i, width=size)
            bin_arr = np.fromstring(bin_str, 'u1') - ord('0')
            all_perms[i, :] = bin_arr

        return all_perms

    def perm_construct(self, org_dim_context: int, max_no_red_context: int) -> np.array:
        """Computes all possible observation actions(permutations) for org_dim_context sources(features) and max_no_red_context
         max_no_red_context is the maximum number of sources that can be selected.
        """

        # If all permutations are needed - a quicker procedure is available.
        if org_dim_context == max_no_red_context:
            return self.full_perm_construct(org_dim_context)

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

    def state_construct(self, all_feature_counts: np.array, all_contexts: np.array, one_perm: np.array) -> tuple:
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

    def save_feature_values(self, all_contexts) -> tuple:
        """Save unique values for each feature and their count.

        This is used later to enumerate all possible states and get index of state from
        observed context (state = partial vector in paper).
        """
        org_dim_context = all_contexts.shape[1]
        feature_values = defaultdict(list)
        all_feature_counts = np.zeros(org_dim_context)

        for i in range(org_dim_context):
            unique_features = np.unique(all_contexts[:, i])
            # None represents not observed feature
            values = [None] + sorted(list(unique_features))
            feature_values[i] = values

            all_feature_counts[i] = len(values)

        return feature_values, all_feature_counts

    def state_extract(self, feature_values, all_feature_counts, context_at_t, observation_action_at_t) -> int:
        """Return the state index by context and observation.

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

    def state_create(self, state_index, feature_values):
        org_dim_context = len(feature_values)

        context = np.zeros(org_dim_context)
        for i in reversed(range(org_dim_context)):
            feature_count = len(feature_values[i])
            feature_index = state_index % feature_count
            context[i] = feature_values[i][feature_index]
            state_index -= feature_index
            state_index = int(state_index/feature_count)

        return np.array([c if not np.isnan(c) else None for c in context])

    def __init__(self,
                 all_contexts: np.array,
                 number_of_actions: int,
                 max_no_red_context: int,
                 beta: float,
                 delta: float,
                 window_length: int,
                 ):

        self.name = f"Algorithm1 (beta={beta}, delta={delta}, w={window_length})"

        self.time_horizon = all_contexts.shape[0]
        self.org_dim_context = all_contexts.shape[1]
        self.max_no_red_context = max_no_red_context
        self.number_of_actions = number_of_actions
        self.beta = beta
        self.delta = delta
        self.w = window_length

        # All possible subsets of features (I in paper)
        self.all_perms = self.perm_construct(self.org_dim_context, self.max_no_red_context)

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
        self.feature_values, self.all_feature_counts = self.save_feature_values(all_contexts)

        for i in range(self.number_of_perms_SimOOS):
            # psi[i] = number of different partial vectors(realizations) with given observation action self.all_perms[i]
            # How many partial vectors with support given by self.all_perms[i].
            # Equal to cardinality of Psi(I) in the paper. Used to determine Psi_total, for confidence bounds.

            # s_o[i] - size of state array for a given observation action. It is bigger than psi[i] because
            # it also includes states which have None for observed features (although they are unreachable).
            self.psi[i], self.s_o[i] = self.state_construct(self.all_feature_counts, all_contexts,
                                                                self.all_perms[i])

        # s_o = contains the number of all different states(reaqlizations) with the same observation action
        # up to "max_no_red_context" number of permitted observations.
        self.s_o_max_SimOOS = int(np.amax(self.s_o))
        self.Psi_total = int(np.sum(self.psi))

        # Sliding windows and counters
        # Different Tau variables are implemented as binary vectors of len=window, for easy dot products with window.
        self.reward_window = collections.deque(maxlen=self.w)
        self.Tau_aso = []
        for a in range(self.number_of_actions):
            self.Tau_aso.append([])
            for s in range(self.s_o_max_SimOOS):
                self.Tau_aso[-1].append([])
                for o in range(self.number_of_perms_SimOOS):
                    self.Tau_aso[-1][-1].append(collections.deque(maxlen=self.w))

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
        self.rounds = 0

    def find_optimal_policy(self, t, cost_vector):
        self.rounds += 1
        if t % 500 == 0:
            print(f"Round {t}, time {datetime.datetime.now()}")

        # Optimistic cost value c_tilde is same for all observations.
        c_tilde = np.zeros(self.org_dim_context)
        for f in range(self.org_dim_context):
            confidence_interval_cost_f = min(1, math.sqrt(
                math.log((self.org_dim_context * self.w * self.time_horizon) / self.delta) / (
                    2 * self.N_t_f[f]
                )
            ))
            c_tilde[f] = self.c_hat_t[f] + confidence_interval_cost_f

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

            prob_hat = self.d_t_os[i, :z]

            confidence_interval_prob = min(1, math.sqrt(
                math.log((self.Psi_total * self.time_horizon) / self.delta) / 2 * self.N_t_o[i]))

            observation_action_in_optimization = self.all_perms[i]

            # Construct the problem, Equation (3) in the paper
            prob_tilde = cp.Variable(z)

            objective = cp.Maximize(
                (self.beta * (np.array(r_star[i]) * prob_tilde))
                - np.dot(observation_action_in_optimization, c_tilde)
            )

            constraints = [cp.norm((prob_tilde - prob_hat), 1) <= confidence_interval_prob, cp.sum(prob_tilde) == 1]

            prob = cp.Problem(objective, constraints)

            prob.solve()

            # Similar to paper, set V_hat[i] = nu_t[i] as the maximizer
            self.nu_t[i] = prob.value

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
            s_t = self.state_extract(self.feature_values, self.all_feature_counts, context_at_t,
                                         self.observation_action_at_t)
            # Sanity check
            # created_context = self.state_create(s_t, self.feature_values)
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

        return pool_indices.index(self.action_at_t)

    def update(self, t, action_index_at_t, reward_at_t, cost_vector_at_t, context_at_t, pool_indices):

        cost_at_t = np.dot(cost_vector_at_t, self.observation_action_at_t)

        action_at_t = pool_indices[action_index_at_t]

        s_t = self.state_extract(self.feature_values, self.all_feature_counts, context_at_t,
                                     self.observation_action_at_t)
        # Sanity check
        # created_context = self.state_create(s_t, self.feature_values)
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

        for obs in range(self.number_of_perms_SimOOS):
            for state in range(int(self.s_o[obs])):
                for a in range(self.number_of_actions):
                    if t > self.w:
                        self.Tau_aso[a][state][obs].popleft()
                    if a == action_at_t and state == s_t and obs == o_t:
                        self.Tau_aso[a][state][obs].append(1)
                    else:
                        self.Tau_aso[a][state][obs].append(0)

                    tau_aso_array = np.array(self.Tau_aso[a][state][obs])
                    self.N_t_aso[a][state][obs] = np.count_nonzero(tau_aso_array)

                    sum_of_window_rewards = np.dot(np.array(self.reward_window), tau_aso_array)
                    # r_hat_t is the empirical average reward.
                    self.r_hat_t[a][state][obs] = sum_of_window_rewards / self.N_t_aso[a][state][obs]

        # Cost window and counters
        for f in range(self.org_dim_context):
            if t > self.w:
                self.cost_window[f].popleft()
                self.Tau_f[f].popleft()

            self.cost_window[f].append(cost_vector_at_t[f])
            self.Tau_f[f].append(self.all_perms[o_t][f])

            tau_f_array = np.array(self.Tau_f[f])

            self.N_t_f[f] = np.count_nonzero(tau_f_array)
            sum_of_window_costs_one_feature = np.dot(np.array(self.cost_window[f]), tau_f_array)
            self.c_hat_t[f] = sum_of_window_costs_one_feature / self.N_t_f[f]

        # These two counters do not use window, but rather all observations.
        self.N_t_o[o_t] += 1
        self.N_t_os[o_t, s_t] += 1

        self.selected_context_SimOOS[t, :] = self.selected_observation_action_at_t
        if t < self.number_of_perms_SimOOS:
            # This makes sense, since in this for loop, both N_t_os and N_t_o are being updated at every time
            # and each observation is seen only once.
            self.d_t_os[o_t, s_t] = 1

        else:
            # Optimistic Policy Optimization
            self.d_t_os[o_t, :] = self.N_t_os[o_t, :] / self.N_t_o[o_t]

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

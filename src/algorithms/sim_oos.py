import math
import datetime
import numpy as np
import cvxpy as cp

from src.algorithms import utilities


class SimOOSAlgorithm:
    """
    Sim-OOS policy.

    This algorithm was designed for Contextual Multi-Armed Bandit with Costly observations (CMAB-CO) problem.
    Designed for stationary environments (both rewards and costs).

    Works with categorical features, as it is based on enumerating all possible observations.
    Cost vectors are provided to this algorithm.

    From paper:
    "Data-Driven Online Recommender Systems with Costly Information Acquisition"
    Atan et al. 2021
    """

    def __init__(self,
                 all_contexts: np.array,
                 number_of_actions: int,
                 max_num_observations: int,
                 delta: float,
                 beta: float = 1.0,
                 ):
        """Initialize the Sim-OOS algorithm.

        Parameters
        ----------
        all_contexts:
            matrix of contexts, each row corresponds to one context. This algorithm needs to know all possible values
            of each feature to enumerate all possible states. One state corresponds to one partial context vector.
        number_of_actions:
            number of arms of the multi-armed bandit problem.
        max_num_observations:
            maximum number of features NCC-UCRL2 is allowed to observe.
        delta:
            hyperparameter, influences the size of confidence bounds.
        beta:
            scaling factor for rewards, gain = beta*rewards - costs. Used to test the algorithm with larger costs
            so that gain stays positive. Defaults to 1.

        Notes
        ----------
        Observations are binary vectors of length self.context_dimensionality
        This algorithm counts and enumerates all possible partial state vectors for each observation. Counts are saved
        in variable self.s_o, enumeration is implemented by function utilities.state_extract, that returns the state
        s_t for a given context and observation. Together s_t and observation define the state underlying MDP is in.
        """

        self.name = f"SimOOS (beta={beta}, delta={delta})"

        self.time_horizon = all_contexts.shape[0]
        self.context_dimensionality = all_contexts.shape[1]
        self.max_num_observations = max_num_observations
        self.number_of_actions = number_of_actions
        self.beta = beta
        self.delta = delta

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

        ########################################################################################
        # Feature values and counts are used to extract state index s_t for a given observation and partial context.
        self.feature_values, self.all_feature_counts = utilities.save_feature_values(all_contexts)

        for i in range(self.number_of_perms):
            # s_o[i] - size of state array for a given observation action.
            self.s_o[i] = utilities.state_construct(self.all_feature_counts, all_contexts, self.all_perms[i])

        # s_o_max - the largest state vector for all observations, needed to create arrays.
        self.s_o_max = int(np.amax(self.s_o))
        # Total number of possible partial state vectors.
        self.Psi_total = int(np.sum(self.s_o))

        # Define the counters and variables
        # a - action, s - state (partial vector), o - observation (subset of features)
        self.r_hat_t = np.zeros((self.number_of_actions, self.s_o_max, self.number_of_perms))
        self.conf1_t = np.zeros((self.number_of_actions, self.s_o_max, self.number_of_perms))
        self.N_t_aso = np.zeros((self.number_of_actions, self.s_o_max, self.number_of_perms))
        self.N_t_o = np.zeros(self.number_of_perms)
        self.N_t_os = np.zeros((self.number_of_perms, self.s_o_max))
        self.d_t_os = np.zeros((self.number_of_perms, self.s_o_max))  # Definition: N_t_os / N_t_o
        self.N_t_as = np.zeros((self.number_of_actions, self.s_o_max))

        self.N_old_aso = np.zeros((self.number_of_actions, self.s_o_max, self.number_of_perms))

        # Last observed feature subset.
        self.observation_action_at_t = None
        # Last chosen action
        self.action_at_t = None

        # Important Variables
        # Optimistic reward estimates, r_hat + conf_1 for all actions, realizations (states), and permutations.
        self.upsilon_t = np.zeros((self.number_of_actions, self.s_o_max, self.number_of_perms))

        # Optimistic action policy. This is actually the h_hat in the paper
        # For each state and observation we store not the optimal action, but rather all actions sorted
        # by their optimistic reward estimates. This is needed so that at each trial we can choose
        # the best arm from the pool of available arms.
        self.a_hat_t = np.zeros((self.s_o_max, self.number_of_perms, self.number_of_actions))

        # Optimistic value of observation. This is actually the V_hat in the paper.
        self.nu_t = np.zeros(self.number_of_perms)

        self.index_of_observation_action_at_t = 0

        self.selected_observation_action_at_t = np.zeros(self.context_dimensionality)

        # This algorithm updates its policy at the start of round and then pulls arms accordingly for some number
        # of time steps. This variable initialized to 1 so initialize_new_round can run at the beginning.
        self.new_round = 1

    def initialize_new_round(self, t, cost_vector):
        """Solve the optimization problem to find the policy that will be used for the duration of this round."""
        for i in range(self.number_of_perms):

            if i == 0:
                F = [[] for i in range(self.number_of_perms)]  # the r_star values will be stored here

            z = int(self.s_o[i])
            for j in range(z):

                for k in range(self.number_of_actions):

                    if self.N_t_aso[k, j, i] == 0:
                        self.upsilon_t[k, j, i] = 1  # min(1, !) = 1
                        confidence_interval_1 = 0
                    else:
                        confidence_interval_1 = min(1, math.sqrt(
                            math.log((20 * self.Psi_total * self.number_of_actions * (t ** 5)) / self.delta) / (
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
                (10 * self.Psi_total * math.log(4 * t / self.delta)) / self.N_t_o[i]))

            observation_action_in_optimization = self.all_perms[i]

            # Construct the problem, Equation (3) in the paper
            prob_tilde = cp.Variable(z)

            objective = cp.Maximize(
                (self.beta * (np.array(F[i]) @ prob_tilde))
                - np.dot(observation_action_in_optimization, cost_vector)
            )

            constraints = [cp.norm((prob_tilde - prob_hat), 1) <= confidence_interval_2, cp.sum(prob_tilde) == 1]

            prob = cp.Problem(objective, constraints)

            prob.solve()

            # Similar to paper, set V_hat[i] = nu_t[i] as the maximizer
            self.nu_t[i] = prob.value

        self.index_of_observation_action_at_t = np.argmax(
            self.nu_t)  # Find which all_perms[i](= index_of_observation_action_at_t) gives the highest prob_tilde

        self.selected_observation_action_at_t = self.all_perms[self.index_of_observation_action_at_t]

        self.N_old_aso = np.copy(self.N_t_aso)

        self.new_round = 0  # New round initialized, no new round needed.

    def choose_features_to_observe(self, t, feature_indices, cost_vector):
        if t < self.number_of_perms:
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
        if t < self.number_of_perms:
            # Random Source Selection Part
            self.action_at_t = np.random.choice(pool_indices)
        else:
            # Optimistic Policy Optimization
            s_t = utilities.state_extract(self.feature_values, self.all_feature_counts, context_at_t,
                                          self.observation_action_at_t)

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

        return pool_indices.index(self.action_at_t)

    def update(self, t, action_index_at_t, reward_at_t, cost_vector_at_t, context_at_t, pool_indices):

        if t % 1000 == 0:
            print(f"Trial {t}, time {datetime.datetime.now()}")

        cost_at_t = np.dot(cost_vector_at_t, self.observation_action_at_t)

        action_at_t = pool_indices[action_index_at_t]

        s_t = utilities.state_extract(self.feature_values, self.all_feature_counts, context_at_t,
                                      self.observation_action_at_t)

        if t < self.number_of_perms:
            # Random Source Selection Part
            # r_hat_t is basically the empirical average reward.
            self.r_hat_t[action_at_t, s_t, t] = (self.N_t_aso[action_at_t, s_t, t] * self.r_hat_t[
                action_at_t, s_t, t] + reward_at_t) / (
                                                        self.N_t_aso[action_at_t, s_t, t] + 1)

            self.N_t_aso[action_at_t, s_t, t] += 1
            self.N_t_o[t] += 1
            self.N_t_os[t, s_t] += 1
            self.d_t_os[
                t, s_t
            ] = 1  # This makes sense, since in this for loop, both N_t_os and N_t_o are being updated at every time.
            # Moreover, each observation is seen only once.
            self.N_t_as[action_at_t, s_t] += 1

            self.selected_context[t, :] = np.array([c if c is not None else 0 for c in context_at_t])

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

            self.selected_context[t, :] = self.selected_observation_action_at_t

        # This part is common both for random observation selection part and for optimistic policy optimization.
        self.all_gain[t + 1] = self.all_gain[t] + reward_at_t - cost_at_t

        self.selected_action[t] = action_at_t

        self.collected_gains[t] = reward_at_t - cost_at_t
        self.collected_rewards[t] = reward_at_t
        self.collected_costs[t] = cost_at_t

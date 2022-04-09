import numpy as np


class UCB1:
    """
    UCB1 algorithm implementation. Deterministic.

    From paper "Finite-time Analysis of the Multiarmed Bandit Problem" Auer. et. al. 2002
    """

    def __init__(self, n_trials: int, n_arms: int, alpha: float):
        self.alpha = alpha
        self.name = f"UCB1 (α={self.alpha})"

        self.q = np.zeros(n_arms)  # average reward for each arm
        self.n = np.ones(n_arms)  # number of times each arm was chosen

        # self.ucbs = np.zeros((n_trials+1, n_arms))
        # self.qs = np.zeros((n_trials+1, n_arms))
        # self.ns = np.zeros((n_trials+1, n_arms))

    def choose_arm(self, trial, context, pool_indices):
        """
        Returns the best arm's index relative to the pool
        """

        ucbs = self.q[pool_indices] + np.sqrt(
            self.alpha * np.log(trial + 1) / self.n[pool_indices]
        )
        # self.ucbs[trial] = ucbs
        # self.qs[trial] = self.q[pool_indices]
        # self.ns[trial] = self.n[pool_indices]
        return np.argmax(ucbs)

    def update(self, trial, displayed_article_index, reward, cost, context, pool_indices):
        """
        Updates algorithm's parameters: q, n
        """

        chosen_arm_index = pool_indices[displayed_article_index]

        self.n[chosen_arm_index] += 1
        self.q[chosen_arm_index] += (reward - self.q[chosen_arm_index]) / self.n[
            chosen_arm_index
        ]

    def choose_features_to_observe(self, trial, feature_indices, cost_vector):
        # UCB1 has no feature selection so it uses all available features.
        return []

import numpy as np


class UCB1:
    """
    UCB1 algorithm implementation. Deterministic.

    From paper "Finite-time Analysis of the Multiarmed Bandit Problem" Auer. et. al. 2002
    """

    def __init__(self, n_arms: int, alpha: float):
        self.alpha = alpha
        self.name = f"UCB1 (α={self.alpha})"

        self.q = np.zeros(n_arms)  # average reward for each arm
        self.n = np.ones(n_arms)  # number of times each arm was chosen

    def choose_arm(self, trial, context, pool_indexes):
        """
        Returns the best arm's index relative to the pool
        """

        ucbs = self.q[pool_indexes] + np.sqrt(
            self.alpha * np.log(trial + 1) / self.n[pool_indexes]
        )
        return np.argmax(ucbs)

    def update(self, trial, displayed_article_index, reward, context, pool_indexes):
        """
        Updates algorithm's parameters: q, n
        """

        chosen_arm_index = pool_indexes[displayed_article_index]

        self.n[chosen_arm_index] += 1
        self.q[chosen_arm_index] += (reward - self.q[chosen_arm_index]) / self.n[
            chosen_arm_index
        ]

    def choose_features_to_observe(self, trial, feature_indexes):
        # UCB1 has no feature selection so it uses all available features.
        return feature_indexes

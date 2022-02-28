import numpy as np
from typing import List


class EpsilonGreedy:
    def __init__(self, n_arms: int, epsilon: float):
        self.epsilon = epsilon
        self.name = f"E-greedy(epsilon={epsilon})"
        self.q = np.zeros(n_arms)  # estimated reward for each arm
        self.n = np.zeros(n_arms)  # total number of times an arm was drawn

    def choose_arm(self, trial: int, context: List[float], pool_indices: List[int]):
        """
        Returns best arm's index relative to the pool
        """
        p = np.random.rand()
        if p > self.epsilon:
            return np.argmax(self.q[pool_indices])
        else:
            return np.random.randint(low=0, high=len(pool_indices))

    def update(
        self,
        trial: int,
        displayed_article_index: int,
        reward: int,
        context: List[float],
        pool_indices: List[int],
    ):
        """
        Updates algorithm's parameters after seeing reward.

        Parameters
        ----------
        trial :
            number of trial
        displayed_article_index :
            displayed article index relative to the pool
        reward :
            user clicked or not (0 or 1)
        context :
            user features
        pool_indices :
            pool indices for article identification
        """

        a = pool_indices[displayed_article_index]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]

    def choose_features_to_observe(self, trial, feature_indices, cost_vector):
        return feature_indices

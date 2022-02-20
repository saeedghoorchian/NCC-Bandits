import numpy as np


class RandomPolicy:
    def __init__(self):
        self.name = "Random policy"

    def choose_arm(self, trial, context, pool_indices):
        """
        returns best arm's index relative to the pool
        """
        return np.random.randint(low=0, high=len(pool_indices))

    def update(self, trial, displayed_article_index, reward, context, pool_indices):
        pass

    def choose_features_to_observe(self, trial, feature_indices):
        return feature_indices

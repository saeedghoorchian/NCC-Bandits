import numpy as np


class RandomPolicy:
    def __init__(self):
        self.name = "Random policy"

    def choose_arm(self, trial, context, pool_indexes):
        """
        returns best arm's index relative to the pool
        """
        return np.random.randint(low=0, high=len(pool_indexes))

    def update(self, trial, displayed_article_index, reward, context, pool_indexes):
        pass

    def choose_features_to_observe(self, trial, feature_indexes):
        return feature_indexes

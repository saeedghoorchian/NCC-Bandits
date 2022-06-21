import numpy as np


class RandomPolicy:
    def __init__(self):
        self.name = "Random policy"

    def choose_arm(self, trial, context, pool_indices):
        return np.random.randint(low=0, high=len(pool_indices))

    def update(self, trial, displayed_article_index, reward, cost_vector, context, pool_indices):
        pass

    def choose_features_to_observe(self, trial, feature_indices, cost_vector):
        return []

import abc
import numpy as np


class BaseCosts(metaclass=abc.ABCMeta):
    """This class simulates feature costs"""

    @abc.abstractmethod
    def get_cost_of_features(self, observed_indexes: list[int]) -> float:
        pass


class ZeroCosts(BaseCosts):
    def get_cost_of_features(self, observed_indexes: list[int]) -> float:
        return 0.0


class BernoulliCosts(BaseCosts):
    """Class for Bernoulli feature costs"""
    def __init__(self, feature_vector_size, probabilities, cost_values):
        self.feature_vector_size = feature_vector_size
        # Parameter of bernoulli distribution for every feature
        self.probabilities = probabilities
        self.cost_values = cost_values

        assert self.feature_vector_size == len(self.probabilities), (
            "Probabilities must have same size as feature vector"
        )
        assert self.feature_vector_size == len(self.cost_values), (
            "All parameters must have same size"
        )

    def get_cost_of_features(self, observed_indexes: list[int]) -> float:
        total_cost = 0.0
        for ind in observed_indexes:
            p = self.probabilities[ind]
            total_cost += np.random.binomial(1, p) * self.cost_values[ind]

        return total_cost

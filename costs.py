import abc
import numpy as np
from typing import List


class BaseCosts(metaclass=abc.ABCMeta):
    """This class simulates feature costs"""

    @abc.abstractmethod
    def get_one_feature_cost(self, index: int, trial: int):
        pass

    def get_total_cost_of_features(self, observed_indexes: List[int], trial: int) -> float:
        total_cost = 0.0
        for ind in observed_indexes:
            total_cost += self.get_one_feature_cost(ind, trial)

        return total_cost


class ZeroCosts(BaseCosts):
    def get_one_feature_cost(self, index: int, trial: int):
        return 0.0


class BernoulliCosts(BaseCosts):
    """Class for Bernoulli feature costs"""

    def __init__(self,
                 feature_vector_size: int,
                 probabilities: List[float],
                 cost_values: List[float],
                 random_seed: int = None
                 ):
        self.feature_vector_size = feature_vector_size
        # Parameter of bernoulli distribution for every feature
        self.probabilities = probabilities
        self.cost_values = cost_values
        self.rng = np.random.default_rng(random_seed)

        assert self.feature_vector_size == len(self.probabilities), (
            "Probabilities must have same size as feature vector"
        )
        assert self.feature_vector_size == len(self.cost_values), (
            "Cost values must have same size as feature vector"
        )

    def get_one_feature_cost(self, index: int, trial: int):
        p = self.probabilities[index]
        return self.rng.binomial(1, p) * self.cost_values[index]

    def get_separate_costs(self, trial):
        costs = []
        for i in range(self.feature_vector_size):
            costs.append(
                self.get_one_feature_cost(i, trial)
            )
        return costs


class GaussianCosts(BaseCosts):
    def __init__(self,
                 feature_vector_size: int,
                 means: List[float],
                 stds: List[float],
                 random_seed: int = None
                 ):
        self.feature_vector_size = feature_vector_size
        # Parameters of Gaussian distribution for every feature
        self.means = means
        self.stds = stds
        self.rng = np.random.default_rng(random_seed)

        assert self.feature_vector_size == len(self.means), (
            "Means must have same size as feature vector"
        )
        assert self.feature_vector_size == len(self.stds), (
            "Stds must have same size as feature vector"
        )

    def get_one_feature_cost(self, index: int, trial: int):
        return self.rng.normal(loc=self.means[index], scale=self.stds[index])

    def get_separate_costs(self, trial):
        costs = []
        for i in range(self.feature_vector_size):
            costs.append(
                self.get_one_feature_cost(i, trial)
            )
        return costs


class NonstationaryBernoulliCosts(BaseCosts):
    def __init__(self,
                 feature_vector_size: int,
                 change_points: List[int],
                 interval_probabilities: List[List[float]],
                 interval_cost_values: List[List[float]],
                 random_seed: int = None
                 ):
        self.feature_vector_size = feature_vector_size

        assert change_points == sorted(change_points), (
            "Change points must be increasing"
        )
        assert change_points == sorted(list(set(change_points))), (
            "Change points must be unique"
        )
        for cp in change_points:
            assert isinstance(cp, int), f"Change points must be integer, cp {cp} is not"
            assert cp > 0, f"Change points must be positive, cp {cp} is not"
        self.change_points = [0] + change_points

        # Parameters of bernoulli distribution for every feature for each interval

        self.interval_probabilities = interval_probabilities
        self.interval_cost_values = interval_cost_values
        assert len(self.change_points) == len(interval_probabilities) == len(interval_cost_values), (
            "There must be (change_points + 1) number of parameter lists for p and values"
        )
        for probabilities, cost_values in zip(self.interval_probabilities, self.interval_cost_values):
            assert self.feature_vector_size == len(probabilities), (
                "Probabilities must have same size as feature vector"
            )
            assert self.feature_vector_size == len(cost_values), (
                "Cost values must have same size as feature vector"
            )

        self.rng = np.random.default_rng(random_seed)

    def get_one_feature_cost(self, index: int, trial: int):
        # Determine at which interval is the current trial
        interval = 0
        for i, cp in enumerate(self.change_points):
            if trial > cp:
                interval = i

        # Calculate feature cost for this trial
        probabilities = self.interval_probabilities[interval]
        cost_values = self.interval_cost_values[interval]
        p = probabilities[index]
        return self.rng.binomial(1, p) * cost_values[index]

    def get_separate_costs(self, trial):
        costs = []
        for i in range(self.feature_vector_size):
            costs.append(
                self.get_one_feature_cost(i, trial)
            )
        return costs


class NonstationaryGaussianCosts(BaseCosts):
    def __init__(self,
                 feature_vector_size: int,
                 change_points: List[int],
                 interval_means: List[List[float]],
                 interval_stds: List[List[float]],
                 random_seed: int = None
                 ):
        self.feature_vector_size = feature_vector_size

        assert change_points == sorted(change_points), (
            "Change points must be increasing"
        )
        assert change_points == sorted(list(set(change_points))), (
            "Change points must be unique"
        )
        for cp in change_points:
            assert isinstance(cp, int), f"Change points must be integer, cp {cp} is not"
            assert cp > 0, f"Change points must be positive, cp {cp} is not"
        self.change_points = [0] + change_points

        # Parameters of Gaussian distribution for every feature for every interval
        assert len(self.change_points) == len(interval_means) == len(interval_stds), (
            "There must be (change_points + 1) number of parameter lists for p and values"
        )
        self.interval_means = interval_means
        self.interval_stds = interval_stds

        for means, stds in zip(self.interval_means, self.interval_stds):
            assert self.feature_vector_size == len(means), (
                "Means must have same size as feature vector"
            )
            assert self.feature_vector_size == len(stds), (
                "Stds must have same size as feature vector"
            )

        self.rng = np.random.default_rng(random_seed)

    def get_one_feature_cost(self, index: int, trial: int):
        interval = 0
        for i, cp in enumerate(self.change_points):
            if trial > cp:
                interval = i
        means = self.interval_means[interval]
        stds = self.interval_stds[interval]
        
        return self.rng.normal(loc=means[index], scale=stds[index])

    def get_separate_costs(self, trial):
        costs = []
        for i in range(self.feature_vector_size):
            costs.append(
                self.get_one_feature_cost(i, trial)
            )
        return costs


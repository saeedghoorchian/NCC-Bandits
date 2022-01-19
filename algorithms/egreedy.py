import numpy as np


class EpsilonGreedy:

    def __init__(self, epsilon, total_n_arms):
        self.epsilon = epsilon
        self.name = f"E-greedy(epsilon={epsilon})"
        self.q = np.zeros(total_n_arms)  # estimated reward for each arm
        self.n = np.zeros(total_n_arms)  # total number of times an arm was drawn

    def choose_arm(self, trial, context, pool_indexes):
        """
        Returns best arm's index relative to the pool
        """
        p = np.random.rand()
        if p > self.epsilon:
            return np.argmax(self.q[pool_indexes])
        else:
            return np.random.randint(low=0, high=len(pool_indexes))

    def update(self, trial, displayed_article_index, reward, context, pool_indexes):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        trial :
            number of trial
        displayed_article_index :
            displayed article index relative to the pool
        reward :
            user clicked or not
        context :
            user features
        pool_indexes :
            pool indexes for article identification
        """

        a = pool_indexes[displayed_article_index]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]

    def choose_features_to_observe(self, trial, feature_indexes):
        return feature_indexes

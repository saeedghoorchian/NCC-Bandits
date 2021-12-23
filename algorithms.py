import cupy as np


class RandomPolicy:
    def __init__(self):
        self.name = "Random policy"

    def choose_arm(self, trial, context, pool_indexes):
        """
        returns best arm's index relative to the pool
        """
        return np.random.randint(low=0, high=len(pool_indexes))

    def update(self, displayed_article_index, reward, context, pool_indexes):
        pass

    def choose_features_to_observe(self, trial, feature_indexes):
        return feature_indexes


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

    def update(self, displayed_article_index, reward, context, pool_indexes):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
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


class LinUCB:
    def __init__(self, context_dimension: int, n_arms: int, alpha: float):
        self.name = f"LinUCB (alpha={alpha}"
        self.context_dimension = context_dimension
        self.alpha = round(alpha, 1)
        self.n_arms = n_arms

        # Vertical array of size n_arms. Each element is a matrix d x d. One matrix for each arm.
        self.A = np.array([np.identity(context_dimension)] * n_arms)
        # Same, shape is(n_arms, context_dimension, context_dimension)
        self.A_inv = np.array([np.identity(context_dimension)] * n_arms)
        # Vertical array of size n_arms, each element is vector of size d x 1
        self.b = np.zeros((n_arms, context_dimension, 1))

    def choose_arm(self, trial, context, pool_indexes):
        """Return best arm's index relative to the pool.

        This method uses vectorized implementation of LinUCB algorithm.
        Comments should help understand the dimensions.
        """
        # Take only subset of arms relevant to this trial (only some arms are shown at each event).
        n_pool = len(pool_indexes)
        A_inv = self.A_inv[pool_indexes]
        b = self.b[pool_indexes]

        x = np.array([context] * n_pool)  # Broadcast context vector (same for each arm), shape (n_pool, d)
        x = x.reshape(n_pool, self.context_dimension, 1)  # Shape is now (n_pool, d, 1) so for each arm (d,1) vector.

        theta = A_inv @ b  # One theta vector (d x 1) for each arm. (n_pool, d, 1)

        theta_T = np.transpose(theta, axes=(0, 2, 1))  # (n_pool, 1, d)
        estimated_reward = theta_T @ x  # (n_pool, 1, 1)

        x_T = np.transpose(x, axes=(0, 2, 1))  # (n_pool, 1, d)
        upper_confidence_bound = self.alpha * np.sqrt(x_T @ A_inv @ x)  # (n_pool, 1, 1)

        score = estimated_reward + upper_confidence_bound

        return np.argmax(score)

    def update(self, displayed_article_index, reward, context, pool_indexes):
        """Update the parameters of the model after each trial."""
        chosen_arm_index = pool_indexes[displayed_article_index]

        x = context.reshape((self.context_dimension, 1))  # Turn 1-dimensional context vector into (d,1) matrix

        self.A[chosen_arm_index] += x @ x.T
        self.b[chosen_arm_index] += reward * x
        self.A_inv[chosen_arm_index] = np.linalg.inv(self.A[chosen_arm_index])

    def choose_features_to_observe(self, trial, feature_indexes):
        # LinUCB has no feature selection so it uses all available features.
        return feature_indexes

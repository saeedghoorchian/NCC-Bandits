import numpy as np


class LinUCB:
    def __init__(self, context_dimension: int, n_arms: int, alpha: float):
        self.context_dimension = context_dimension
        # alpha parameter controls how large the ucb is, larger alpha means more exploration
        assert alpha > 0.0, "Alpha parameter must be positive"
        self.alpha = alpha
        self.name = f"LinUCB (alpha={self.alpha})"
        self.n_arms = n_arms

        # Vertical array of size n_arms. Each element is a matrix d x d. One matrix for each arm.
        self.A = np.array([np.identity(context_dimension)] * n_arms, dtype=np.float32)
        # Same, shape is(n_arms, context_dimension, context_dimension)
        self.A_inv = np.array(
            [np.identity(context_dimension)] * n_arms, dtype=np.float32
        )
        # Vertical array of size n_arms, each element is vector of size d x 1
        self.b = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)
        # Vertical array of size n_arms, each element is vector of size d x 1
        self.theta = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)

    def choose_arm(self, trial, context, pool_indices):
        """Return best arm's index relative to the pool.

        This method uses vectorized implementation of LinUCB algorithm.
        Comments should help understand the dimensions.
        """
        # Take only subset of arms relevant to this trial (only some arms are shown at each event).
        n_pool = len(pool_indices)
        A_inv = self.A_inv[pool_indices]
        theta = self.theta[pool_indices]

        x = np.array(
            [context] * n_pool
        )  # Broadcast context vector (same for each arm), shape (n_pool, d)
        x = x.reshape(
            n_pool, self.context_dimension, 1
        )  # Shape is now (n_pool, d, 1) so for each arm (d,1) vector.

        theta_T = np.transpose(theta, axes=(0, 2, 1))  # (n_pool, 1, d)
        estimated_reward = theta_T @ x  # (n_pool, 1, 1)

        x_T = np.transpose(x, axes=(0, 2, 1))  # (n_pool, 1, d)
        upper_confidence_bound = self.alpha * np.sqrt(x_T @ A_inv @ x)  # (n_pool, 1, 1)

        score = estimated_reward + upper_confidence_bound

        return np.argmax(score)

    def update(self, trial, displayed_article_index, reward, cost, context, pool_indices):
        """Update the parameters of the model after each trial."""
        chosen_arm_index = pool_indices[displayed_article_index]

        x = context.reshape(
            (self.context_dimension, 1)
        )  # Turn 1-dimensional context vector into (d,1) matrix

        self.A[chosen_arm_index] += x @ x.T
        self.b[chosen_arm_index] += reward * x
        # Precompute inverse of A and theta, as arm update happens less often them arm choosing.
        self.A_inv[chosen_arm_index] = np.linalg.inv(self.A[chosen_arm_index])
        self.theta[chosen_arm_index] = (
            self.A_inv[chosen_arm_index] @ self.b[chosen_arm_index]
        )

    def choose_features_to_observe(self, trial, feature_indices, cost_vector):
        # LinUCB has no feature selection so it uses all available features.
        return feature_indices

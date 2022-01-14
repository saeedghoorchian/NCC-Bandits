import collections
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


class LinUCB:
    def __init__(self, context_dimension: int, n_arms: int, alpha: float):
        self.context_dimension = context_dimension
        # alpha parameter controls how large the ucb is, larger alpha means more exploration
        assert alpha > 0.0, "Alpha parameter must be positive"
        self.alpha = round(alpha, 2)
        self.name = f"LinUCB (alpha={self.alpha}"
        self.n_arms = n_arms

        # Vertical array of size n_arms. Each element is a matrix d x d. One matrix for each arm.
        self.A = np.array([np.identity(context_dimension)] * n_arms, dtype=np.float32)
        # Same, shape is(n_arms, context_dimension, context_dimension)
        self.A_inv = np.array([np.identity(context_dimension)] * n_arms, dtype=np.float32)
        # Vertical array of size n_arms, each element is vector of size d x 1
        self.b = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)

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

    def update(self, trial, displayed_article_index, reward, context, pool_indexes):
        """Update the parameters of the model after each trial."""
        chosen_arm_index = pool_indexes[displayed_article_index]

        x = context.reshape((self.context_dimension, 1))  # Turn 1-dimensional context vector into (d,1) matrix

        self.A[chosen_arm_index] += x @ x.T
        self.b[chosen_arm_index] += reward * x
        self.A_inv[chosen_arm_index] = np.linalg.inv(self.A[chosen_arm_index])

    def choose_features_to_observe(self, trial, feature_indexes):
        # LinUCB has no feature selection so it uses all available features.
        return feature_indexes


class PSLinUCB:
    """Piecewise-Stationary LinUCB Algorithm under the Disjoint Payoff Model

    UCB-like algorithm that considers piecewise-stationarity in rewards.
    From paper "Contextual-Bandit Based Personalized Recommendation with
    Time-Varying User Interests" Xu et al. 2020
    """
    def __init__(self, context_dimension: int, n_arms: int, alpha: float, omega: int, delta: float):
        self.context_dimension = context_dimension
        # alpha parameter controls how large the ucb is, larger alpha means more exploration
        assert alpha > 0.0, "Alpha parameter must be positive"
        self.alpha = round(alpha, 2)
        assert omega > 0, "Omega parameter must be positive"
        self.omega = omega
        assert delta > 0.0, "Delta parameter must be positive"
        self.delta = delta
        self.name = f"PSLinUCB (alpha={self.alpha}, omega={self.omega}, delta={self.delta})"
        self.n_arms = n_arms
        self.context_dimension = context_dimension

        # Vertical arrays of size n_arms. Each element is a matrix d x d. One matrix for each arm.
        self.A_pre = np.array([np.identity(context_dimension)] * n_arms, dtype=np.float32)
        self.A_cur = np.array([np.identity(context_dimension)] * n_arms, dtype=np.float32)
        self.A_cum = np.array([np.identity(context_dimension)] * n_arms, dtype=np.float32)
        # Same, shape is(n_arms, context_dimension, context_dimension)
        self.A_cum_inv = np.array([np.identity(context_dimension)] * n_arms, dtype=np.float32)
        # Vertical arrays of size n_arms, each element is vector of size d x 1
        self.b_pre = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)
        self.b_cur = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)
        self.b_cum = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)

        # Sliding window for each arm.
        self.SW = [collections.deque() for _ in range(n_arms)]

        self.change_points = []

    def choose_arm(self, trial, context, pool_indexes):
        """Return best arm's index relative to the pool.

        This part of algorithm is similar to LinUCB, for each arm all observations since the last
        change point are used to estimate the expected rewards and upper confidence bounds.

        Comments should help understand the dimensions.
        """
        # Take only subset of arms relevant to this trial (only some arms are shown at each event).
        n_pool = len(pool_indexes)
        A_cum_inv = self.A_cum_inv[pool_indexes]
        b_cum = self.b_cum[pool_indexes]

        x = np.array([context] * n_pool)  # Broadcast context vector (same for each arm), shape (n_pool, d)
        x = x.reshape(n_pool, self.context_dimension, 1)  # Shape is now (n_pool, d, 1) so for each arm (d,1) vector.

        theta_cum = A_cum_inv @ b_cum  # One theta vector (d x 1) for each arm. (n_pool, d, 1)

        theta_cum_T = np.transpose(theta_cum, axes=(0, 2, 1))  # (n_pool, 1, d)
        estimated_reward = theta_cum_T @ x  # (n_pool, 1, 1)

        x_T = np.transpose(x, axes=(0, 2, 1))  # (n_pool, 1, d)
        upper_confidence_bound = self.alpha * np.sqrt(x_T @ A_cum_inv @ x)  # (n_pool, 1, 1)

        score = estimated_reward + upper_confidence_bound

        return np.argmax(score)

    def update(self, trial, displayed_article_index, reward, context, pool_indexes):
        """Update the parameters of the model after each trial."""
        chosen_arm_index = pool_indexes[displayed_article_index]

        x = context.reshape((self.context_dimension, 1))  # Turn 1-dimensional context vector into (d,1) matrix

        self.SW[chosen_arm_index].append((x, reward, trial))

        self.A_cur[chosen_arm_index] += x @ x.T
        self.A_cum[chosen_arm_index] += x @ x.T
        self.b_cur[chosen_arm_index] += reward * x
        self.b_cum[chosen_arm_index] += reward * x

        # Change detection and model update
        sliding_window = self.SW[chosen_arm_index]
        if len(sliding_window) >= self.omega:
            A_pre_inv = np.linalg.inv(self.A_pre[chosen_arm_index])  # (d,d)
            theta_pre = A_pre_inv @ self.b_pre[chosen_arm_index]  # (d,1)
            # Broadcast theta_pre to multiply with observations from sliding window
            theta_pre = np.array([theta_pre] * self.omega)  # (omega,d,1)
            window_x = np.array([x for x, r, t in sliding_window][:self.omega])  # (omega,d,1)
            window_x_T = np.transpose(window_x, axes=(0, 2, 1))  # (omega,1,d)
            window_r = np.array([r for x, r, t in sliding_window][:self.omega])  # (omega,)
            estimated_r_pre = (window_x_T @ theta_pre).reshape((self.omega,))
            window_estimation_error = np.abs(np.mean(estimated_r_pre - window_r))
            if window_estimation_error >= self.delta:
                # Can't estimate the reward inside sliding window well by using data before the window.
                # This means change point is detected at the start of sliding window.
                # Restart learning for this arm from the detected change point.

                # Use observations withing the sliding window as a warm start for A_pre and A_cum.
                self.A_pre[chosen_arm_index] = self.A_cur[chosen_arm_index]
                self.A_cum[chosen_arm_index] = self.A_cur[chosen_arm_index]
                self.b_pre[chosen_arm_index] = self.b_cur[chosen_arm_index]
                self.b_cum[chosen_arm_index] = self.b_cur[chosen_arm_index]

                # Empty the sliding window for this arm.
                self.A_cur[chosen_arm_index] = np.identity(self.context_dimension)
                self.b_cur[chosen_arm_index] = np.zeros((self.context_dimension, 1))

                _, _, change_point = self.SW[chosen_arm_index].popleft()
                self.change_points.append(change_point)
                self.SW[chosen_arm_index] = collections.deque()
            else:
                # No change is detected on this arm (previous and current reward observations
                # follow the same model.
                x_1, r_1, _ = self.SW[chosen_arm_index].popleft()
                self.A_pre[chosen_arm_index] += x_1 @ x_1.T
                self.A_cur[chosen_arm_index] -= x_1 @ x_1.T
                self.b_pre[chosen_arm_index] += r_1 * x_1
                self.b_cur[chosen_arm_index] -= r_1 * x_1

        # Precompute the inverse here to reduce the number of matrix inversions.
        self.A_cum_inv[chosen_arm_index] = np.linalg.inv(self.A_cum[chosen_arm_index])

    def choose_features_to_observe(self, trial, feature_indexes):
        # LinUCB has no feature selection so it uses all available features.
        return feature_indexes


class Ucb1:
    def __init__(self, alpha, n_arms):
        self.alpha = round(alpha, 1)
        self.name = "UCB1 (α=" + str(self.alpha) + ")"

        self.q = np.zeros(n_arms)  # average reward for each arm
        self.n = np.ones(n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx):
        ucbs = self.q[pool_idx] + np.sqrt(self.alpha * np.log(t + 1) / self.n[pool_idx])
        return np.argmax(ucbs)

    def update(self, displayed, reward, user, pool_idx):
        a = pool_idx[displayed]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]

    def choose_features_to_observe(self, trial, feature_indexes):
        return feature_indexes

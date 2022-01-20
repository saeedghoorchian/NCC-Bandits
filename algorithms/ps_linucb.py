import collections
import numpy as np


class PSLinUCB:
    """Piecewise-Stationary LinUCB Algorithm under the Disjoint Payoff Model

    UCB-like algorithm that considers piecewise-stationarity in rewards.
    From paper "Contextual-Bandit Based Personalized Recommendation with
    Time-Varying User Interests" Xu et al. 2020
    """

    def __init__(
        self,
        context_dimension: int,
        n_arms: int,
        alpha: float,
        omega: int,
        delta: float,
    ):
        self.context_dimension = context_dimension
        # alpha parameter controls how large the ucb is, larger alpha means more exploration
        assert alpha > 0.0, "Alpha parameter must be positive"
        self.alpha = round(alpha, 2)
        assert omega > 0, "Omega parameter must be positive"
        self.omega = omega
        assert delta > 0.0, "Delta parameter must be positive"
        self.delta = delta
        self.name = (
            f"PSLinUCB (alpha={self.alpha}, omega={self.omega}, delta={self.delta})"
        )
        self.n_arms = n_arms
        self.context_dimension = context_dimension

        # Vertical arrays of size n_arms. Each element is a matrix d x d. One matrix for each arm.
        self.A_pre = np.array(
            [np.identity(context_dimension)] * n_arms, dtype=np.float32
        )
        self.A_cur = np.array(
            [np.identity(context_dimension)] * n_arms, dtype=np.float32
        )
        self.A_cum = np.array(
            [np.identity(context_dimension)] * n_arms, dtype=np.float32
        )
        # Same, shape is(n_arms, context_dimension, context_dimension)
        self.A_cum_inv = np.array(
            [np.identity(context_dimension)] * n_arms, dtype=np.float32
        )
        # Vertical arrays of size n_arms, each element is vector of size d x 1
        self.b_pre = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)
        self.b_cur = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)
        self.b_cum = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)
        self.theta_cum = np.zeros((n_arms, context_dimension, 1), dtype=np.float32)

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
        theta_cum = self.theta_cum[
            pool_indexes
        ]  # One theta vector (d x 1) for each arm. (n_pool, d, 1)

        x = np.array(
            [context] * n_pool
        )  # Broadcast context vector (same for each arm), shape (n_pool, d)
        x = x.reshape(
            n_pool, self.context_dimension, 1
        )  # Shape is now (n_pool, d, 1) so for each arm (d,1) vector.

        theta_cum_T = np.transpose(theta_cum, axes=(0, 2, 1))  # (n_pool, 1, d)
        estimated_reward = theta_cum_T @ x  # (n_pool, 1, 1)

        x_T = np.transpose(x, axes=(0, 2, 1))  # (n_pool, 1, d)
        upper_confidence_bound = self.alpha * np.sqrt(
            x_T @ A_cum_inv @ x
        )  # (n_pool, 1, 1)

        score = estimated_reward + upper_confidence_bound

        return np.argmax(score)

    def update(self, trial, displayed_article_index, reward, context, pool_indexes):
        """Update the parameters of the model after each trial."""
        chosen_arm_index = pool_indexes[displayed_article_index]

        x = context.reshape(
            (self.context_dimension, 1)
        )  # Turn 1-dimensional context vector into (d,1) matrix

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
            window_x = np.array(
                [x for x, r, t in sliding_window][: self.omega]
            )  # (omega,d,1)
            window_x_T = np.transpose(window_x, axes=(0, 2, 1))  # (omega,1,d)
            window_r = np.array(
                [r for x, r, t in sliding_window][: self.omega]
            )  # (omega,)
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

        # Precompute the inverse and theta here to reduce the number of matrix operations.
        self.A_cum_inv[chosen_arm_index] = np.linalg.inv(self.A_cum[chosen_arm_index])
        self.theta_cum[chosen_arm_index] = (
            self.A_cum_inv[chosen_arm_index] @ self.b_cum[chosen_arm_index]
        )

    def choose_features_to_observe(self, trial, feature_indexes):
        # PS-LinUCB has no feature selection so it uses all available features.
        return feature_indexes

import numpy as np


class SW_UCB:
    """
    Sliding Window UCB algorithm implementation. 
    Focuses on recent observations to handle non-stationary environments.

    Window size `tau` determines how many of the latest rewards and arm pulls
    are considered in the algorithm's decision-making process.

    From paper:
    "On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems"
    Garivier & Moulines. 2008
    """
    def __init__(self, n_trials, n_arms: int, alpha: float, tau: int):
        self.alpha = alpha
        self.tau = tau
        self.name = f"SW-UCB (α={self.alpha}, τ={self.tau})"

        # Initialize lists to store reward and timestamp for each arm
        self.rewards = [[] for _ in range(n_arms)]
        self.timestamps = [[] for _ in range(n_arms)]

    def choose_arm(self, trial, context, pool_indices):
        """
        Returns the best arm's index relative to the pool of indices.
        """
        ucbs = []
        for idx in pool_indices:
            if len(self.rewards[idx]) > 0:
                avg_reward = np.mean(self.rewards[idx])
                ucb = avg_reward + np.sqrt(self.alpha * np.log(min(trial + 1, self.tau)) / len(self.rewards[idx]))
            else:
                ucb = float('inf')  # Arm not yet pulled
            ucbs.append(ucb)

        return np.argmax(ucbs)

    def update(self, trial, displayed_article_index, reward, cost, context, pool_indices):
        """
        Updates algorithm's parameters based on the received reward.
        """
        chosen_arm_index = pool_indices[displayed_article_index]

        # Update rewards and timestamps
        self.rewards[chosen_arm_index].append(reward)
        self.timestamps[chosen_arm_index].append(trial)

        # Maintain the window size
        if len(self.rewards[chosen_arm_index]) > self.tau:
            self.rewards[chosen_arm_index].pop(0)
            self.timestamps[chosen_arm_index].pop(0)

    def choose_features_to_observe(self, trial, feature_indices, cost_vector):
        # SW-UCB observes no features.
        return []

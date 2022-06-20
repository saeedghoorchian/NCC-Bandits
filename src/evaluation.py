import time
import numpy as np


def evaluate_on_synthetic_data(
        bandit_algorithm,
        contexts: np.array,
        rewards: np.array,
        costs_vector: np.array,
        beta: int = 1.0,
        stop_after: int = None,
        return_full: bool = False,
) -> tuple:

    """
    Function to evaluate a bandit algorithm using synthetic dataset.

    At each timestamp environment generates a context vector, algorithm chooses which features to observe,
    then observes the subset of the context vector (user features).
    After observing features the algorithm chooses an arm and recieves the reward of the chosen arm.

    Parameters
    ----------
    bandit_algorithm :
        algorithm under evaluation
    contexts:
        matrix with all context vectors for every time t
    rewards:
        matrix with rewards for all arms for every time t
    costs_vector:
        matrix with feature costs for every time t
    beta:
        value by which rewards are multiplied. Only concerns gain calculation, algorithms always get reward 1.
    stop_after:
        number of trials after which to stop evaluation
    return_full:
        if true - return tuple of gain, reward, cost and arms

    Returns
    -------
    cumulative_gain :
        contains cumulative gain (reward minus cost) for each trial
        OR
    tuple(cumulative_gain, cumulative_reward, cumulative_cost, chosen_arms):
        if return_full == True
    """
    assert contexts.shape[0] == rewards.shape[0]
    assert contexts.shape == costs_vector.shape

    num_trials = contexts.shape[0]
    num_features = contexts.shape[1]
    num_arms = rewards.shape[1]

    start = time.time()
    total_reward = 0  # total reward
    total_cost = 0  # total cost
    total_gain = 0  # total gain

    cumulative_reward = []  # contains cumulative reward for each trial
    cumulative_cost = []  # contains cumulative cost for each trial
    cumulative_gain = []  # contains cumulative gain for each trial
    chosen_arms = []  # contains arms chosen in each trial

    for trial in range(num_trials):

        context_at_t = contexts[trial]
        cost_vector_at_t = costs_vector[trial]

        features_to_observe = bandit_algorithm.choose_features_to_observe(
            trial, feature_indices=list(range(num_features)), cost_vector=cost_vector_at_t,
        )
        observed_features = np.array(
            [
                feature if index in features_to_observe else None
                for index, feature in enumerate(context_at_t)
            ]
        )

        pool_indices = list(range(num_arms))
        chosen_arm_index = bandit_algorithm.choose_arm(
            trial, observed_features, pool_indices
        )
        chosen_arm = pool_indices[chosen_arm_index]
        chosen_arms.append(chosen_arm)

        cost_at_t = sum(cost_vector_at_t[features_to_observe])
        total_cost += cost_at_t

        reward_at_t = rewards[trial, chosen_arm]
        # Collected reward is multiplied by beta, but the algorithms see the original reward 0 or 1.
        gain_at_t = beta*reward_at_t - cost_at_t

        total_reward += beta*reward_at_t
        total_gain += gain_at_t

        bandit_algorithm.update(
            trial,
            chosen_arm_index,
            reward_at_t,
            cost_vector_at_t,
            observed_features,
            pool_indices,
        )
        cumulative_cost.append(total_cost)
        cumulative_reward.append(total_reward)
        cumulative_gain.append(total_gain)

        if stop_after is not None and trial >= stop_after-1:
            break

    end = time.time()

    execution_time = round(end - start, 1)
    execution_time = (
        str(round(execution_time / 60, 1)) + "m"
        if execution_time > 60
        else str(execution_time) + "s"
    )
    print(
        f"{bandit_algorithm.name}\n"
        f"Beta = {beta}\n"
        f"Total gain: {total_gain}\n"
        f"\tTotal reward: {total_reward}\n"
        f"\tTotal cost: {total_cost}\n"
        f"Execution time: {execution_time}"
    )

    if return_full:
        return cumulative_gain, cumulative_reward, cumulative_cost, chosen_arms
    else:
        return cumulative_gain


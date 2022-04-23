import time
import numpy as np

import costs
import dataset as dataset_module


def evaluate(
    bandit_algorithm,
    dataset: dataset_module.Dataset,
    feature_costs: costs.BaseCosts = costs.ZeroCosts(),
    stop_after: int = None,
) -> list:
    """
    Function to evaluate a bandit algorithm using user click log data in an offline manner.
    Iterate through logged data. At each timestamp algorithm chooses which features to observe,
    then observes the subset of the context vector (user features) and a subset of arms.
    If the article displayed to user in log data is the same as chosen by the algorithm then this trial
    counts and the algorithm is updated with corresponding reward - cost of observed features.
    If the articles are different then this trial is just skipped.

    This offline evaluation method was proposed in Li et al. 2011
    "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms"

    Parameters
    ----------
    bandit_algorithm :
        algorithm under evaluation
    dataset:
        yahoo data with click events
    feature_costs:
        feature costs
    stop_after:
        Number of trials after which to stop evaluation

    Returns
    -------
    ctr :
        contains the ctr for each trial
    """

    start = time.time()
    total_reward = 0  # cumulative reward
    trial = (
        0  # counter of valid events (when chosen arm was same as displayed in log data)
    )

    ctr = []  # contains ctr for each trial

    events = dataset.events

    for event in events:

        # Next trial is considered because trial is incremented later. So we get costs for trial that will
        # happen if algorithm chooses the same arm as in log data.
        cost_vector = feature_costs.get_full_cost_vector(trial+1, feature_indices=list(range(len(event.user_features))))

        features_to_observe = bandit_algorithm.choose_features_to_observe(
            trial, feature_indices=list(range(len(event.user_features))), cost_vector=cost_vector,
        )
        observed_features = np.array(
            [
                feature if index in features_to_observe else None
                for index, feature in enumerate(event.user_features)
            ]
        )

        chosen = bandit_algorithm.choose_arm(
            trial, observed_features, event.pool_indices
        )
        if chosen == event.displayed_pool_index:
            trial += 1
            cost_at_t = feature_costs.get_total_cost_of_features(
                features_to_observe, trial
            )
            total_reward += event.user_click - cost_at_t
            bandit_algorithm.update(
                trial,
                event.displayed_pool_index,
                event.user_click,
                cost_vector,
                observed_features,
                event.pool_indices,
            )
            ctr.append(total_reward / trial)
            # TODO save event timestamp together with ctr for clearer explanation of results.

        if stop_after is not None and trial >= stop_after:
            break

    end = time.time()

    execution_time = round(end - start, 1)
    execution_time = (
        str(round(execution_time / 60, 1)) + "m"
        if execution_time > 60
        else str(execution_time) + "s"
    )
    print(
        f"{bandit_algorithm.name} with {feature_costs.__class__.__name__}\n"
        f"Average reward: {round(total_reward/trial, 4)}\n"
        f"Execution time: {execution_time}"
    )

    return ctr


def evaluate_on_synthetic_data(
        bandit_algorithm,
        contexts: np.array,
        rewards: np.array,
        costs_vector: np.array,
        stop_after: int = None,
        num_repetitions: int = 1,
        return_full: bool = False,
) -> np.array:
    assert num_repetitions >= 1
    length = stop_after if stop_after is not None else contexts.shape[0]
    gains = np.zeros((num_repetitions, length))
    achieved_rewards = np.zeros((num_repetitions, length))
    paid_costs = np.zeros((num_repetitions, length))
    for i in range(num_repetitions):
        gain, reward, cost = evaluate_on_synthetic_data_once(bandit_algorithm, contexts, rewards, costs_vector, stop_after)
        print()
        gains[i, :] = np.array(gain)
        achieved_rewards[i, :] = np.array(reward)
        paid_costs[i, :] = np.array(cost)

    gains = np.mean(gains, axis=0)
    achieved_rewards = np.mean(achieved_rewards, axis=0)
    paid_costs = np.mean(paid_costs, axis=0)

    if num_repetitions > 1:
        print(f"Result averaged over {num_repetitions} repetitions:")
        print(
            f"{bandit_algorithm.name}\n"
            f"Total gain: {gains[-1]}\n"
            f"\tTotal reward: {achieved_rewards[-1]}\n"
            f"\tTotal cost: {paid_costs[-1]}\n"
        )
    if return_full:
        return gains, achieved_rewards, paid_costs
    return gains


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
    stop_after:
        number of trials after which to stop evaluation

    Returns
    -------
    cumulative_gain :
        contains cumulative gain (reward minus cost) for each trial
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
        return cumulative_gain, cumulative_reward, cumulative_cost
    else:
        return cumulative_gain


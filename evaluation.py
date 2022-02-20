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
        feature coss
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

        features_to_observe = bandit_algorithm.choose_features_to_observe(
            trial, feature_indices=list(range(len(event.user_features)))
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
            total_reward += event.user_click - feature_costs.get_total_cost_of_features(
                features_to_observe, trial
            )
            bandit_algorithm.update(
                trial,
                event.displayed_pool_index,
                event.user_click,
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

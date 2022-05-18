import sys

sys.path.append('..')

import argparse
import json
import pickle
import time
import warnings
import numpy as np

import algorithms
import evaluation

BETA = 1.0
NUM_REPETITIONS = 3
COSTS_RANGE_SIZE = 0.05


def evaluate_algorithm(data_path, trials, parameters):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    contexts, rewards, costs_vector = data

    s = time.time()

    gains = np.zeros((NUM_REPETITIONS, trials))
    rews = np.zeros((NUM_REPETITIONS, trials))
    costs = np.zeros((NUM_REPETITIONS, trials))
    for i in range(NUM_REPETITIONS):

        policy = algorithms.Algorithm1(
            all_contexts=contexts,
            number_of_actions=rewards.shape[1],
            max_no_red_context=contexts.shape[1],
            beta=BETA,
            costs_range=COSTS_RANGE_SIZE,
            **parameters,
        )
        print(f"Creation took {time.time() - s} seconds")


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = time.time()
            gain, rew, cost = evaluation.evaluate_on_synthetic_data(
                policy,
                contexts[:trials],
                rewards[:trials],
                costs_vector[:trials],
                beta=BETA,
                stop_after=trials,
                return_full=True,
            )
            print(f"Took {time.time() - s} seconds")

        gains[i, :] = gain
        rews[i, :] = rew
        costs[i, :] = cost

    gain = np.mean(gains, axis=0)
    rew = np.mean(rews, axis=0)
    cost = np.mean(costs, axis=0)

    return gain, rew, cost


def save_results(results, trials, params):

    gain, rew, cost = results

    params_string = ""
    for k, v in params.items():
        params_string += f"{k}_{v}_"
    params_string = params_string[:-1]

    policy = "alg1"

    with open(f"results/results_{policy}_t_{trials}_{params_string}.pickle", "wb") as f:
        pickle.dump(gain, f)

    with open(f"results/results_{policy}_t_{trials}_{params_string}.txt", "w") as f:
        f.write(f"Algorithm 1 {params_string} gain: {gain[-1]}\n")


def validate_params(params):
    assert "delta" in params
    delta = params["delta"]
    assert isinstance(delta, float)
    assert delta >= 0

    assert "window_length" in params
    window_length = params["window_length"]
    assert isinstance(window_length, int)
    assert window_length > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--trials",
        metavar="TRIALS",
        type=int,
        help="Number of trials",
        required=True,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file for algorithm evaluation",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to pickle file with dataset"
    )

    args = parser.parse_args()

    if args.trials < 0:
        args.trials = None

    with open(args.config, "r") as f:
        experiment_config = json.load(f)

    params = experiment_config["params"]

    validate_params(params)

    results = evaluate_algorithm(args.data, args.trials, params)

    save_results(results, args.trials, params)
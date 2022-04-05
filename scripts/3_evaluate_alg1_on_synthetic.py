import sys

sys.path.append('..')

import argparse
import json
import os
import pickle
import time
import warnings

import algorithms
import evaluation

DATA_PATH = f"../dataset/synthetic/synthetic_data_costs_{os.getenv('SLURM_ARRAY_TASK_ID')}.pickle"


def evaluate_algorithm(data_path, trials, parameters):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    contexts, rewards, costs_vector = data

    s = time.time()
    policy = algorithms.Algorithm1(
        all_contexts=contexts,
        number_of_actions=rewards.shape[1],
        max_no_red_context=contexts.shape[1],
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
            stop_after=trials,
            return_full=True,
        )
        print(f"Took {time.time() - s} seconds")

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

    with open(f"results/results_{policy}_t_{trials}_{params_string}_perf.txt", "w") as f:
        f.write(f"Algorithm 1 {params_string} gain {gain[-1]} performance: {rew[-1]/ cost[-1]}\n")


def validate_params(params):
    assert "beta" in params
    beta = params["beta"]
    assert isinstance(beta, float)
    assert beta >= 0.0

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

    args = parser.parse_args()

    if args.trials < 0:
        args.trials = None

    with open(args.config, "r") as f:
        experiment_config = json.load(f)

    params = experiment_config["params"]

    validate_params(params)

    results = evaluate_algorithm(DATA_PATH, args.trials, params)

    save_results(results, args.trials, params)
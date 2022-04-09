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
import utilities

DATA_PATH = "../dataset/R6/subsample/data_04_medium.pickle"
# Parameter of discretization
NUM_BINS = 4


def preprocess_data(raw_data):
    """Preprocess R6A data by discretizing user feature values"""
    X = np.stack(
        [
            ev.user_features for ev in raw_data.events
        ]
    )
    X_disc = np.ones(X.shape, dtype=np.single)

    for column in range(6):
        X_disc[:, column] = utilities.discretize(X[:, column], NUM_BINS)

    for event, x in zip(raw_data.events, X_disc):
        event.user_features = x

    return raw_data, X_disc


def evaluate_algorithm(data_path, trials, parameters):
    with open(data_path, "rb") as f:
        raw_data = pickle.load(f)

    data, contexts = preprocess_data(raw_data)

    s = time.time()
    policy = algorithms.Algorithm1(
        all_contexts=contexts,
        number_of_actions=data.n_arms,
        max_no_red_context=len(data.events[0].user_features),
        **parameters,
    )
    print(f"Creation took {time.time() - s} seconds")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = time.time()
        ctr = evaluation.evaluate(
            policy,
            data,
            stop_after=trials,
        )
        total_time = time.time() - s
        print(f"Took {total_time} seconds")

    return ctr, total_time


def save_results(results, trials, params):

    ctr, total_time = results

    params_string = ""
    for k, v in params.items():
        params_string += f"{k}_{v}_"
    params_string = params_string[:-1]

    policy = "alg1"

    with open(f"results/results_r6a_{policy}_t_{trials}_{params_string}.pickle", "wb") as f:
        pickle.dump(ctr, f)

    with open(f"results/results_r6a_{policy}_t_{trials}_{params_string}.txt", "w") as f:
        f.write(f"Algorithm 1 {params_string}, time - {total_time}s, ctr: {ctr[-1]}\n")


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
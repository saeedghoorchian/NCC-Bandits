import sys
sys.path.append('..')

import argparse
import json
import pickle
import time
import warnings

import algorithms
import evaluation

DATA_PATH = "../dataset/synthetic/context_dependent.pickle"


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
        gain = evaluation.evaluate_on_synthetic_data(
            policy,
            contexts[:trials],
            rewards[:trials],
            costs_vector[:trials],
            stop_after=trials,
        )
        print(f"Took {time.time() - s} seconds")

    return gain


def save_results(gain, trials, params):

    params_string = ""
    for k, v in params.items():
        params_string += f"{k}_{v}_"
    params_string = params_string[:-1]

    policy = "alg1"

    with open(f"results/results_{policy}_t_{trials}_{params_string}.pickle", "wb") as f:
        pickle.dump(gain, f)

    with open(f"results/results_{policy}_t_{trials}_{params_string}.txt", "w") as f:
        f.write(f"Algorithm 1 {params_string} gain: {gain[-1]}\n")


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
        "--beta",
        type=float,
        required=True,
        help="Beta parameter for algorithm",
    )

    parser.add_argument(
        "--delta",
        type=float,
        required=True,
        help="Delta parameter for algorithm",
    )

    parser.add_argument(
        "--window",
        type=int,
        required=True,
        help="Window length parameter for algorithm",
    )
    args = parser.parse_args()

    if args.trials < 0:
        args.trials = None

    params = {
        "beta": args.beta,
        "delta":  args.delta,
        "window_length": args.window,
    }

    gain = evaluate_algorithm(DATA_PATH, args.trials, params)

    save_results(gain, args.trials, params)
import sys

sys.path.append("..")

import argparse
import json
import pickle

from evaluation_runner import run_evaluation

DATA_PATH = "../dataset/r6b/subsample/data_large.pickle"


def save_results(evaluation_results, config_file, t):
    config_name = config_file.split("/")[-1].split(".")[0]
    with open(f"results/results_{config_name}_t_{t}.pickle", "wb") as f:
        pickle.dump(evaluation_results, f)

    with open(f"results/results_{config_name}_t_{t}.json", "w") as f:
        json.dump(evaluation_results[0], f)


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

    results = run_evaluation(args.trials, DATA_PATH, args.config)

    save_results(results, args.config, args.trials)

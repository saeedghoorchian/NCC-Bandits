import sys
sys.path.append('..')

import gc
import json
import pickle

import algorithms


def get_data(data_file):
    with open(data_file, "rb") as f:
        gc.disable()
        data = pickle.load(f)
        gc.enable()

    return data


def get_bandit_object(class_name, parameters, n_arms, context_dimension):
    parameters["n_arms"] = n_arms
    contextual_bandits = {"LinUCB", "PSLinUCB"}
    if class_name in contextual_bandits:
        parameters["context_dimension"] = context_dimension
    bandit_class = getattr(algorithms, class_name)
    bandit = bandit_class(**parameters)
    return bandit


def run_evaluation(trials, data_file, config_file):
    with open(config_file, "r") as f:
        experiment_config = json.load(f)

    data = get_data(data_file)

    for experiment in experiment_config:
        bandit_name = experiment["algo"]

        params_list = experiment.get("params", [])
        if not params_list:
            params_list = [{}]
        for params in params_list:

            bandit_alg = get_bandit_object(bandit_name, params, data.n_arms, len(data.events[0].user_features))

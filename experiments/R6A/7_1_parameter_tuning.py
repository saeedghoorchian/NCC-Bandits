# This is pure python version of parameter tuning experiment.
from contextlib import contextmanager
import gc
import multiprocessing as mp
import pickle
import time
import sys

sys.path.append("../../")

import algorithms, evaluation

NUM_OF_TRIALS = None


@contextmanager
def timer(name):
    begin = time.time()
    try:
        yield
    finally:
        execution_time = round(time.time() - begin, 1)
        execution_time = (
            str(round(execution_time / 60, 1)) + "m"
            if execution_time > 60
            else str(execution_time) + "s"
        )
        print(f"{name} took {execution_time}")


def run_ps_linucb(params):
    data, alpha, delta, omega = params
    with timer(str(params)):
        ps_linucb = algorithms.PSLinUCB(
            context_dimension=len(data.events[0].user_features),
            n_arms=data.n_arms,
            alpha=alpha,
            omega=omega,
            delta=delta,
        )
        ctr_pslinucb = evaluation.evaluate(ps_linucb, data, stop_after=NUM_OF_TRIALS)
        param_str = f"PS-LinUCB a={alpha} o={omega} d={delta}"

    return param_str, ctr_pslinucb


if __name__ == "__main__":
    with open("../../dataset/R6/subsample/data_05.pickle", "rb") as f:
        gc.disable()
        data = pickle.load(f)
        gc.enable()
    alphas = [0.3, 0.35]
    omegas = [4000, 4500, 5000, 5500, 6000]
    deltas = [0.035, 0.04]

    params = []
    for alpha in alphas:
        for omega in omegas:
            for delta in deltas:
                params.append((data, alpha, delta, omega))

    with timer("Total pool map"):
        pool = mp.Pool()
        results = pool.map(run_ps_linucb, params)
    with open("tuning_results.pickle", "wb") as f:
        pickle.dump(results, f)

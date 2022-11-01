# Online Learning with Costly Features in Non-stationary Environments
This repository contains source code for the paper "Online Learning with Costly Features in Non-stationary Environments".
The source code includes:
* Implementation of the [NCC-UCRL2](https://github.com/SaeedGhoorchian/costly_nonstationary_bandits/blob/main/src/algorithms/alg1.py) algorithm presented in the paper.
* Benchmark bandit algorithms: [Epsilon-greedy](https://github.com/SaeedGhoorchian/costly_nonstationary_bandits/blob/main/src/algorithms/egreedy.py),
[UCB1](https://github.com/SaeedGhoorchian/costly_nonstationary_bandits/blob/main/src/algorithms/ucb.py),
[LinUCB](https://github.com/SaeedGhoorchian/costly_nonstationary_bandits/blob/main/src/algorithms/linucb.py), 
[PS-LinUCB](https://github.com/SaeedGhoorchian/costly_nonstationary_bandits/blob/main/src/algorithms/ps_linucb.py),
[Sim-OOS](https://github.com/SaeedGhoorchian/costly_nonstationary_bandits/blob/main/src/algorithms/sim_oos.py).
* [Experiments](https://github.com/SaeedGhoorchian/costly_nonstationary_bandits/blob/main/experiments/nursery) 
on the UCI Nursery dataset (https://archive.ics.uci.edu/ml/datasets/nursery).

You can use this repository to reproduce our results or try other algorithms, datasets or settings.
 
## Getting started

#### Clone this repository:
```
git clone https://github.com/SaeedGhoorchian/costly_nonstationary_bandits
cd costly_nonstationary_bandits 
```

#### Setup the environment:

The [environment.yml](https://github.com/SaeedGhoorchian/costly_nonstationary_bandits/blob/main/environment.yml)
file describes the list of library dependencies. For a fast and easy way to setup the environment we encourage you to 
use [Conda](https://docs.conda.io/en/latest/miniconda.html). Use the following commands to install the dependencies in
a new conda environment and activate it:
```
conda env create -f environment.yml
conda activate costly_nonstationary_bandits
```

#### Reproduce the experiments:

Run the notebook
[reproducing/reproducing_nursery.ipynb](https://github.com/SaeedGhoorchian/costly_nonstationary_bandits/blob/main/reproducing/reproducing_nursery.ipynb)
to reproduce the experiments and figures from the paper.

This notebook should serve as an entry point for understanding the data and the code used in the paper.
It consists of three parts:
* Preparing the dataset. This includes preprocessing and introducing costs and non-stationarity.
* Evaluating all the bandit algorithms in a simulated environment using the prepared data.
* Plotting the figures using the evaluation data.

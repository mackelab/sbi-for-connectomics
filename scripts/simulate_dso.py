import pickle
from pathlib import Path

import torch
from sbi.inference import prepare_for_sbi, simulate_for_sbi
from torch.distributions import MultivariateNormal

from consbi.simulators import RuleSimulator, default_rule

# set parameters
BASE_DIR = Path(__file__).resolve().parent.parent.as_posix()
path_to_model = BASE_DIR + "/data/structural_model"
save_folder = BASE_DIR + "/data"
save_data = True
verbose = True
# set number of neuron pairs sampled from the connectome to mimick experimental settings, e.g., 50
num_subsampling_pairs = 50
num_simulations = 10  # 1_000_000
num_workers = 1  # 20

prior = MultivariateNormal(torch.ones(3), torch.eye(3))
simulator = RuleSimulator(
    path_to_model,
    default_rule,
    verbose=verbose,
    num_subsampling_pairs=num_subsampling_pairs,
)

batch_simulator, prior = prepare_for_sbi(simulator, prior)

theta, x = simulate_for_sbi(
    batch_simulator, prior, num_simulations=num_simulations, num_workers=num_workers
)

if save_data:
    with open(save_folder + f"/presimulated_dso_rule_n{num_simulations}.p", "wb") as fh:
        pickle.dump(dict(prior=prior, theta=theta, x=x), fh)

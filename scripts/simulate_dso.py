import pickle
from pathlib import Path

import torch
from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform

from consbi.simulators import (
    RuleSimulator,
    default_rule,
    default_rule_constrained_two_param,
    default_rule_linear,
    default_rule_linear_constrained_2p,
    dso_linear_two_param,
    one_param_rule_linear_constrained,
    two_param_rule_dependent,
)
from consbi.simulators.utils import get_batch_simulator

# set parameters
BASE_DIR = Path(__file__).resolve().parent.parent.as_posix()
path_to_model = BASE_DIR + "/data/structural_model"
save_folder = BASE_DIR + "/data"
save_data = True
verbose = True
# set number of neuron pairs sampled from the connectome to mimick experimental settings, e.g., 50
num_subsampling_pairs = 50
num_simulations = 1_000_000
batch_size = 1000
num_workers = 80
# prior_upper_bound = 3
prior_scale = 0.03

# NOTE: make sure to set number of parameters here.
num_dim = 3
rule = default_rule

rule_str = "dso"
prior_str = f"gaussian003"

# Set up prior.
# prior = BoxUniform(torch.zeros(num_dim), 3.0 * torch.ones(num_dim))
from torch.distributions import MultivariateNormal, Uniform

# from sbi.utils import process_prior
# prior, *_ = process_prior([
#     Uniform(0.4 * torch.ones(num_dim), 0.6 * torch.ones(num_dim)),
#     Uniform(1.6 * torch.ones(num_dim), 2.0 * torch.ones(num_dim))])
prior = MultivariateNormal(
    torch.ones(num_dim), prior_scale * torch.eye(num_dim)
)

# Set up simulator.
simulator = RuleSimulator(
    path_to_model,
    rule,
    verbose=verbose,
    num_subsampling_pairs=num_subsampling_pairs,
    prelocate_postall_offset=True,
)
batch_simulator = get_batch_simulator(simulator)

# Simulate and save.
theta, x = simulate_for_sbi(
    batch_simulator,
    prior,
    num_simulations=num_simulations,
    num_workers=num_workers,
    simulation_batch_size=batch_size,
)

if save_data:
    with open(
        save_folder
        + f"/presimulated_{rule_str}_{prior_str}_n{num_simulations}.p",
        "wb",
    ) as fh:
        pickle.dump(dict(prior=prior, theta=theta, x=x), fh)

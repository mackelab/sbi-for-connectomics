import pickle
from functools import partial
from pathlib import Path

import torch
from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform
from torch.distributions import MultivariateNormal

from consbi.simulators import (
    RuleSimulator, 
    default_rule, 
    default_rule_linear, 
    dso_linear_two_param, 
    two_param_rule_dependent,
    default_rule_constrained_two_param,
    default_rule_linear_constrained_2p,
    one_param_rule_linear_constrained,
    
)

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
num_workers = 90
num_dim = 3
# prior_upper_bound = 3
prior_scale = 0.05

rule = default_rule

rule_str = "dso"
prior_str = f"uniform_06_16"

prior = BoxUniform(0.6 * torch.ones(num_dim), 1.6 * torch.ones(num_dim))
# from torch.distributions import Uniform
# from sbi.utils import process_prior
# prior = process_prior([
#     Uniform(0.6 * torch.ones(1), 0.8 * torch.ones(1)), 
#     Uniform(1.2 * torch.ones(1), 1.6 * torch.ones(1))])
# prior = MultivariateNormal(torch.ones(num_dim), prior_scale * torch.eye(num_dim))

simulator = RuleSimulator(
    path_to_model,
    rule,
    verbose=verbose,
    num_subsampling_pairs=num_subsampling_pairs,
    prelocate_postall_offset=True,
)

def batch_simulator(theta):
        """Return a batch of simulations by looping over a batch of parameters."""
        assert theta.ndim > 1, "Theta must have a batch dimension."
        # Simulate in loop
        xs = list(map(simulator, theta))
        # Stack over batch to keep x_shape
        return torch.stack(xs)

theta, x = simulate_for_sbi(
    batch_simulator, prior, num_simulations=num_simulations, num_workers=num_workers, 
    simulation_batch_size=batch_size,
)

if save_data:
    with open(save_folder + f"/presimulated_{rule_str}_{prior_str}_n{num_simulations}.p", "wb") as fh:
        pickle.dump(dict(prior=prior, theta=theta, x=x), fh)

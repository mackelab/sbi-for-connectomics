import pickle
import torch

from consbi.simulators import RuleSimulator, default_rule
from torch.distributions import MultivariateNormal

from sbi.inference import prepare_for_sbi, simulate_for_sbi

# set parameters
path_to_model = "../data/structural_model"
# set number of neuron pairs sampled from the connectome to mimick experimental settings, e.g., 50
num_subsampling_pairs = 50
num_simulations = 10
num_workers = 5
save_data = True


prior = MultivariateNormal(torch.ones(3), torch.eye(3))
simulator = RuleSimulator(path_to_model, default_rule, verbose=False, num_subsampling_pairs=num_subsampling_pairs)

batch_simulator, prior = prepare_for_sbi(simulator, prior)

theta, x = simulate_for_sbi(batch_simulator, prior, num_simulations=num_simulations, num_workers=num_workers)

if save_data:
    with open(f"../data/presimulated_dso_rule_n{num_simulations}.p", "wb") as fh:
        pickle.dump(dict(prior=prior, theta=theta, x=x), fh)

print(x)
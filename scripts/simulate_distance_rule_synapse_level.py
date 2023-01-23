## Script for simulating data according to the distance based rule at the synapse level. 
# See main paper and tutorials/1_simulation for details.

import pickle
from pathlib import Path

import torch
from sbi.inference import prepare_for_sbi, simulate_for_sbi

from consbi.simulators import RuleSimulator, peters_rule_subcellular

# set parameters
BASE_DIR = Path(__file__).resolve().parent.parent.as_posix()
path_to_model = BASE_DIR + "/data/structural_model"
path_to_cellular_features = BASE_DIR + "/data/cube_model"
path_to_subcellular_features = BASE_DIR + "/data/subcellular_features"
save_folder = BASE_DIR + "/data"

# set number of neuron pairs sampled from the connectome to mimick experimental settings, e.g., 50
num_subsampling_pairs = 50
cube_size = 1
num_simulations = 10
num_workers = 5
save_data = False

# Select feature sets:
#     "set-7"  # holds subcellular level features: 1 if two neurons meet in a voxel, 0 else., i x j x k entries, used for the synapse-level rule.
#     "set-6"  # Set 6 holds the common cubes for different cube sizes. i x j entries, used for the neuron-level rule.
#     "set-5"  # Set 5 holds common cubes, min distance and axon-dendrite product for cube size 50mu-m

# for the synapse level rule we use a Beta prior because the parameter is a Bernoulli probability.
prior = torch.distributions.Beta(
    concentration1=torch.ones(1) * 0.5, concentration0=torch.ones(1) * 0.5
)

simulator = RuleSimulator(
    path_to_subcellular_features,
    peters_rule_subcellular,
    verbose=False,
    num_subsampling_pairs=num_subsampling_pairs,
    experiment_name="peters-subcellular",
)
batch_simulator, prior = prepare_for_sbi(simulator, prior)


theta, x = simulate_for_sbi(
    batch_simulator, prior, num_simulations=num_simulations, num_workers=num_workers
)

if save_data:
    with open(
        save_folder + f"/presimulated_distance_rule_synapse_level_n{num_simulations}.p", "wb"
    ) as fh:
        pickle.dump(dict(prior=prior, theta=theta, x=x), fh)

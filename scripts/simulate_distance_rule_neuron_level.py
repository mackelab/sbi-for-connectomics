import pickle
from pathlib import Path

import torch
from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform

from consbi.simulators import DistanceRuleSimulator

# set parameters
BASE_DIR = Path(__file__).resolve().parent.parent.as_posix()
path_to_model = BASE_DIR + "/data/structural_model"
path_to_cellular_features = BASE_DIR + "/data/cube_model"
path_to_subcellular_features = BASE_DIR + "/data/subcellular_features"
save_folder = BASE_DIR + "/data"

# set number of neuron pairs sampled from the connectome to mimick experimental settings, e.g., 50
num_subsampling_pairs = 50
cube_size = 50
num_simulations = 10
num_workers = 5
save_data = True

# Select feature sets:
#     "set-7"  # holds subcellular level features: 1 if two neurons meet in a voxel, 0 else., ixjxk entries
#     "set-6"  # Set 6 holds the common cubes for different cube sizes. ixj entries
#     "set-5"  # Set 5 holds common cubes, min distance and axon-dendrite product for cube size 50mu-m

prior = BoxUniform(torch.zeros(1), torch.ones(1) * 100)
model = DistanceRuleSimulator(
    path_to_model,
    path_to_cellular_features,
    feature_set_name="set-6",
    num_subsampling_pairs=num_subsampling_pairs,
    cube_size=cube_size,
)


def batch_simulator(theta):
    return model.rule(
        theta,
        feature=model.common_cubes,  #  Corresponds to feature set 6.
        connection_fun=model.cutoff_rule,  # cutoff rule forms a connection when feature crosses a threshold.
    )


theta, x = simulate_for_sbi(
    batch_simulator, prior, num_simulations=num_simulations, num_workers=num_workers
)

if save_data:
    with open(
        save_folder + f"/presimulated_distance_rule_n{num_simulations}.p", "wb"
    ) as fh:
        pickle.dump(dict(prior=prior, theta=theta, x=x), fh)

import pickle
from pathlib import Path

import torch
from sbi.inference import SNPE, simulate_for_sbi

from consbi.simulators import (
    RuleSimulator,
    default_rule,
)
from consbi.simulators.utils import get_batch_simulator, seed_all_backends

seed = 42
seed_all_backends(seed)

# set parameters
BASE_DIR = Path(__file__).resolve().parent.parent.as_posix()
path_to_model = BASE_DIR + "/data/structural_model"
load_folder = BASE_DIR + "/data"
save_folder = BASE_DIR + "/results"
save_data = True
verbose = True

# collect and concatenate training data
filenames = [
    # "/presimulated_dso_constrained_2p_uniform0.6_n500000.p",
    "/presimulated_dso_gaussian005_n1000000.p",
]

# in case we have multiple files we concatenate the data and parameters.
x = []
theta = []
for filename in filenames:
    with open(load_folder + filename, "rb") as fh:
        prior, th, xs = pickle.load(fh).values()
        x.append(xs.squeeze())
        theta.append(th)

x = torch.cat(x)
theta = torch.cat(theta)

# exlude simulation prefix from filename.
save_name = f"/npe_{filenames[0][filenames[0].index('_'):]}"

# hyper parameters
training_batch_size = 500
validation_fraction = 0.1
stop_after_epochs = 20
de = "nsf"

# training
trainer = SNPE(
    prior=prior,
    show_progress_bars=verbose,
    density_estimator=de,
)
density_estimator = trainer.append_simulations(theta, x).train(
    # max_num_epochs=1,
    training_batch_size=training_batch_size,
    validation_fraction=validation_fraction,
    stop_after_epochs=stop_after_epochs,
)

# posterior predictive
posterior = trainer.build_posterior(density_estimator)
num_predictive_samples = 10000
xo = torch.tensor([[0.4300, 0.4300, 0.4200, 0.6400, 0.1700, 0.4400, 0.0900]])

simulator = RuleSimulator(
    path_to_model,
    default_rule,
    verbose=verbose,
    num_subsampling_pairs=50,
    prelocate_postall_offset=True,  # only for two-param variants.
)
batch_simulator = get_batch_simulator(simulator)

# Simulate and save.
thos, xos = simulate_for_sbi(
    batch_simulator,
    posterior.set_default_x(xo),
    num_simulations=num_predictive_samples,
    num_workers=50,
    simulation_batch_size=200,
)

with open(save_folder + save_name, "wb") as fh:
    pickle.dump(
        dict(
            prior=prior,
            density_estimator=density_estimator,
            posterior=posterior,
            thos=thos,
            xos=xos,
            kwargs=dict(
                training_batch_size=training_batch_size,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
            ),
            seed=seed,
        ),
        fh,
    )

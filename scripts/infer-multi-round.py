import pickle
from pathlib import Path

import numpy as np
import torch

from sbi.inference import SNPE, simulate_for_sbi

from consbi.simulators.utils import get_batch_simulator, seed_all_backends

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

seed = 42
seed_all_backends(seed)

# set parameters
BASE_DIR = Path(__file__).resolve().parent.parent.as_posix()
path_to_model = BASE_DIR + "/data/structural_model"
load_folder = BASE_DIR + "/data"
save_folder = BASE_DIR + "/results"
save_data = True
verbose = True

xo = np.array([[0.4300, 0.4300, 0.4200, 0.6400, 0.1700, 0.4400, 0.0900]])

rule = default_rule
num_dim = 3
batch_size = 100

## Set up simulator
simulator = RuleSimulator(
    path_to_model,
    rule,
    verbose=True,
    num_subsampling_pairs=50,
    prelocate_postall_offset=True,
)
# wrap simulator to handle batches of parameters.
batch_simulator = get_batch_simulator(simulator)


## collect and concatenate training data
filenames = [
    # "/presimulated_dso_linear_uniform_n120000.p",
    # "/presimulated_dso_linear_uniform_n100000.p",
    # "/presimulated_dso_two_param_uniform_n200000.p"
    # "/presimulated_default_rule_constrained_two_param_uniform_n200000.p",
    # "/presimulated_dso_linear_constrained_two_param_uniform_n100000.p",
    # "/presimulated_dso_gaussian_05_n500000.p",
    # "/presimulated_dso_constrained_2p_gaussian_05_n500000.p"
    "/presimulated_dso_uniform0-3_n500000.p",
    # "/presimulated_dso_linear_constrained_2p_uniform1_n500000.p",
    # "/presimulated_dso_constrained_2p_uniform0.2-0.4_n100000.p",
    # "/presimulated_dso_constrained_2p_uniform0.6_n500000.p",
#     "/presimulated_dso_uniform_06_16_n1000000.p",
]
save_str = filenames[0][filenames[0].index("_"):-2]

x = []
theta = []
for filename in filenames:    
    with open(load_folder + filename, "rb") as fh: 
        prior, th, xs = pickle.load(fh).values()
        x.append(xs.squeeze())
        theta.append(th)

x = torch.cat(x)
theta = torch.cat(theta)


# hyper parameters
training_batch_size = 10000
validation_fraction = 0.1
stop_after_epochs = 20
de = "nsf"
num_rounds = 2
num_simulations_per_round = 100_000
num_workers = 24
save_name = f"/npe{save_str}r{num_rounds}x{int(num_simulations_per_round/1000)}k.p"

# training
trainer = SNPE(
    prior=prior, 
    show_progress_bars=verbose, 
    density_estimator=de,
)
# train first round with presimulated data.
density_estimator = trainer.append_simulations(theta, x).train(
#     max_num_epochs=1, 
    training_batch_size=training_batch_size,
    validation_fraction=validation_fraction, 
    stop_after_epochs=stop_after_epochs,
)
posteriors = [trainer.build_posterior(density_estimator).set_default_x(xo)]


# continue inference over multiple rounds.
for r_idx in range(num_rounds - 1):
    proposal = posteriors[r_idx]
    
    theta, x = simulate_for_sbi(
        batch_simulator, 
        proposal, 
        num_simulations=num_simulations_per_round, 
        num_workers=num_workers, 
        simulation_batch_size=batch_size,
    )

    density_estimator = trainer.append_simulations(theta, x, proposal=proposal).train(
#             max_num_epochs=1, 
            training_batch_size=training_batch_size,
            validation_fraction=validation_fraction, 
            stop_after_epochs=stop_after_epochs,
    )
    
    posteriors.append(trainer.build_posterior(density_estimator).set_default_x(xo))
    

# Simulate posterior predictive samples.
thos, xos = simulate_for_sbi(
        batch_simulator, 
        posteriors[-1], 
        num_simulations=10000, 
        num_workers=num_workers, 
       simulation_batch_size=batch_size,
    )

with open(save_folder + save_name, "wb") as fh:
    pickle.dump(dict(
            prior=prior, 
            density_estimator=density_estimator, 
            posteriors=posteriors,
            thos=thos, 
            xos=xos,
            kwargs=dict(
                training_batch_size=training_batch_size,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
                num_simulations_per_round=num_simulations_per_round,
                ),
            seed=seed,
        ), 
        fh,
    )

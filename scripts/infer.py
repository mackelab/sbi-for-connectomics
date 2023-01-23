import pickle
from pathlib import Path

import torch
from sbi.inference import SNPE

from consbi.simulators.utils import seed_all_backends

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
    "/presimulated_dso_uniform_06_16_n1000000.p",
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
training_batch_size = 10000
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

with open(save_folder + save_name, "wb") as fh:
    pickle.dump(
        dict(
            prior=prior,
            density_estimator=density_estimator,
            kwargs=dict(
                training_batch_size=training_batch_size,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
            ),
            seed=seed,
        ),
        fh,
    )

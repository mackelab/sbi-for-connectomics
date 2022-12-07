import pickle
from os.path import join
from pathlib import Path

import pyro
import torch
from pyro import distributions as pdist

with open(
    join(
        Path(__file__).parent,
        "files/features_selection_D2_highestOverlap_seed10_dim10.npz",
    ),
    "rb",
) as fh:
    design_matrix = torch.as_tensor(pickle.load(fh)["features"], dtype=torch.float32)

log_features = design_matrix
features = torch.exp(design_matrix)

prior_params = dict(low=0.01 * torch.ones(3), high=2.0 * torch.ones(3))


def poisson_glm(parameters, upper_rate_bound: int = 100_000):

    assert parameters.ndim == 2
    assert parameters.shape[1] == 3

    # Set theta3 negative to divide by it in the rule.
    parameters[:, 2] = -parameters[:, 2]
    rate = torch.exp(log_features.mm(parameters.T)).T
    data = pyro.sample("data", pdist.Poisson(rate=rate.clamp(0, upper_rate_bound)))
    return data

def two_param_poisson_glm(parameters, upper_rate_bound: int = 100_000, offset = 2):

    assert parameters.ndim == 2
    assert parameters.shape[1] == 2

    # Set theta3 negative to divide by it in the rule.
    theta = torch.zeros((parameters.shape[0], 3))
    theta[:, 0] = parameters[:, 0]
    theta[:, 1] = offset - parameters[:, 0]
    theta[:, 2] = -parameters[:, 1]
    rate = torch.exp(log_features.mm(theta.T)).T
    data = pyro.sample("data", pdist.Poisson(rate=rate.clamp(0, upper_rate_bound)))

    return data


def poisson_glm_constrained(parameters, upper_rate_bound: int = 100_000):
    # Add theta2 * x2 to the denominator to constrain the rule:
    # pst all should also contain the scaled post features of the current voxel, or not?
    # Maybe it should not, because the remaining voxels are the competitors.

    # Design matrix holds the log features.

    theta1_batch, theta2_batch, theta3_batch = parameters.T
    num_batch = theta1_batch.shape[0]

    log_pre = theta1_batch.unsqueeze(1) * log_features[:, 0].unsqueeze(0)
    log_post = theta2_batch.unsqueeze(1) * log_features[:, 1].unsqueeze(0)

    # Calculate postall in linear space to combine it with current post feature.
    log_post_all = theta3_batch * torch.log(
        features[:, 2].repeat(num_batch, 1).T
        - features[:, 1].repeat(num_batch, 1).T
        + features[:, 1].unsqueeze(1) ** theta2_batch.unsqueeze(0)
    )

    log_dso = log_pre + log_post - log_post_all.T
    rate = torch.exp(log_dso)
    data = pyro.sample("data", pdist.Poisson(rate=rate.clamp(0, upper_rate_bound)))
    return data


def get_prior_dist():
    return pdist.Uniform(low=prior_params["low"], high=prior_params["high"]).to_event(1)


if __name__ == "__main__":

    num_samples = 5
    theta = get_prior_dist()((num_samples,))

    torch.manual_seed(1)
    x = poisson_glm_constrained(theta)
    torch.manual_seed(1)
    x2 = []
    for th in theta:
        x2.append(poisson_glm_constrained(th.unsqueeze(0)))

    x2 = torch.cat(x2)

    assert (x == x2).all(), "Batch vs sequential not equal."

    assert x.shape == (
        num_samples,
        10,
    )

import pathlib

from consbi.simulators.poisson_glm import poisson_glm, get_prior_dist

BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH.joinpath("data")
RESULTS_PATH = BASE_PATH.joinpath("results")
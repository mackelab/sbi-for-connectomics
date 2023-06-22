import pathlib
import matplotlib

from matplotlib.font_manager import fontManager

from consbi.simulators.poisson_glm import get_prior_dist, poisson_glm

# Define paths
BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH.joinpath("data")
RESULTS_PATH = BASE_PATH.joinpath("results")

# Add fonts
fontManager.addfont(BASE_PATH.joinpath(".fonts/arial.ttf").as_posix())
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": "Arial"})
fontManager.addfont(BASE_PATH.joinpath(".fonts/arial_bold.ttf").as_posix())
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": "Arial-bold"})

del matplotlib

from setuptools import find_packages, setup

# Package meta-data.
NAME = "consbi"
DESCRIPTION = "simulation-based inference for computational connectomics"
URL = "https://github.com/mackelab/synaptic-rule-inference"
AUTHOR = "Jan Bölts, Philipp Harth"
REQUIRES_PYTHON = ">=3.6.0"

REQUIRED = [
    "matplotlib",
    "numpy",
    "scipy",
    "torch>=1.5.1",
    "pyro-ppl>=1.3.1",
    "sbibm @ git+https://github.com/mackelab/sbibm@connectomics-task#egg=sbibm",
    "tqdm",
    "sbi>=0.18.0",
    "scikit-learn",
    "pandas",
    "svgutils==0.3.1",
    "invoke",
    "jupyterlab",
]

EXTRAS = {
    "dev": [
        "autoflake",
        "black",
        "deepdiff",
        "flake8",
        "isort",
        "jupyter",
        "pep517",
        "pytest",
        "pyyaml",
    ],
}

setup(
    name=NAME,
    version="0.1.0",
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]
    ),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="AGPLv3",
)

<!-- [![GitHub license](https://img.shields.io/github/license/mackelab/sbi)](https://github.com/mackelab/sbi/blob/master/LICENSE.txt)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02505/status.svg)](https://doi.org/10.21105/joss.02505) -->

# Simulation-based inference for computational connectomics

This repository contains research code and figures of the "SBI for connectomics" paper (link to paper).
In the paper, we show how to use [simulation-based inference (SBI)](http://simulation-based-inference.org) to infer parameters of computational models used in connectomics. 

The repository is based on a small python package called [`consbi`](consbi/) that allows to simulate different wiring rules in the structural model of the rat barrel cortex. 
Additionally, it contains jupyter notebooks with [`tutorials`](tutorials/) and code for reproducing the [`figures`](figures/) shown in the paper.
Binary files with [`data`](data/) and [`results`](results/) for the work presented in the paper are stored using `git-lfs`. 

The repository for running SBI including detailed tutorials is located at https://github.com/mackelab/sbi. 

Please reach out and create an issue if you have any questions or encounter problems with using this repository.

## Usage
<!-- Google colab: You can play around with the [`tutorials`](tutorials/) in the browser using the following google colab link:  -->
Tutorials for the SBI workflow and code for reproducing the figures are available as Jupyter Notebooks that can be opened in the browser (without executing them). 

To run and play around with the code, you need to clone and install this repository locally, e.g., in the command line, run: 
```shell
git clone https://github.com/mackelab/sbi-for-connectomics.git
cd sbi-for-connectomics
pip install -e .
```
If you then start an `jupyter notebook` server locally you should be able to open and execute all notebooks. 

## Binary files are provided via `git-lfs`

The repository contains a number of large files that are provided via `git-lfs`. 
See the `git-lfs` documentation for installation and general info, https://git-lfs.com. 
Once `git-lfs` is installed locally, one can pull the large files contained in the repository just by cloning or pulling it. 




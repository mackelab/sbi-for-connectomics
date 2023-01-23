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

## Getting started
Google colab: You can play around with the [`tutorials`](tutorials/) in the browser using the following google colab link: 


Locally: clone the repository, `cd` into the local `sbi-for-connectomics` folder and run `pip install -e .`

## Binary files are provided via `git-lfs`

The repository contains a number of large files that are provided via `git-lfs`. 
See the `git-lfs` documentation for installation and general info, https://git-lfs.com. 
Once `git-lfs` is installed locally, one can pull the large files contained in the repository just by cloning or pulling it. 




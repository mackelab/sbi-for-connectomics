<!-- [![GitHub license](https://img.shields.io/github/license/mackelab/sbi)](https://github.com/mackelab/sbi/blob/master/LICENSE.txt)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02505/status.svg)](https://doi.org/10.21105/joss.02505) -->

# sbi-for-connectomics

This repository contains research code and figures of the "SBI for connectomics" paper (link to paper). 
It also contains a small python package called `consbi` that allows to simulate different wiring rules in the structural model of the rat barrel cortex. 

The repository for running SBI including detailed tutorials is located at https://github.com/mackelab/sbi. 

Please reach out and create an issue if you have any questions or encounter problems with using this repository.

## Getting started
You can play around with the tutorials (see below) in the browser using the following link: 

To run code locally: clone the repository, `cd` into the local `sbi-for-connectomics` folder and run `pip install -e .`

## Content
The `tutorials` folder contains three notebooks showing how to `simulate` connectomics data, how to use SBI to `infer` parameters, and how to `analyze` (TODO) the resulting posteriors.

The `figures` folder contains notebooks for reproducing all figures presented in the main text of the paper.

The `data` and `results` folders contain binary files saved via `git-lfs` (see below).

## Binary files are provided via `git-lfs`

The repository contains a number of large files that are provided via `git-lfs`. 
See the `git-lfs` documentation for installation and general info, https://git-lfs.com. 
Once `git-lfs` is installed locally, one can pull the large files contained in the repository just by cloning or pulling it. 




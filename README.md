# About

Master repository for `Improved Methods for k-Core Community Search`.

# Directory Structure

## Methods

All software is in `src/`.
Descriptions can be found in `src/README.md`.

## Running Experiments

The root should contain `input/` and `output/`.
These may be symlinked to a scratch directory.

To run each methods, we ingest in an edgelist with contiguous node ids, in a csv format.
The header is `source,target`, and all edges are present once (undirected graph).

The exact commands, as well as resources given for each method, are in the `slurm/pipeline.toml` directory.
This `toml` structure follows the specification according to [cc-slurm](https://github.com/IanChenUIUC/cc-slurm), as of version `v1.0.0`.

# Notes

## numactl

At the moment, we do not run any of the codes with `numactl`.

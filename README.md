# About

This repository hosts all code and experiments for `Improved Methods for k-Core Community Search`.
We provide implementations of SteinerKCore and Par-ShellStruct, novel scalable methods for retrieving communities with large min-degree.
In particular, given a set $Q$, return the $k$-core containing $Q$, of the largest $k$.

## Recommended scripts for running community search

### Many queries

When the user wants to retrieve communities for many distinct sets, e.g. $> 10$, then we recommend a three-phase approach that we call Par-ShellStruct.

1. compute the core decomposition of the graph, using `src/ours-icebug-commsearch/core_decomposition.py`
2. build the ShellStruct tree from the core decomposition, using `src/ours-icebug-commsearch/build_shellstruct.py.py`
3. use the ShellStruct tree to answer community search queries, using `src/ours-icebug-commsearch/query_shellstruct.py`

This package uses the `icebug` fork of NetworKit, a C++ library for large-scale graph analytics.
More details can be found in the README under `src/ours-icebug-commsearch`.

### Few queries

If the user wants to retrieve only a few queries, e.g. $< 10$, then we recommend a two-phase approach that we call SteinerKCore.

1. compute the core decomposition of the graph, using `src/ours-icebug-commsearch/core_decomposition.py`
2. use the core decomposition to answer community search queries, using `src/ours-python-commsearch/query_steiner.py`

This package uses the `icebug` fork of NetworKit, a C++ library for large-scale graph analytics.
The search is done using just-in-time compiled python.
More details can be found in the README under `src/ours-python-commsearch`.

## Reproducing experiments

The root should contain `input/` and `output/`.
These may be symlinked to a scratch directory.

To run each methods, we ingest in an edgelist with contiguous node ids, in a csv format.
The header is `source,target`, and all edges are present once (undirected graph).
At the moment, we do not run any of the codes with `numactl`.

The exact commands, as well as resources given for each method, are in the `slurm/pipeline.toml` file.
This `toml` structure follows the specification according to [cc-slurm](https://github.com/IanChenUIUC/cc-slurm), as of version `v1.0.1`.

Experimental data (stored in parquet files), as well as an R script for visualization, is given in the `analysis/` subdirectory.

# About

File format conversions, and a timing script.
The main dependency is [format_conversion](https://github.com/IanChenUIUC/format-conversion.git) a pybind11 module for converting edgelists to csr formats.

## timing

A custom timing script is used.
This mostly mimics the behavior of `/usr/bin/time -v`, but augments with some additional fields, namely

- anonymous memory
- proportional set size (off by default)

## conversions

The canonical input format is an edgelist in csv format, representing an undirected unweighted graph.
Nodes are from 0 to n-1, and there are no parallel-edges nor self-loops.

We then convert them to

- CSR format, stored as Arrow parquet or feather files
- GBBS format, converting to a text representation.
    later running the `src/gbbs-core-decomposition/bazel-bin/utils/converter` to get a binary representation of the graph.
- PKC format, stored as a text representation.

A utility `build_nodelist.py` is used when the graph may not have continuous IDs.
In this case, the row-index of the original ID is the compact ID, and must be the rank in the sorted list.

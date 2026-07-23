# About

A collection of drivers for NetworKit functions.
Each file takes in a networkit binary format (except for csv2nk).

## File Conversions

Takes in an edgelist (csv) for an undirected graph (with a header of `source,target`) and outputs the networkit binary format.

```
$ uv run csv2nk.py --help
Usage: csv2nk.py [OPTIONS]

Options:
  --edgelist PATH  [required]
  --output PATH    [required]
  --help           Show this message and exit.
```

## Centrality

For a given network, builds a csv of their coreness, local clustering, pagerank, and degree centrality.

```
$ uv run centrality.py --help

Usage: centrality.py [OPTIONS]

Options:
  --graph FILE   [required]
  --output FILE  [required]
  --help         Show this message and exit.
```

## Core Decomposition

```
$ uv run main.py --help
Usage: main.py [OPTIONS]

Options:
  --graph PATH       [required]
  --output PATH      [required]
  --threads INTEGER  Pin NetworKit to this many threads (default: NetworKit's own).
  --help             Show this message and exit.
```

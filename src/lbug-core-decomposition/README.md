# About

Using Ladybug to compute the $k$-core decomposition of a graph.
Injests data in CSR format (with uint64-typed parquet tables).

See [kcore](https://docs.ladybugdb.com/extensions/algo/kcore/) for more details.

## Building and Running

Dependencies are managed through [uv](https://docs.astral.sh/uv).

```
$ uv run main.py --help

Usage: main.py [OPTIONS] INDPTR_PATH INDICES_PATH OUT_PATH

  Core Decomposition using LadyBug

Options:
  --help  Show this message and exit.
```

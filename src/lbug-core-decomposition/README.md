# About

Using Ladybug to compute the $k$-core decomposition of a graph.
Injests data in CSR format (with uint64-typed parquet tables).
Note that the graph must not have symmetrized edges.

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

## Deprecation

As of July 22nd, 2026, I have been informed by the developers of LadybugDB that this way of calling Algorithms is no longer ideal, with plans to shift away.
From preliminary experiments, this is also much slower than any other core decomposition code.

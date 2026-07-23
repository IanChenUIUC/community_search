# About

The python implementations of the novel community search methods proposed.
We provide an implementation of ShellStruct (sequential) and SteinerKCore.

## Directory Structure

```
.
├── build_shellstruct.py
├── query_shellstruct.py
├── query_steiner.py
├── README.md
├── src
│   └── commsearch
│       ├── __init__.py
│       ├── base.py
│       ├── graph.py
│       ├── shellstruct.py
│       ├── steiner.py
│       └── structures
│           └── ...
├── tests
│   └── ...
├── pyproject.toml
├── uv.lock
└── warmup_jit.py
```

## Graph Format

We ingest graphs in CSR format, with an `indptr` and `indices` array in an arrow array, roughly according to [icebug-format](https://pypi.org/project/icebug-format/).
We deviate in that our `indices` is 32-bit instead of the 64-bit that Icebug requires.
An example for the `dnc` network can be found in `tests/data/`.

## Usage

We provide the scripts `query_*.py` and `build_shellstruct.py` for community search.
The source module can be installed directly and used.

We provide first-class support for answering a batch of queries in parallel.
This is controlled by the `threads` and `max_batch_size` parameters.
It greedily chunks the queries into batches based on the two supplied parameters, `spawn`-ing python interpreters to execute in parallel.

```
$ uv run build_shellstruct.py --help
Usage: build_shellstruct.py [OPTIONS] INDPTR_PATH INDICES_PATH SHELL_BASE_PATH
                            COREDECOMP

Options:
  --help  Show this message and exit.
```

```
$ uv run query_shellstruct.py --help
Usage: query_shellstruct.py [OPTIONS] SHELL_BASE_PATH QUERIES_PATH OUTPUT

Options:
  -b, --max_batch_size INTEGER
  --help                        Show this message and exit.
```

```
$ uv run query_steiner.py --help
Usage: query_steiner.py [OPTIONS] INDPTR_PATH INDICES_PATH COREDECOMP
                        QUERIES_PATH OUTPUT

Options:
  -t, --num_threads INTEGER
  -b, --max_batch_size INTEGER
  --help                        Show this message and exit.
```

## Some Details

This codebase is written in python, with a combination of JIT-compiled kernels (via Numba).
Caching is enabled by default (see the `__pycache__`), and a `warmup_jit.py` is useful for pre-compiling most of the kernels.

Testing is done via pytest, and can be run with `uv run pytest`.

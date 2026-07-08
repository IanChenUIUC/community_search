Community Search
================
Ian Chen and Haotian Yi

# Requirements and Dependencies

This project is setup using [uv](https://docs.astral.sh/uv/).
Running all scripts with uv should manage all the necessary dependencies automatically.

Ensure all dependencies are setup with
`uv run pytest`

# Directory Structure

Datastructures (such as heaps, disjoint sets, ...) are in the `src/csk/ds/` directory.
The main community search algorithms are in `src/csk/algs/` directory.
The driver code is in `src/csk/cli`.

```
├── examples
│   ├── data
│   │   └── ...
│   ├── ex_...
│   │   ├── query.txt
│   │   └── run.sh
├── pyproject.toml
├── pyrightconfig.json
├── README.md
├── src
│   └── csk
│       ├── algs
│       │   ├── ...
│       ├── cli
│       │   ├── ...
│       ├── ds
│       │   ├── ...
├── tests
│   ├── ...
└── uv.lock
```

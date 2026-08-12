# About

A collection of drivers for Icebug functions.

## Formats

Follows [icebug-format](https://pypi.org/project/icebug-format/).
Preferred usage is with zero-copy memory-mapped feather files.

## Scripts

### network-stats-extended

The ShellStruct format is written to two feather files:

- components.feather
- tree.feather

where the components table is sized to the number of vertices in the graph, and the tree to the number of vertices in the ShellStruct index.
the components table maps the vertices to the tree node index, under the "assignment" column.
the tree table contains four columns:

- the coreness of each node
- the vertices of each treenode (as a LargeList)
- the indptr and indices of the tree

where the indices has an additional padding (a tree of `n` nodes has `n-1` edges), and the `indptr` is missing the implicit `n` at the end.

The `network-stats-extended.py` script then analyzes the shellstruct, writing two CSVs: the per-network statistics (community count, average community size and volume as fractions), and each possible maximal k-core community in the graph with its size.

### network-stats

Computes basic statistics on the input network, namely the number of nodes/edges (of the graph and in the biggest component), and the clustering coefficient.

### shellstruct_strong_scaling

An experiment to measure how shellstruct scales on a fixed input network when varying number of threads.

### core-decomposition, build_shellstruct, query_*

For community search.

```
$ uv run core_decomposition.py --help
Usage: core_decomposition.py [OPTIONS] INDPTR_PATH INDICES_PATH OUTPUT

Options:
  --threads INTEGER  Pin NetworKit to this many threads (default: NetworKit's
                     own).
  --help             Show this message and exit.
```

```
$ uv run build_shellstruct.py --help
Usage: build_shellstruct.py [OPTIONS] INDPTR_PATH INDICES_PATH SHELL_BASE_PATH
                            [CORES]

Options:
  --help  Show this message and exit.
```

```
$ uv run query_shellstruct.py --help
Usage: query_shellstruct.py [OPTIONS] INDPTR_PATH INDICES_PATH COMPONENTS_PATH
                            TREE_PATH QUERIES_PATH OUTPUT

Options:
  --help  Show this message and exit.
```

```
$ uv run query_steiner.py --help
Usage: query_steiner.py [OPTIONS] INDPTR_PATH INDICES_PATH COREDECOMP_PATH
                        QUERIES_PATH OUTPUT

Options:
  --help  Show this message and exit.
```

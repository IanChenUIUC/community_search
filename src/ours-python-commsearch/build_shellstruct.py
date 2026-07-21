import sys
import pathlib
import time

import click
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as pf

from commsearch import Graph, ShellStruct


def _read_column(path: str, column: str) -> pa.Array:
    if path.endswith(".feather"):
        return pf.read_table(path, memory_map=True)[column].chunk(0)
    elif path.endswith(".parquet"):
        return pq.read_table(path)[column].combine_chunks()

    print(f"Files must be .parquet or .feather, got {path}", file=sys.stderr)
    sys.exit(1)


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("shell_base_path", type=click.Path(dir_okay=False))
@click.argument("coredecomp", type=click.Path(dir_okay=False, exists=True))
def main(indptr_path, indices_path, coredecomp, shell_base_path):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = Graph.from_csr(indptr, indices)
    cores = pl.read_csv(
        coredecomp,
        has_header=False,
        new_columns=["node_id", "core"],
        schema_overrides={"core": pl.UInt64},
    ).sort("node_id")
    scores = cores.get_column("core").to_numpy()
    shell = ShellStruct.build(graph, scores)

    components_path = pathlib.Path(shell_base_path).with_suffix(".components.feather")
    tree_path = pathlib.Path(shell_base_path).with_suffix(".tree.feather")
    shell.save(components_path, tree_path)


if __name__ == "__main__":
    main()

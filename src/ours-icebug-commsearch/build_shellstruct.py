import sys
import pathlib

import click
import networkit as nk
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq


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
@click.argument("cores", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--threads", type=int, default=None,
              help="Pin NetworKit to this many threads (default: NetworKit's own).")
def main(indptr_path, indices_path, shell_base_path, cores, threads):
    if threads:
        nk.setNumberOfThreads(threads)

    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    scores = None
    if cores:
        df = pl.read_csv(
            cores,
            has_header=False,
            new_columns=["node_id", "core"],
            schema_overrides={"core": pl.UInt64},
        ).sort("node_id")
        scores = df.get_column("core").to_numpy()

    shell = nk.scd.ShellStruct(graph)
    shell.build(scores)

    components = pathlib.Path(shell_base_path).with_suffix(".components.feather")
    tree = pathlib.Path(shell_base_path).with_suffix(".tree.feather")
    shell.save(components, tree)


if __name__ == "__main__":
    main()

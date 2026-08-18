import sys
import pathlib

import click
import networkit as nk
import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

SEED = 1234


def _read_column(path: str, column: str) -> pa.Array:
    if path.endswith(".feather"):
        return pf.read_table(path, memory_map=True)[column].chunk(0)
    elif path.endswith(".parquet"):
        return pq.read_table(path)[column].combine_chunks()

    print(f"Files must be .parquet or .feather, got {path}", file=sys.stderr)
    sys.exit(1)


def _parse_cells(ctx, param, value):
    cells = []
    for item in value:
        size, _, rest = item.partition(":")
        batch, _, out = rest.partition(":")
        if not out:
            raise click.BadParameter(f"expected SIZE:BATCH:OUTDIR, got {item!r}")
        cells.append((int(size), int(batch), out))
    return cells


def _sample(cc, valid, query_size, batch_size, reps):
    """One cell's query batches.

    The generator is seeded per *cell*, not per process. Every cell used to run in its
    own process and so started from this seed; reproducing those exact files is what
    keeps already-collected downstream results valid."""
    rng = np.random.default_rng(SEED)

    def single_query(size):
        for _ in range(1_000):
            query = rng.choice(valid, size, replace=False)
            labels = [cc.componentOfNode(q) for q in query]
            if all(label == labels[0] for label in labels):
                return query
        raise RuntimeError("single query failed after 1000 trials")

    return [[single_query(query_size) for _ in range(batch_size)] for _ in range(reps)]


def _write(out_directory, query_batches):
    out_directory = pathlib.Path(out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)
    for rep, batch in enumerate(query_batches):
        with open(out_directory / f"query{rep}.csv", "w") as f:
            f.writelines(",".join(map(str, query)) + "\n" for query in batch)


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_directory", type=click.Path(file_okay=False), required=False)
@click.option("-n", "--query_size", type=int, default=1)
@click.option("-b", "--batch_size", type=int, default=1)
@click.option("-r", "--reps", type=int, default=1)
@click.option("--cell", "cells", multiple=True, callback=_parse_cells,
              metavar="SIZE:BATCH:OUTDIR",
              help="Emit this cell; repeatable, and mutually exclusive with "
                   "OUT_DIRECTORY/-n/-b. Every cell is served from one graph build, "
                   "which on the largest networks costs minutes and dwarfs the sampling.")
def main(indptr_path, indices_path, out_directory, query_size, batch_size, reps, cells):
    if cells and out_directory:
        raise click.UsageError("pass either OUT_DIRECTORY or --cell, not both")
    if not cells and not out_directory:
        raise click.UsageError("need OUT_DIRECTORY, or at least one --cell")
    cells = cells or [(query_size, batch_size, out_directory)]

    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    cc = nk.components.ParallelConnectedComponents(graph).run()
    deg = nk.centrality.DegreeCentrality(graph).run().scores()
    valid = np.flatnonzero(deg >= np.quantile(deg, 0.99))  # top 1%

    for size, batch, out in cells:
        _write(out, _sample(cc, valid, size, batch, reps))


if __name__ == "__main__":
    main()

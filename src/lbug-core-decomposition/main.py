import pathlib
import sys

import click
import ladybug as lb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq


def _read_column(path: str, column: str) -> pa.Array:
    if path.endswith(".feather"):
        return pf.read_table(path)[column].combine_chunks()
    elif path.endswith(".parquet"):
        return pq.read_table(path)[column].combine_chunks()

    print(f"Files must be .parquet or .feather, got {path}", file=sys.stderr)
    sys.exit(1)


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path(dir_okay=False))
def main(indptr_path: str, indices_path: str, out_path: str) -> None:
    """Core Decomposition using LadyBug"""
    indptr_arr = _read_column(indptr_path, "indptr")  # pa.uint64
    indices_arr = _read_column(indices_path, "indices")  # pa.uint64
    n_nodes, n_edges = len(indptr_arr) - 1, len(indices_arr) // 2

    db = lb.Database()
    conn = lb.Connection(db)

    node_ids = pa.array(np.arange(n_nodes, dtype=np.uint64))
    nodes = pa.table({"id": node_ids})
    edges = pa.table({"to": indices_arr})

    conn.create_arrow_table(table_name="node", dataframe=nodes)
    conn.create_arrow_rel_table(
        table_name="edge",
        dataframe=edges,
        src_table_name="node",
        dst_table_name="node",
        layout=lb.ArrowRelTableLayout.CSR,
        indptr_dataframe=pa.table({"ptr": indptr_arr}),
        dst_col_name="to",
    )

    conn.execute("INSTALL algo")
    conn.execute("LOAD algo")
    conn.execute("CALL project_graph('G', ['node'], ['edge'])")

    pathlib.Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    cores = conn.execute(
        "CALL k_core_decomposition('G') RETURN node.id AS id, k_degree AS core"
    ).get_as_pl()

    (
        pl.DataFrame({"id": node_ids})
        .join(cores, on="id", how="left")
        .with_columns(pl.col("core").fill_null(0).cast(pl.Int64))
        .sort("id")
        .write_csv(out_path)
    )

    print(
        f"Writing core decomposiiton on {n_nodes=} {n_edges=} to {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

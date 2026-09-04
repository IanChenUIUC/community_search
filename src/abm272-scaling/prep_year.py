"""Build the CSR of the subgraph induced by every node present in or before a given year."""

import sys
import pathlib

import click
import numpy as np
import polars as pl
import format_conversion.format as fmt


def _edge_read_spec(path: str):
    if path.endswith(".parquet"):
        return fmt.EdgelistParquet.Read(source_col="source", target_col="target", base_index=0)
    elif path.endswith(".csv"):
        return fmt.CsvEdgelist.Read(sep=",", skip_rows=1, base_index=0)

    print(f"Edgelist must be .parquet or .csv, got {path}", file=sys.stderr)
    sys.exit(1)


def _scan_nodelist(path: str) -> pl.LazyFrame:
    if path.endswith(".parquet"):
        return pl.scan_parquet(path)
    elif path.endswith(".csv"):
        return pl.scan_csv(path)

    print(f"Nodelist must be .parquet or .csv, got {path}", file=sys.stderr)
    sys.exit(1)


@click.command()
@click.argument("edgelist_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("nodelist_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("year", type=int)
@click.argument("output_csr_base", type=click.Path(dir_okay=False))
@click.option("--threads", type=int, default=4)
def main(edgelist_path, nodelist_path, year, output_csr_base, threads):
    pathlib.Path(output_csr_base).parent.mkdir(parents=True, exist_ok=True)

    maxid = (
        _scan_nodelist(nodelist_path)
        .filter(pl.col("year") <= year)
        .select(pl.col("node_id").max())
        .collect()
        .item()
    )

    nodes_path = pathlib.Path(output_csr_base).with_suffix(".nodes.csv")
    pl.DataFrame({"node_id": np.arange(maxid + 1, dtype=np.int64)}).write_csv(nodes_path)

    nodes = fmt.NodeDescriptor(nodes_path, fmt.Nodelist.Csv(skip_rows=1, base_index=0))
    graph_in = fmt.GraphDescriptor(edgelist_path, _edge_read_spec(edgelist_path))
    graph_out = fmt.GraphDescriptor(
        output_csr_base,
        fmt.CsrParquet.Write(u64_indices=True),
    )
    fmt.convert(graph_in, graph_out, nodes=nodes, sort_neighbors=True, num_threads=threads)

    print(f"nodes={maxid + 1}")


if __name__ == "__main__":
    main()

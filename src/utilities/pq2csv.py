from pathlib import Path

import click
import format_conversion.format as fmt


@click.command()
@click.argument("input_parquet", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_csv", type=click.Path(dir_okay=False))
@click.option("--source-col", type=str, default="source")
@click.option("--target-col", type=str, default="target")
@click.option("--sep", type=str, default=",")
@click.option("--threads", type=int, default=1)
def main(input_parquet, output_csv, source_col, target_col, sep, threads):
    """Convert a parquet edge list to a csv representation of the edgelist with sorted rows"""
    graph = fmt.GraphDescriptor(
        input_parquet,
        fmt.EdgelistParquet.Read(
            source_col=source_col,
            target_col=target_col,
            base_index=0,
            keep_self_loops=False,
            directed=False,
        ),
    )
    output = fmt.GraphDescriptor(
        output_csv,
        fmt.CsvEdgelist.Write(
            sep=sep,
            source_col=source_col,
            target_col=target_col,
            base_index=0,
            header=True,
            expand_symmetric=False,
        ),
    )

    Path(output_csv).parent.mkdir(exist_ok=True, parents=True)
    fmt.convert(graph, output, nodes=None, sort_neighbors=True, num_threads=threads)


if __name__ == "__main__":
    main()

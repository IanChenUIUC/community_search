from pathlib import Path

import click
import format_conversion.format as fmt


@click.command()
@click.argument("input_csv", type=click.Path(exists=True))
@click.argument("output_csr", type=click.Path(dir_okay=False))
@click.option("--symmetrize/--no-symmetrize", "-s/-ns", type=bool, default=True)
@click.option("--sep", type=str, default=",")
@click.option("--skip-rows", type=int, default=1)
def main(input_csv, output_csr, symmetrize, sep, skip_rows):
    graph = fmt.GraphDescriptor(
        input_csv,
        fmt.CsvEdgelist.Read(
            sep=sep,
            skip_rows=skip_rows,
            directed=not symmetrize,
        ),
    )
    # A write spec takes a prefix and appends .indices.parquet / .indptr.parquet.
    output = fmt.GraphDescriptor(output_csr, fmt.CsrParquet.Write(u64_indices=True))

    Path(output_csr).parent.mkdir(exist_ok=True, parents=True)
    fmt.convert(graph, output, nodes=None, sort_neighbors=True)


if __name__ == "__main__":
    main()

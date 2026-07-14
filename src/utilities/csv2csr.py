from pathlib import Path

import click
import format_conversion.format as fmt


@click.command()
@click.argument("input_csv", type=click.Path(exists=True))
@click.argument("output_csr", type=click.Path(dir_okay=False))
@click.option("--symmetrize/--no-symmetrize", "-s/-ns", default=True)
def main(input_csv, output_csr, symmetrize=True) -> None:
    input_fmt = fmt.EdgesFormat.CSV_EDGELIST
    output_fmt = fmt.EdgesFormat.CSR_PARQUET

    input_opts = fmt.ParseOptions()
    input_opts.skip_rows = 1
    input_opts.sep = "\t" if input_csv.endswith(".tsv") else ","
    input_opts.directed = not symmetrize
    input_opts.sort_neighbors = True
    output_opts = fmt.ParseOptions()
    output_opts.use_u64_indices = True
    graph = fmt.GraphDescriptor(input_csv, input_fmt, input_opts)

    Path(output_csr).parent.mkdir(exist_ok=True, parents=True)
    fmt.convert(
        graph,
        nodes=None,
        output_path=output_csr,
        output_fmt=output_fmt,
        output_opts=output_opts,
    )


if __name__ == "__main__":
    main()

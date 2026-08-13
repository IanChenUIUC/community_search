import click

import pyarrow.csv as csv
import pyarrow.parquet as pq


@click.command()
@click.argument("files", nargs=-1, required=True)
def main(files):
    """Write each CSV alongside itself as .parquet."""
    for path in files:
        pq.write_table(csv.read_csv(path), path.replace(".csv", ".parquet"))


if __name__ == "__main__":
    main()

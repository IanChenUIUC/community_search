import pathlib
import itertools as it

import click
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

from commsearch import Graph, ShellStruct, SteinerKCore, Community

"""
report the statistics of the community and writes to the output
    - coreness
    - number of nodes
    - histogram of diferent fields
TODO: compute real statistics, e.g. diameter, conductance, density, etc.
"""

field_codes = [
    "math",
    "physics",
    "chemistry",
    "bio",
    "scientometrics",
    "math|physics",
    "math|chemistry",
    "math|bio",
    "math|scientometrics",
    "physics|chemistry",
    "physics|bio",
    "physics|scientometrics",
    "chemistry|bio",
    "chemistry|scientometrics",
    "bio|scientometrics",
]


@click.command()
@click.argument("working_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("all_nodelist", type=click.Path(exists=True, dir_okay=False))
@click.argument("seed_nodelist", type=click.Path(exists=True, dir_okay=False))
def main(working_dir, all_nodelist, seed_nodelist):
    ShellStruct.warmup()
    working_dir = pathlib.Path(working_dir)

    nodes = pl.read_csv(all_nodelist)
    seeds = pl.read_csv(seed_nodelist)
    queries = [[q] for q in seeds.filter(pl.col("role") == "founder").get_column("integer_id")]

    schema = ["year", "query", "key", "value"]
    data = []

    for year in range(2026, nodes.get_column("year").max() + 1):
        components = working_dir / str(year) / "shell.components.feather"
        tree = working_dir / str(year) / "shell.tree.feather"
        shell = ShellStruct.load(components, tree)

        for query in queries:
            coreness, comm = shell.expand_one_community(query)
            data.append([year, query[0], "coreness", coreness])

            fields = nodes.filter(pl.col("node_id").is_in(comm)).get_column("field")
            counts = np.bincount(fields, minlength=15)
            for key, value in zip(field_codes, counts):
                data.append([year, query[0], key, value])

    df = pl.DataFrame(data, schema=schema, orient="row")
    df.write_csv(working_dir / "comm-evolve-stats.csv")


if __name__ == "__main__":
    main()

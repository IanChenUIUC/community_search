"""Derive a dense per-node field label for the whole ABM graph."""

import sys
import pathlib
from itertools import combinations

import click
import numpy as np
import polars as pl
from numba import njit

FIELDS = ["math", "physics", "chemistry", "bio", "scientometrics"]
LEGEND = {name: i for i, name in enumerate(FIELDS)}
LEGEND.update(
    {f"{FIELDS[i]}|{FIELDS[j]}": 5 + k for k, (i, j) in enumerate(combinations(range(5), 2))}
)


def field_code(field: str) -> int:
    return LEGEND["|".join(sorted(field.split("|"), key=FIELDS.index))]


def scan_nodelist(path: str) -> pl.LazyFrame:
    """Scan a nodelist, dispatching on extension (both real and test inputs are parquet)."""
    if path.endswith(".parquet"):
        return pl.scan_parquet(path)
    elif path.endswith(".csv"):
        return pl.scan_csv(path)

    print(f"Nodelist must be .parquet or .csv, got {path}", file=sys.stderr)
    sys.exit(1)


@njit
def propagate(field, gen):
    for i in range(field.size):
        p = gen[i]
        if p >= 0:  # agent
            field[i] = field[p]


@click.command()
@click.argument("nodelist_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tc_nodelist_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
def main(nodelist_path, tc_nodelist_path, output_path):
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    legend = out.parent / "field_codes.csv"

    nl = (
        scan_nodelist(nodelist_path)
        .select(
            pl.col("node_id").cast(pl.Int64),
            pl.col("year").cast(pl.Int32),
            pl.col("generator_node_string").cast(pl.Int32, strict=False).fill_null(-1),
        )
        .collect()
    )
    n = nl["node_id"].max() + 1
    dense = pl.DataFrame({"node_id": pl.arange(0, n, eager=True)}).join(
        nl, on="node_id", how="left", maintain_order="left"
    )
    gen = dense["generator_node_string"].fill_null(-1).to_numpy()
    year = dense["year"].fill_null(-1).to_numpy()

    field = np.full(n, -1, dtype=np.int8)
    tc = pl.read_csv(tc_nodelist_path, columns=["integer_id", "field"])
    codes = np.fromiter((field_code(f) for f in tc["field"]), dtype=np.int8, count=tc.height)
    field[tc["integer_id"].to_numpy()] = codes

    propagate(field, gen)

    df = pl.DataFrame(
        {"node_id": np.arange(n, dtype=np.int64), "year": year, "field": field}
    ).with_columns(
        pl.when(pl.col("year") >= 0).then(pl.col("year")).alias("year"),
        pl.when(pl.col("field") >= 0).then(pl.col("field")).alias("field"),
    )
    df.write_csv(out)
    pl.DataFrame({"code": list(LEGEND.values()), "fields": list(LEGEND.keys())}).write_csv(legend)


if __name__ == "__main__":
    main()

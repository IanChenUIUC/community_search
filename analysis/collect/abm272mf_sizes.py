import pathlib

import click
import pandas as pd
import pyarrow.feather as pf

import common

COLUMNS = ["year", "nodes", "edges"]


def year_size(path):
    """A year's (nodes, edges) read off its CSR indptr, or (None, None) if it never built."""
    path = pathlib.Path(path)
    if not path.exists():
        return None, None

    indptr = pf.read_table(path, memory_map=True)["indptr"].chunk(0)
    return len(indptr) - 1, indptr[-1].as_py() // 2


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the abm272 per-year graph sizes into analysis/abm-sizes.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    out = root / "output"
    csv = root / "analysis" / "abm272mf-sizes.csv"

    rows = []
    for year in spec["recipe"]["abm272-build-network"]["years"]:
        d = out / "abm272" / str(year) / "abm272-build-network"
        rows.append([year, *year_size(d / "graph.indptr.feather")])

    df = pd.DataFrame(rows, columns=COLUMNS)
    df[["nodes", "edges"]] = df[["nodes", "edges"]].astype("Int64")
    df.to_csv(csv, index=False)
    print(f"wrote {csv}: {len(df)} rows, {df.nodes.notna().sum()} built")


if __name__ == "__main__":
    main()

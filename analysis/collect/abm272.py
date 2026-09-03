import pathlib
import re

import click
import pandas as pd

import common

COLUMNS = ["year", "nodes", "stage", "rep", "stat", "value", "status"]

# abm272-build-network appends three mytime records to one timing.txt
BUILD_STAGES = ["prep-year", "pq2pf32", "pq2pf64"]
YEAR_STAGES = ["genquery", "core-decomp", "shellstruct-offline"]
REP_STAGES = ["steiner", "shellstruct-online"]

NODES = re.compile(r"^nodes=(\d+)$", re.M)


def node_count(path):
    """The `nodes=` line prep_year.py prints -- the x-axis -- or None if the year never built."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    match = NODES.search(path.read_text())
    return int(match.group(1)) if match else None


def year_rows(year, reps, out, states):
    """One year's stages: three build sub-stages, three per-year stages, two per-rep stages."""
    d = out / "abm272" / str(year)
    node = f"abm272-build-network-{year}"
    nodes = node_count(d / "abm272-build-network" / "stdout.txt")

    records = common.read_mytimes(d / "abm272-build-network" / "timing.txt")
    records += [None] * (len(BUILD_STAGES) - len(records))

    rows = []
    for stage, mytime in zip(BUILD_STAGES, records):
        status = common.row_status(f"{node} {stage}", mytime, states.get(node))
        common.emit(rows, [year, nodes, stage, None], mytime, status)

    for stage in YEAR_STAGES:
        node = f"abm272-{stage}-{year}"
        mytime = common.read_mytime(d / f"abm272-{stage}" / "timing.txt")
        common.emit(rows, [year, nodes, stage, None], mytime,
                    common.row_status(node, mytime, states.get(node)))

    for stage in REP_STAGES:
        sd = d / f"abm272-{stage}"
        for rep in reps:
            node = f"abm272-{stage}-{year}-{rep}"
            mytime = common.read_mytime(sd / f"timing-rep{rep}.txt")
            common.emit(rows, [year, nodes, stage, rep], mytime,
                        common.row_status(node, mytime, states.get(node)),
                        common.querytimes(sd / f"querytimes-rep{rep}.csv", "pycs"))

    return rows


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the abm272 per-year scaling timings into analysis/abm272.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    states = common.task_states(root / "slurm" / ".pipeline" / "run.jsonl")
    out = root / "output"
    csv = root / "analysis" / "abm272.csv"

    years = spec["recipe"]["abm272-build-network"]["years"]
    reps = spec["recipe"]["abm272-genquery"]["reps"]

    rows = []
    for year in years:
        rows += year_rows(year, reps, out, states)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["rep"] = df["rep"].astype("Int64")
    df.to_csv(csv, index=False)
    cells = df.drop_duplicates(["year", "stage", "rep"])
    print(f"wrote {csv}: {len(df)} rows, {len(cells)} cells")
    print(pd.crosstab(cells.stage, cells.status, margins=True,
                      margins_name="total").to_string())


if __name__ == "__main__":
    main()

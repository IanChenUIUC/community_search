import pathlib

import click
import pandas as pd

import common

COLUMNS = ["network", "framework", "stage", "stat", "value", "status"]

FRAMEWORKS = [("icebug", "icebug"), ("python", "pycs")]
STAGES = [("build_shell", None), ("query_shell", "shell"), ("query_steiner", "steiner")]


def network_rows(network, out, states):
    """One network's seven phases, all run by the single pyvib-query job.

    genquery produces the query set both frameworks read, so it repeats under each of them."""
    d = out / network / "pyvib-query"
    node = f"pyvib-query-{network}"
    task = states.get(node)

    genquery = common.read_mytime(d / "timing-genquery.txt")
    genquery_status = common.row_status(f"{node} genquery", genquery, task)

    rows = []
    for framework, package in FRAMEWORKS:
        common.emit(rows, [network, framework, "genquery"], genquery, genquery_status)
        for stage, stem in STAGES:
            mytime = common.read_mytime(d / f"timing-{framework}_{stage}.txt")
            query_s = (common.querytimes(d / f"{framework}_{stem}_querytimes.csv", package)
                       if stem else common.NO_QUERY)
            common.emit(rows, [network, framework, stage], mytime,
                        common.row_status(f"{node} {framework}_{stage}", mytime, task),
                        query_s)
    return rows


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the python-vs-icebug comparison into analysis/py-v-ib.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    states = common.task_states(root / "slurm" / ".pipeline" / "run.jsonl")
    out = root / "output"
    csv = root / "analysis" / "py-v-ib.csv"

    rows = []
    for network in spec["defaults"]["training_networks"]:
        rows += network_rows(network, out, states)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(csv, index=False)
    cells = df.drop_duplicates(["network", "framework", "stage"])
    print(f"wrote {csv}: {len(df)} rows, {len(cells)} cells")
    print(pd.crosstab([cells.framework, cells.stage], cells.status, margins=True,
                      margins_name="total").to_string())


if __name__ == "__main__":
    main()

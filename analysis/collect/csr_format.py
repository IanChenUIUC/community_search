import pathlib

import click
import pandas as pd

import common

COLUMNS = ["network", "stage", "stat", "value", "status"]

STAGES = ["csv2csr", "pq2pf32", "pq2pf64"]


def read_mytimes(path):
    """The appended mytime records in order, one dict each; `exit_code` opens a new record."""
    path = pathlib.Path(path)
    if not path.exists():
        return []

    records = []
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key == "exit_code":
            records.append({})
        if key in common.MYTIME_KEYS and records:
            records[-1][key] = float(value)
    return records


def network_rows(network, out, states):
    """One network's three conversion stages, all run by the single csr-format task."""
    d = out / network / "csr-format"
    node = f"csr-format-{network}"
    task = states.get(node)

    records = read_mytimes(d / "timing.txt")
    records += [None] * (len(STAGES) - len(records))

    rows = []
    for stage, mytime in zip(STAGES, records):
        status = common.row_status(f"{node} {stage}", mytime, task)
        common.emit(rows, [network, stage], mytime, status)
    return rows


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the CSV-to-CSR conversion timings into analysis/csr-format.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    states = common.task_states(root / "slurm" / ".pipeline" / "run.jsonl")
    out = root / "output"
    csv = root / "analysis" / "csr-format.csv"

    rows = []
    for network in spec["defaults"]["all_networks"]:
        rows += network_rows(network, out, states)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(csv, index=False)
    cells = df.drop_duplicates(["network", "stage"])
    print(f"wrote {csv}: {len(df)} rows, {len(cells)} cells")
    print(pd.crosstab(cells.stage, cells.status, margins=True,
                      margins_name="total").to_string())


if __name__ == "__main__":
    main()

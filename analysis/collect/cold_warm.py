import pathlib
import re
import sys

import click
import pandas as pd

import common

COLUMNS = ["network", "cache", "mode", "op", "rep", "stat", "value", "status"]
DIR = "cold-warm"
INNER = re.compile(r"^\d+\.\d+$", re.M)

SLOTS = [("serial", "serial", 0, "coreness"),
         ("serial", "serial", 0, "query"),
         ("simult", "simult1", 0, "coreness"),
         ("simult", "simult2", 1, "coreness"),
         ("simult", "simult1", 0, "query"),
         ("simult", "simult2", 1, "query")]

PHASES = [(cache, mode, op, rep, f"timing-{cache}-{slot}-{op}.txt")
          for cache in ("cold", "warm")
          for mode, slot, rep, op in SLOTS]


def inner_times(path, node):
    """The elapsed seconds each phase prints, in the order the driver runs them.

    The two simult processes tee into one stdout concurrently, so which of a simult pair a
    number came from is arbitrary and only their aggregate is meaningful."""
    path = pathlib.Path(path)
    times = INNER.findall(path.read_text()) if path.exists() else []
    if len(times) != len(PHASES):
        print(f"warning: {node}: stdout has {len(times)} elapsed times, "
              f"expected {len(PHASES)}; leaving inner_s empty", file=sys.stderr)
        return [None] * len(PHASES)
    return [float(t) for t in times]


def network_rows(network, out, states):
    """Every phase of one network's run, sharing the one job state that covers them all."""
    rows = []
    d = out / network / DIR
    node = f"{DIR}-{network}"
    task = states.get(node)
    for (cache, mode, op, rep, filename), inner in zip(PHASES,
                                                       inner_times(d / "stdout.txt", node)):
        mytime = common.read_mytime(d / filename)
        status = common.row_status(f"{node} {cache} {mode} {op} rep{rep}", mytime, task)
        key = [network, cache, mode, op, rep]
        common.emit(rows, key, mytime, status)
        rows.append([*key, "inner_s", inner, status])
    return rows


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the cold/warm page-cache experiment into analysis/cold-warm.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    states = common.task_states(root / "slurm" / ".pipeline" / "run.jsonl")
    output = root / "output"
    out = root / "analysis" / "cold-warm.csv"

    rows = []
    for network in spec["defaults"]["training_networks"]:
        rows.extend(network_rows(network, output, states))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["rep"] = df["rep"].astype("Int64")
    df.to_csv(out, index=False)
    cells = df.drop_duplicates(["network", "cache", "mode", "op", "rep"])
    print(f"wrote {out}: {len(df)} rows, {len(cells)} cells")
    print(pd.crosstab([cells.cache, cells["mode"], cells.op], cells.status, margins=True,
                      margins_name="total").to_string())


if __name__ == "__main__":
    main()

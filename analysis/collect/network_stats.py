import pathlib
import re

import click
import pandas as pd

import common

COLUMNS = ["network", "stat", "value", "status"]

BASE_STATS = [("n", r"^n = (\d+), m = \d+"),
              ("m", r"^n = \d+, m = (\d+)"),
              ("avg_clustering_coeff", r"average clustering coeff = ([\d.eE+-]+)"),
              ("giant_n", r"giant component: n = (\d+) m = \d+"),
              ("giant_m", r"giant component: n = \d+ m = (\d+)")]

EXTENDED_STATS = ["communities", "avg_community_size", "avg_community_size_fraction",
                  "avg_volume_fraction", "degeneracy", "peeling_complexity"]


def base_rows(network, out, states):
    """The base statistics, which the script only ever prints to stdout."""
    d = out / network / "network-stats"
    node = f"stat-network-stats-{network}"
    mytime = common.read_mytime(d / "timing.txt")
    status = common.row_status(node, mytime, states.get(node))

    stdout = d / "stdout.txt"
    text = stdout.read_text() if stdout.exists() else ""
    return [[network, stat, float(m.group(1)) if (m := re.search(p, text, re.M)) else None,
             status] for stat, p in BASE_STATS]


def extended_rows(network, out, states):
    """The shellstruct and GBBS statistics, which share one stats.csv."""
    d = out / network / "network-stats-extended"
    node = f"stat-network-stats-extended-{network}"
    mytime = common.read_mytime(d / "timing.txt")
    status = common.row_status(node, mytime, states.get(node))

    result = d / "stats.csv"
    values = (pd.read_csv(result).set_index("stat")["value"] if result.exists()
              else pd.Series(dtype=float))
    return [[network, stat, values.get(stat), status] for stat in EXTENDED_STATS]


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the per-network statistics table into analysis/network-stats.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    states = common.task_states(root / "slurm" / ".pipeline" / "run.jsonl")
    out = root / "output"
    csv = root / "analysis" / "network-stats.csv"

    rows = []
    for network in spec["defaults"]["all_networks"]:
        rows += base_rows(network, out, states)
        rows += extended_rows(network, out, states)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(csv, index=False)
    print(f"wrote {csv}: {len(df)} rows, {df.network.nunique()} networks")
    print(pd.crosstab(df.stat, df.status, margins=True, margins_name="total").to_string())


if __name__ == "__main__":
    main()

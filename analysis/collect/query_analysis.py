import pathlib

import click
import pandas as pd

import common

COLUMNS = ["network", "centrality", "size", "threshold", "cores", "status"]
DIR = "query-analysis"


def network_rows(network, out, states):
    """One network's per-query rows tagged with its job status, and that status."""
    d = out / network / DIR
    node = f"stat-{DIR}-{network}"
    mytime = common.read_mytime(d / "timing.txt")
    status = common.row_status(node, mytime, states.get(node))

    result = d / "query_analysis.csv"
    if not result.exists():
        return None, status

    df = pd.read_csv(result)
    df["status"] = status
    return df[COLUMNS], status


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the query-analysis sweep into analysis/query-analysis.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    states = common.task_states(root / "slurm" / ".pipeline" / "run.jsonl")
    output = root / "output"
    out = root / "analysis" / "query-analysis.csv"

    frames, statuses = [], []
    for network in spec["defaults"]["all_networks"]:
        df, status = network_rows(network, output, states)
        statuses.append([network, status])
        if df is not None:
            frames.append(df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    df.to_csv(out, index=False)
    cells = df.drop_duplicates(["network", "centrality", "size", "threshold"])
    print(f"wrote {out}: {len(df)} rows, {len(cells)} cells")
    st = pd.DataFrame(statuses, columns=["network", "status"])
    print(pd.crosstab(st.network, st.status, margins=True, margins_name="total").to_string())


if __name__ == "__main__":
    main()

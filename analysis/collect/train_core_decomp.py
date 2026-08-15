import pathlib

import click
import pandas as pd

import common

METHODS = {"gbbs": "traincd-gbbs-core-decomp",
           "ucr": "traincd-ucr-core-decomp",
           "pkc": "traincd-pkc-core-decomp",
           "lbug": "traincd-lbug-core-decomp",
           "nk": "traincd-nk-core-decomp",
           "ib": "traincd-icebug-core-decomp"}


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the training core-decomposition comparison into analysis/train-core-decomp.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    states = common.task_states(root / "slurm" / ".pipeline" / "run.jsonl")
    output = root / "output"
    out = root / "analysis" / "train-core-decomp.csv"

    networks = spec["defaults"]["training_networks"]

    rows = []
    for network in networks:
        for method, recipe in METHODS.items():
            timing = output / network / recipe / "timing.txt"
            mytime = common.read_mytime(timing)
            status = common.row_status(f"{recipe} {network}", mytime,
                                       states.get(f"{recipe}-{network}"))
            for stat in common.MYTIME_KEYS:
                rows.append([network, method, stat, (mytime or {}).get(stat), status])

    df = pd.DataFrame(rows, columns=["network", "method", "stat", "value", "status"])
    df.to_csv(out, index=False)
    cells = df.drop_duplicates(["network", "method"])
    print(f"wrote {out}: {len(df)} rows, {len(cells)} cells")
    print(pd.crosstab(cells.method, cells.status, margins=True,
                      margins_name="total").to_string())


if __name__ == "__main__":
    main()

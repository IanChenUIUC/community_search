#!/usr/bin/env python3
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
@click.option("--spec", default=common.SPEC, type=click.Path(exists=True, dir_okay=False))
@click.option("--log", default=common.LOG, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", default=common.OUTPUT, type=click.Path(exists=True, file_okay=False))
@click.option("--out", default=common.ANALYSIS / "train-core-decomp.csv",
              type=click.Path(dir_okay=False))
def main(spec, log, output, out):
    spec = common.load_spec(spec)
    states = common.task_states(log)
    networks = spec["defaults"]["training_networks"]
    common.report("traincd-*-core-decomp", networks=len(networks), methods=len(METHODS))

    rows = []
    for network in networks:
        for method, recipe in METHODS.items():
            timing = pathlib.Path(output) / network / recipe / "timing.txt"
            mytime = common.read_mytime(timing)
            status = common.row_status(f"{recipe} {network}", mytime,
                                       states.get(f"{recipe}-{network}"))
            for stat in common.MYTIME_KEYS:
                rows.append([network, method, stat, (mytime or {}).get(stat), status])

    df = pd.DataFrame(rows, columns=["network", "method", "stat", "value", "status"])
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, "
          f"{(df.status != 'ok').sum() // len(common.MYTIME_KEYS)} cells not ok")


if __name__ == "__main__":
    main()

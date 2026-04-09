from pathlib import Path

import click
import numpy as np
import pandas as pd


def write_query(path: Path, queries: np.ndarray):
    with path.open("wb") as f:
        np.savetxt(f, queries, delimiter="\n", fmt="%d")


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--outdir", required=True, type=click.Path())
def main(edgelist, outdir):
    rng = np.random.default_rng(1234)
    edges = pd.read_csv(edgelist, sep="\\s+", header=None, names=["u", "v"])
    nodes, degrees = np.unique(edges[["u", "v"]].values.ravel(), return_counts=True)

    bot3 = nodes[degrees <= np.quantile(degrees, 0.03)]
    top3 = nodes[degrees >= np.quantile(degrees, 0.97)]

    dir = Path(outdir)
    dir.mkdir(parents=True, exist_ok=True)

    sizes = [1, 10, 1_000, 10_000]
    for size in sizes:
        high = rng.choice(top3, size, replace=True)
        write_query(dir / f"high_{size}.txt", high)
        rand = rng.choice(nodes, size, replace=True)
        write_query(dir / f"rand_{size}.txt", rand)
        low = rng.choice(bot3, size, replace=True)
        write_query(dir / f"low_{size}.txt", low)


if __name__ == "__main__":
    main()

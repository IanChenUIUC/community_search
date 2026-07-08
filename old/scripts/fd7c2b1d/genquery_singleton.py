from pathlib import Path

import click
import numpy as np
import polars as pl


def write_query(path: Path, queries: np.ndarray):
    with path.open("wb") as f:
        np.savetxt(f, queries, delimiter="\n", fmt="%d")


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--outdir", required=True, type=click.Path())
@click.option("--header", required=False, type=bool, default=True)
@click.option("--sep", required=False, type=str, default=",")
def main(edgelist, outdir, header=True, sep=","):
    sizes = [1, 10, 100]
    size = max(sizes)

    lf = pl.scan_csv(edgelist, separator=sep, has_header=header, new_columns=["u", "v"])

    degs = (
        lf.unpivot(value_name="node")
        .group_by("node")
        .agg(pl.len().alias("d"))
        .collect(engine="streaming")
    )

    threshold = degs["d"].quantile(0.99)

    dir = Path(outdir)
    dir.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        high_candidates = degs.filter(pl.col("d") >= threshold)
        high = high_candidates.sample(n=size, seed=1234)["node"].to_numpy()
        rand = (degs.sample(n=size, seed=5678))["node"].to_numpy()

        write_query(dir / f"high_{size}.txt", high)
        write_query(dir / f"rand_{size}.txt", rand)


if __name__ == "__main__":
    main()

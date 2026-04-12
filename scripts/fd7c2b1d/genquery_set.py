from pathlib import Path

import click
import numpy as np
import polars as pl


def write_queries(dir: Path, name: str, queries: list[list]):
    with (dir / name).open("wb") as f:
        for query in queries:
            np.savetxt(f, np.array(query).reshape(1, -1), fmt="%d", delimiter=",")


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--outdir", required=True, type=click.Path())
@click.option("--header", required=False, type=bool, default=True)
@click.option("--sep", required=False, type=str, default=",")
def main(edgelist, outdir, header=True, sep=","):
    n = 10

    lf = pl.scan_csv(edgelist, separator=sep, has_header=header, new_columns=["u", "v"])
    degs = (
        lf.unpivot(value_name="node")
        .group_by("node")
        .agg(pl.len().alias("d"))
        .collect(engine="streaming")
    )
    t = degs["d"].quantile(0.99)
    roots = (degs.filter(pl.col("d") >= t).sample(n=n, seed=1234))["node"].to_numpy()

    roots = pl.DataFrame({"node": roots}).lazy()

    edges_forward = lf.select(node=pl.col("u"), neighbor=pl.col("v"))
    edges_backward = lf.select(node=pl.col("v"), neighbor=pl.col("u"))
    edges = pl.concat([edges_forward, edges_backward])

    neighbors = (
        edges.join(roots, on="node", how="inner")
        .group_by("node")
        .agg(pl.col("neighbor").alias("all_neighbors"))
        .collect(engine="streaming")
    )

    sampled = neighbors.with_columns(
        pl.col("all_neighbors").list.sample(n=5, seed=5678).alias("sampled_5")
    )

    take_5 = np.vstack(sampled["sampled_5"].to_list())
    take_all = sampled["all_neighbors"].to_list()

    dir = Path(outdir)
    dir.mkdir(parents=True, exist_ok=True)
    write_queries(dir, "take_all.txt", take_all)
    write_queries(dir, "take_5.txt", take_5)


if __name__ == "__main__":
    main()

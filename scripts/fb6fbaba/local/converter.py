import click
import numpy as np
import pandas as pd


def get_degrees(uvals, n):
    return np.bincount(uvals, minlength=n).astype(np.uint64)
    # starts = np.searchsorted(uvals, np.arange(n), side="left")
    # ends = np.searchsorted(uvals, np.arange(n), side="right")
    # return (ends - starts).astype(np.uint64)


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
@click.option("--mapping", required=True, type=click.Path())
def main(edgelist, output, mapping):
    df = pd.read_csv(edgelist, sep=r"\s+", header=None, names=["u", "v"])

    # factorizing for 0-indexed nodes
    all_nodes = pd.concat([df["u"], df["v"]], ignore_index=True)
    codes, uniques = pd.factorize(all_nodes)
    df["u"] = codes[: len(df)].astype("uint64")
    df["v"] = codes[len(df) :].astype("uint64")

    # creating mapping
    n = uniques.size
    mapping_df = pd.DataFrame({"orig": uniques, "new": np.arange(n, dtype=np.uint64)})
    mapping_df.to_csv(mapping, index=False)

    # getting undirected graph
    reversed = df.rename(columns={"u": "v", "v": "u"})
    undirected = pd.concat([df, reversed], ignore_index=True)
    undirected = undirected.drop_duplicates()
    undirected = undirected.sort_values(["u", "v"], ignore_index=True)

    degrees = np.bincount(undirected["u"].to_numpy(), minlength=n).astype(np.uint64)

    offsets = np.zeros(n, dtype=np.uint64)
    offsets[1:] = np.cumsum(degrees[:-1], dtype=np.uint64)
    offsets += n

    edges = undirected["v"].to_numpy()

    with open(output, "wb") as f:
        offsets.tofile(f)
        edges.tofile(f)


if __name__ == "__main__":
    main()

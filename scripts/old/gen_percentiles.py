#!/usr/bin/python3

import click
import pandas as pd


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
@click.option("--percentile", required=True, type=click.INT)
@click.option("--number", required=True, type=click.INT, help="number samples")
def main(edgelist, output, percentile, number):
    edges = pd.read_csv(edgelist, sep="\\s+", names=["src", "dst"])

    # value_counts sorts in descending degree
    # also double counts, but sufficient for getting ranks
    nodes = (
        pd.concat([edges["src"], edges["dst"]])
        .value_counts()
        .rename_axis("node")
        .reset_index(name="degree")
    )

    # select the 1% of nodes at percentile [p, p+1]
    # i.e. p=99 means the 99% to 100% largest node degrees
    k = int(len(nodes) * (1 - percentile / 100))
    l = int(len(nodes) * (1 - (percentile + 1) / 100))
    selected = nodes[l:k]

    # take n samples
    samples = selected.sample(number)
    samples.to_csv(output, sep="\t", index=False, header=False, columns=["node"])


if __name__ == "__main__":
    main()

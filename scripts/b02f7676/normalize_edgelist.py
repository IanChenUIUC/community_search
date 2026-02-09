from pathlib import Path

import click
import pandas as pd


@click.command()
@click.argument("input", required=True, type=click.Path(exists=True))
@click.argument("output", required=True, type=click.Path())
def main(input, output):
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(input, sep="\\s+", names=["u", "v"])
    codes, uniques = pd.factorize(pd.concat([edges["u"], edges["v"]]))
    edges["u"], edges["v"] = codes[: len(edges)], codes[len(edges) :]
    edges.to_csv(output, index=False, header=False, sep=" ")


if __name__ == "__main__":
    main()

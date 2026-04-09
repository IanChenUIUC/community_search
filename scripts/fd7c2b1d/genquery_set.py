from pathlib import Path

import click
import networkit as nk
import numpy as np


def write_queries(dir: Path, name: str, queries: list[np.ndarray]):
    with (dir / name).open("wb") as f:
        for query in queries:
            print(f"{query=}")
            np.savetxt(f, query.reshape(1, -1), fmt="%d", delimiter=",")


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--outdir", required=True, type=click.Path())
def main(edgelist, outdir):
    rng = np.random.default_rng(1234)
    graph = nk.graphio.EdgeListReader("\t", 0, continuous=False).read(edgelist)
    nodes = nk.graphtools.randomNodes(graph, 50)

    take_all = [np.array(list(graph.iterNeighbors(node))) for node in nodes]
    take_5 = [rng.choice(nbrs, size=5) for nbrs in take_all]

    dir = Path(outdir)
    dir.mkdir(parents=True, exist_ok=True)
    write_queries(dir, "take_all.txt", take_all)
    write_queries(dir, "take_5.txt", take_5)


if __name__ == "__main__":
    main()

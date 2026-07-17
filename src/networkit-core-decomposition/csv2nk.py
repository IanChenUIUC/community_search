import click

import networkit as nk


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path(exists=False))
def stats(edgelist, output):
    graph = nk.graphio.EdgeListReader(",", 0, "s", continuous=True).read(edgelist)
    nk.writeGraph(graph, output, nk.Format.NetworkitBinary)


if __name__ == "__main__":
    stats()

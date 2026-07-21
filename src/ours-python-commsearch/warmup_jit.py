import click
import numpy as np

from commsearch import Graph, SteinerKCore, ShellStruct


def smol_graph():
    """pair of 3-clique and a 4-clique"""
    indptr = [0, 2, 4, 7, 11, 14, 17, 20]
    indices = [1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 6, 3, 5, 6, 3, 4, 6, 3, 4, 5]
    coreness = np.array([2, 2, 2, 3, 3, 3, 3], dtype=np.uint32)
    return Graph.from_csr(indptr, indices), coreness


@click.command()
def main():
    graph, scores = smol_graph()

    ## pre-compile the ShellStruct
    shell = ShellStruct.build(graph, scores)
    shell.warmup()

    ## pre-compile the ShellStruct
    steiner = SteinerKCore(graph, scores)
    steiner.warmup()


if __name__ == "__main__":
    main()

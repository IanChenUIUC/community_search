import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd


def get_nodes(comm):
    with open(comm) as f:
        nodes = list(map(int, f.readlines()))
        return np.array(nodes)


def read_graph(graph):
    edgelist = pd.read_csv(graph, sep=r"\s+", header=None, names=["u", "v"])
    reversed = edgelist.rename(columns={"u": "v", "v": "u"})
    undirected = pd.concat([edgelist, reversed], ignore_index=True)

    return undirected.drop_duplicates()


def get_nbrh(graph, comm):
    incident = graph.loc[graph["u"].isin(comm), "v"]
    neighborhood = np.union1d(comm, incident.to_numpy())

    return neighborhood


def main():
    basedir = "/u/ianchen3/ianchen3/csearch/community_search"

    logsfmt = f"{basedir}/output/fb6fbaba/py/{{network}}/query.out"
    cmtyfmt = f"{basedir}/output/fb6fbaba/py/{{network}}/{{q}}/kcore_k10.txt"
    graphfmt = f"{basedir}/../data/normalized/{{network}}/network.tsv"

    output = []

    with Path("./scripts/fb6fbaba/networks.txt").open() as networks:
        for network in networks.readlines():
            network = network.strip()
            graph = read_graph(graphfmt.format(network=network))

            try:
                with open(logsfmt.format(network=network)) as logfile:
                    logs = [
                        tuple(map(int, line.strip().split(",")))
                        for line in it.islice(logfile.readlines(), 1, None, 2)
                    ]

                for query, size, visited in logs:
                    comm = cmtyfmt.format(network=network, q=query)

                    nodes = get_nodes(comm)
                    nbrh = get_nbrh(graph, nodes)

                    assert size == len(nodes)
                    output.append([network, query, len(nodes), len(nbrh), visited])

                print(f"Finished analyzing {network}", flush=True)
            except Exception as e:
                print(f"Error in {network}: {e}", flush=True)

    data = pd.DataFrame(
        output,
        columns=["network", "query", "nodes", "nbrhood", "visited"],
    )

    outfile = Path(f"{basedir}/output/fb6fbaba/analysis/sizes.txt")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(outfile, index=None)


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import pandas as pd


def get_nodes(comm):
    with open(comm) as f:
        nodes = list(map(int, f.readlines()))
        if -1 not in nodes:
            nodes.append(-1)
        return np.array(nodes)


def distance(a, b):
    return 1 - len(np.intersect1d(a, b)) / len(np.union1d(a, b))


def main():
    basedir = "/u/ianchen3/ianchen3/csearch/community_search/output"

    outdirs = [
        f"{basedir}/b02f7676/py/{{network}}/{{q}}/kcore_k10.txt",
        f"{basedir}/fb6fbaba/py/{{network}}/{{q}}/kcore_k10.txt",
    ]

    jaccard_distance = []

    with Path("./scripts/fb6fbaba/networks.txt").open() as networks:
        for network in networks.readlines():
            network = network.strip()

            try:
                querypath = Path(f"{basedir}/b02f7676/py/{network}/query_nodes.txt")
                with querypath.open() as queries:
                    queries = [x.strip().split(" ")[0] for x in queries.readlines()]

                for q in queries:
                    comms = [outdir.format(network=network, q=q) for outdir in outdirs]
                    nodes1 = get_nodes(comms[0])
                    nodes2 = get_nodes(comms[1])

                    jaccard_distance.append([network, q, distance(nodes1, nodes2)])

                print(f"Finished analyzing {network}", flush=True)
            except Exception as e:
                print(f"Error in {network}: {e}", flush=True)

    data = pd.DataFrame(jaccard_distance, columns=["network", "q", "distance"])
    outfile = Path(f"{basedir}/fb6fbaba/analysis/summary.txt")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(outfile, index=None)


if __name__ == "__main__":
    main()

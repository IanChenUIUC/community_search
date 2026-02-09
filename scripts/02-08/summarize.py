# Three k-core based community search implementations:
#     - print out edgelist
#     - print out nodelist
#     - networkit implementation
# 1) check whether all the outputs are equivalent
# 2) determine plots for the runtime
# 3) update a repository to ensure reproducibility

from pathlib import Path

import numpy as np
import pandas as pd


def nodes_from_nodelist(comm):
    with open(comm) as f:
        return np.array(list(map(int, f.readlines())))


def nodes_from_edgelist(comm):
    df = pd.read_csv(comm, sep=" ", header=None, names=["u", "v"])
    nodes = np.unique_values(df.to_numpy())
    return nodes[~np.isnan(nodes)]


def distance(a, b):
    return 1 - len(np.intersect1d(a, b)) / len(np.union1d(a, b))


def main():
    basedir = "/u/ianchen3/ianchen3/csearch/output"

    outdirs = [
        f"{basedir}/02-07/{{network}}/{{q}}/kcore_k10.txt",  # CSK (cpp, edgelist)
        f"{basedir}/02-08/{{network}}/{{q}}/kcore_k10.txt",  # CSK (cpp, nodelist)
        f"{basedir}/02-08/{{network}}/nk/{{q}}/kcore_k10.txt",  # CSK (python)
    ]

    # pairwise jaccard distance between 1-2, 1-3, 2-3
    jaccard_distance = []

    with Path("./scripts/02-08/networks.txt").open() as networks:
        for network in networks.readlines():
            network = network.strip()

            try:
                with Path(
                    f"{basedir}/02-07/{network}/query_nodes.txt"
                ).open() as queries:
                    queries = [x.strip().split(" ")[0] for x in queries.readlines()]

                for q in queries:
                    comms = [outdir.format(network=network, q=q) for outdir in outdirs]
                    nodes1 = nodes_from_edgelist(comms[0])
                    nodes2 = nodes_from_nodelist(comms[1])
                    nodes3 = nodes_from_nodelist(comms[2])

                    jaccard_distance.append(
                        [
                            network,
                            q,
                            distance(nodes1, nodes2),
                            distance(nodes1, nodes3),
                            distance(nodes2, nodes3),
                        ]
                    )

                print(f"Finished analyzing {network}", flush=True)
            except Exception:
                print(f"Error in {network}", flush=True)

    data = pd.DataFrame(jaccard_distance, columns=["network", "q", "d12", "d13", "d23"])
    outfile = Path(f"{basedir}/02-08/analysis/summary.txt")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(outfile, index=None)


if __name__ == "__main__":
    main()

import argparse
import os

import numpy as np
import pandas as pd
from graph_tool.collection import ns

# ---------------------------------------------------------------------------
# Networks to fetch.
# ---------------------------------------------------------------------------
NETWORKS = [
    ("soc_net_comms/friendster", "friendster"),
    ("twitter_social", "twitter_social"),
    ("dbpedia_link", "dbpedia_link"),
    ("microsoft_concept", "microsoft_concept"),
    ("wikipedia_link/en", "wikipedia_link"),
    ("bitcoin", "bitcoin"),
    ("livejournal", "livejournal"),
]

OUTPUT_DIR = "/u/ianchen3/community_search/input"


def clean_edges(g):
    E = g.get_edges()[:, :2].astype(np.int64, copy=False)
    E.sort(axis=1)

    E = E[E[:, 0] != E[:, 1]]
    uniq, inv = np.unique(E, return_inverse=True)
    E = inv.reshape(-1, 2)
    E = np.unique(E, axis=0)
    return E, uniq.size


def write_csv(E, path):
    pd.DataFrame(E, columns=["source", "target"]).to_csv(path, index=False)


def process(name, dataset):
    print(f"[{dataset}] downloading {name}...", flush=True)
    g = ns[name]

    print(f"[{dataset}] cleaning {name}...", flush=True)
    E, n_nodes = clean_edges(g)

    print(f"[{dataset}] writing {name}...", flush=True)
    E, n_nodes = clean_edges(g)

    path = os.path.join(OUTPUT_DIR, f"{dataset}.csv")
    write_csv(E, path)

    print(f"[{dataset}] wrote {len(E)} edges, {n_nodes} nodes -> {path}", flush=True)


def main():
    known = [d for _, d in NETWORKS]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", metavar="DATASET",
                        help=f"any of {', '.join(known)} (default: all of them)")
    args = parser.parse_args()

    unknown = sorted(set(args.datasets) - set(known))
    if unknown:
        parser.error(f"unknown dataset(s) {', '.join(unknown)}; choose from {', '.join(known)}")
    wanted = args.datasets or known

    for name, dataset in NETWORKS:
        if dataset in wanted:
            process(name, dataset)


if __name__ == "__main__":
    main()

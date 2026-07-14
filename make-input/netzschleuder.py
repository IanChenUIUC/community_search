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

OUTPUT_DIR = "../input"


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


def process(name, path):
    print(f"[{name}] downloading...")
    g = ns[name]

    E, n_nodes = clean_edges(g)

    write_csv(E, path)
    print(f"[{name}] wrote {len(E)} edges, {n_nodes} nodes -> {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, path in NETWORKS:
        process(name, path)


if __name__ == "__main__":
    main()

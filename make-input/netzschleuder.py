import os

import numpy as np
import pandas as pd
from graph_tool.collection import ns

# ---------------------------------------------------------------------------
# Networks to fetch.
# ---------------------------------------------------------------------------
NETWORKS = [
    "soc_net_comms/friendster",
    "twitter_social",
    "dbpedia_link",
    "microsoft_concept",
    "wikipedia_link/en",
    "bitcoin",
    "livejournal",
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


def process(name):
    print(f"[{name}] downloading...")
    g = ns[name]

    E, n_nodes = clean_edges(g)

    out = os.path.join(OUTPUT_DIR, f"{name.replace('/', '_')}.csv")
    write_csv(E, out)
    print(f"[{name}] wrote {len(E)} edges, {n_nodes} nodes -> {out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in NETWORKS:
        process(name)


if __name__ == "__main__":
    main()

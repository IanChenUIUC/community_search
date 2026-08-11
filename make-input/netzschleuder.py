"""Download networks from netzschleuder and write them in the canonical edge-list
format: `../input/<dataset>.csv`.

Needs graph_tool, which is not installable from PyPI, so run it under the system
python rather than a venv:

    /usr/bin/python3 netzschleuder.py livejournal
"""

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


def process(name, dataset):
    print(f"[{dataset}] downloading {name}...")
    g = ns[name]

    E, n_nodes = clean_edges(g)

    path = os.path.join(OUTPUT_DIR, f"{dataset}.csv")
    write_csv(E, path)
    print(f"[{dataset}] wrote {len(E)} edges, {n_nodes} nodes -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=[d for _, d in NETWORKS],
                        choices=[d for _, d in NETWORKS])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, dataset in NETWORKS:
        if dataset in args.datasets:
            process(name, dataset)


if __name__ == "__main__":
    main()

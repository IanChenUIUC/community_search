from collections.abc import Generator

import numpy as np
import pandas as pd


def get_queries(nodelist: str, nodemap: str) -> Generator[tuple[int, int, np.ndarray]]:
    """
    Nodelist has the (not-necessarily contiguous) nodeIDs of the original graph.
    Yields a unique query ID, the minimum k (or 0), and the compactified query nodes
    """

    vertices = np.loadtxt(nodemap, dtype=np.int32)
    indexer = pd.Index(vertices)

    with open(nodelist) as nodefile:
        for qID, line in enumerate(nodefile):
            tokens = line.strip().split(" ")
            if len(tokens) == 1:
                kmin, querystr = "0", tokens[0]
            else:
                kmin, querystr = tokens

            query = np.fromstring(querystr, sep=",", dtype=np.int32)
            compact_query = indexer.get_indexer(query)
            print(f"got {query=} {compact_query=}")
            yield qID, int(kmin), compact_query

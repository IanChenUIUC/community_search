import numpy as np
import pytest

from commsearch import Graph


def graph_from_edges(n: int, edges) -> Graph:
    """Build a symmetric CSR Graph from undirected (u, v) pairs."""
    src, dst = [], []
    for u, v in edges:
        src += [u, v]
        dst += [v, u]
    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    order = np.argsort(src, kind="stable")  # group neighbors by source
    src, dst = src[order], dst[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(src, minlength=n))
    return Graph.from_csr(indptr, dst)


def clique_chain(sizes) -> Graph:
    """Chain of cliques: clique i (size s) is fully connected; consecutive
    cliques are joined by a single bridge edge."""
    offsets = np.cumsum([0, *sizes])
    n = int(offsets[-1])
    edges = []
    for i, s in enumerate(sizes):
        start = int(offsets[i])
        for a in range(start, start + s):
            for b in range(a + 1, start + s):
                edges.append((a, b))
        if i + 1 < len(sizes):
            edges.append((start + s - 1, int(offsets[i + 1])))
    return graph_from_edges(n, edges)


@pytest.fixture
def triangle() -> Graph:
    return clique_chain([3])


@pytest.fixture
def chain_343() -> Graph:
    return clique_chain([3, 4, 3])

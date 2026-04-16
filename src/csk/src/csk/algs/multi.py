from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass

# import numba as nb
import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import DisjointSet

from ..ds.maxheap import MaxHeap


@dataclass(frozen=True)
class MultiSearchOutput:
    queryID: int
    coreness: int
    commID: int
    vertices: np.ndarray


def search(
    graph: sp.csr_array, coreness: np.ndarray, queries: list[np.ndarray]
) -> Generator[MultiSearchOutput]:
    """
    @yields
        the index of the query, a unique community ID, and the set of vertices
        if the communityID is repeated, then the vertex set may be empty
    """

    def _get_row(graph: sp.csr_array, row: np.ndarray):
        return graph.indices[graph.indptr[row] : graph.indptr[row + 1]]

    # make it symmetric
    undirected = graph + graph.T

    heap: MaxHeap = MaxHeap()
    ready: MaxHeap = MaxHeap()
    visited: set[int] = set()
    terminals = defaultdict(lambda: defaultdict(int))
    uf = DisjointSet(np.arange(len(coreness), dtype=np.int32))
    commID = 0

    def add_nbrs(vertex):
        for nbr in _get_row(undirected, vertex):
            if nbr in visited:
                continue
            eval = min(coreness[vertex], coreness[nbr])
            heap.push(eval, (vertex, nbr))

    for qID, query in enumerate(queries):
        for v in query:
            terminals[v][qID] += 1
            if terminals[v][qID] == len(query):
                ready.push(coreness[v], (qID, qID))
                del terminals[v][qID]
            if v not in visited:
                visited.add(v)
                add_nbrs(v)

    while not heap.is_empty():
        k, _ = heap.peek()

        while not heap.is_empty() and heap.peek()[0] == k:
            _, (u, v) = heap.pop()

            if v not in visited:
                uf.merge(u, v)
                visited.add(v)
                add_nbrs(v)

            elif uf[u] != uf[v]:
                # merge smaller into larger
                if len(terminals[uf[u]]) < len(terminals[uf[v]]):
                    u, v = v, u

                root = uf[u]
                for qID, count in terminals[uf[v]].items():
                    terminals[uf[u]][qID] += count
                    if terminals[uf[u]][qID] == len(queries[qID]):
                        ready.push(k, (qID, qID))
                        del terminals[uf[u]][qID]
                del terminals[uf[v]]
                uf.merge(u, v)

                if uf[u] != root:
                    terminals[uf[u]] = terminals[root]
                    del terminals[root]

        curr = dict()
        while not ready.is_empty() and ready.peek()[0] >= k:
            k, (qID, _) = ready.pop()
            repr = queries[qID][0]
            if uf[repr] in curr:
                vertices = np.array([])
                yield MultiSearchOutput(qID, k, curr[uf[repr]], vertices)
            else:
                vertices = np.array(list(uf.subset(repr)))
                yield MultiSearchOutput(qID, k, commID, vertices)
                curr[uf[repr]] = commID
            commID += 1

        # early return for yielded all comms
        if commID == len(queries):
            return

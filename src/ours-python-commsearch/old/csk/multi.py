from collections import defaultdict
from collections.abc import Generator

# import numba as nb
import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import DisjointSet

from ..algs.maxheap import MaxHeap


def get_row(graph: sp.csr_array, row: np.ndarray):
    return graph.indices[graph.indptr[row] : graph.indptr[row + 1]]


def search(
    graph: sp.csr_array, coreness: np.ndarray, queries: list[np.ndarray]
) -> Generator[tuple[int, int, set[int]], None, None]:
    """
    @yields
        the index of the query, a unique community ID, and the set of vertices
        if the communityID is repeated, then the vertex set may be empty
    """

    heap: MaxHeap = MaxHeap()
    ready: MaxHeap = MaxHeap()
    visited: set[int] = set()
    terminals = defaultdict(lambda: defaultdict(int))
    uf = DisjointSet(np.arange(len(coreness), dtype=np.int32))
    commID = 0

    def add_nbrs(vertex):
        for nbr in get_row(graph, vertex):
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
                yield qID, curr[uf[repr]], set()
            else:
                yield qID, commID, uf.subset(repr)
                curr[uf[repr]] = commID
            commID += 1

        # early return for yielded all comms
        if commID == len(queries):
            return


def get_clique_chain(clique_sizes: list[int]):
    assert min(clique_sizes) >= 3

    n = sum(clique_sizes)
    edges = []
    offsets = np.cumsum([0] + clique_sizes[:-1])

    for i, size in enumerate(clique_sizes):
        start = offsets[i]
        for u in range(start, start + size):
            for v in range(u + 1, start + size):
                edges.append([v, u])
                edges.append([u, v])

        if i < len(clique_sizes) - 1:
            u = start + size - 1
            v = offsets[i + 1]
            edges.append([u, v])
            edges.append([v, u])

    edges = np.array(edges).T
    rows, cols, data = edges[0], edges[1], np.ones_like(edges[0], dtype=np.bool_)
    return sp.csr_array((data, (rows, cols)), shape=(n, n))


def main():
    cliques = np.array([3, 4, 3])
    coreness = np.repeat(cliques - 1, cliques)
    graph = get_clique_chain(cliques.tolist())

    queries = [np.array([0]), np.array([2]), np.array([4]), np.array([6])]
    for queryID, commID, comm in search(graph, coreness, queries):
        print(f"{queryID=} {commID=} {comm=}")


if __name__ == "__main__":
    main()

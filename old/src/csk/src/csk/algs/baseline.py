import numba
import numpy as np

from .common import MultiSearchOutput


def search(graph, mapping, cores, qs, kmins):
    n = cores.size

    @numba.njit
    def find_kcore(indptr, indices, cores, q, k):
        if cores[q] < k:
            return np.empty(0, dtype=np.int64)

        visited = np.zeros(n, dtype=np.uint8)
        stack, sidx = np.empty(n, dtype=np.int64), 0
        out, oidx = np.empty(n, dtype=np.int64), 0

        stack[0], sidx = q, sidx + 1
        visited[q] = 1

        while sidx > 0:
            v, sidx = stack[sidx - 1], sidx - 1
            out[oidx], oidx = v, oidx + 1

            for i in range(indptr[v], indptr[v + 1]):
                u = indices[i]
                if visited[u] == 0 and cores[u] >= cores[q]:
                    visited[u] = 1
                    stack[sidx], sidx = u, sidx + 1

        return out[:oidx]

    for qID, (q, kmin) in enumerate(zip(qs, kmins)):
        assert q.size == 1  # this baseline can only handle singleton queries

        q = q[0]
        comm = mapping[find_kcore(graph.indptr, graph.indices, cores, q, kmin)]
        yield MultiSearchOutput(qID, max(cores[q], kmin), qID, comm)

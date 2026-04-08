from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class LeastCommonAncestor:
    tour: np.ndarray
    depths: np.ndarray
    pos: np.ndarray
    rmq: np.ndarray


def get_data(matrix: sp.csr_array | sp.csr_matrix, node: int) -> np.ndarray:
    return matrix.indices[matrix.indptr[node] : matrix.indptr[node + 1]]


def parents_to_tree(parents: np.ndarray) -> sp.csr_matrix:
    n = parents.size
    mask = parents != np.arange(len(parents))

    cols = np.nonzero(mask)[0]
    rows = parents[mask]
    data = np.ones_like(rows, dtype=np.bool_)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


# OR, just give it a set of parents
def build_rmq(parents: np.ndarray) -> LeastCommonAncestor:
    """
    @params
        parents represents a rooted tree
    @returns
        rmq: 2k x int(log_2(2k)) array, representing RMQ of the tour tour
            of the tree, where rmq[i, j] = argmin_{v}( d(v) : v in Tour[i:i+2^j] )
    """
    tree = parents_to_tree(parents)
    num_nodes = len(parents)

    tour: np.ndarray  # the tour tour of the tree
    depths: np.ndarray  # the depths of each node in the tour tour
    pos: np.ndarray  # the first (arbitrary) index where node shows up in tour
    rmq: np.ndarray  # the sparse table representation of the LCA structure

    # NOTE: this can probably be just-in-time compiled to make it faster
    def _tour():
        l_tour, l_depths, l_pos = [], [], np.empty(num_nodes, dtype=np.int32)

        # node, depth, c = counting how many times we visited r
        root = num_nodes - 1
        stack = [(root, 0, 0)]

        while stack:
            u, d, c = stack.pop()

            if c == 0:
                l_pos[u] = len(l_tour)
            l_tour.append(u)
            l_depths.append(d)

            adj = get_data(tree, u)
            if c < len(adj):
                stack.append((u, d, c + 1))
                stack.append((adj[c], d + 1, 0))

        return (
            np.array(l_tour, dtype=np.int32),
            np.array(l_depths, dtype=np.int32),
            l_pos,
        )

    def _sparse_table(tour, depths):
        assert len(tour) == 2 * num_nodes - 1

        m = 2 * num_nodes - 1
        n = int(np.log2(m)) + 1

        rmqT = np.zeros((n, m), dtype=np.int32)
        rmqT[0] = np.arange(m)  # base case: length 1
        for j in range(1, n):
            shift = 1 << (j - 1)
            idx1 = rmqT[j - 1, : m - shift]
            idx2 = rmqT[j - 1, shift:m]
            rmqT[j, : m - shift] = np.where(depths[idx1] < depths[idx2], idx1, idx2)
        rmq = rmqT.T
        return rmq

    tour, depths, pos = _tour()
    rmq = _sparse_table(tour, depths)
    return LeastCommonAncestor(tour, depths, pos, rmq)

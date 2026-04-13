from dataclasses import dataclass
from typing import Self

import numba
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from .common import get_data


@dataclass(frozen=True)
class LeastCommonAncestor:
    tour: npt.NDArray[np.int32]
    depths: npt.NDArray[np.int32]
    pos: npt.NDArray[np.int32]
    rmq: npt.NDArray[np.int32]

    @classmethod
    def build_rmq(cls, tree: sp.csr_array) -> Self:
        """
        Reduces LCA to the Range-Minimum-Query problem via an Euler Tour.
        Solves RMQ using a DP, where rmq[i, j] = argmin_{v}( d(v) : v in Tour[i:i+2^j] )
        This requires O(n log n) space and time to build, supporting O(1) queries.

        @params: parents represents a rooted (and connected) tree
            the root is must be the last index.
        @returns: LCA implementation.
        """

        num_nodes = tree.size
        tour_len = 2 * num_nodes - 1

        tour = np.empty(tour_len, dtype=np.int32)
        depths = np.empty_like(tour)
        pos = np.empty(tree.size, dtype=np.int32)
        rmqT = np.empty((int(np.log2(tour_len)) + 1, tour_len), dtype=np.int32)

        @numba.njit
        def _tour():
            # node, depth, c = counting how many times we visited r
            root = num_nodes - 1
            stack = [(root, 0, 0)]

            for idx in range(tour_len):
                u, d, c = stack.pop()  # if this fails, the tree is likely disconnected
                if c == 0:
                    pos[u] = idx
                tour[idx], depths[idx] = u, d
                depths[idx] = d

                adj = get_data(tree, u)
                if c < len(adj):
                    stack.append((u, d, c + 1))
                    stack.append((adj[c], d + 1, 0))

        @numba.njit
        def _sparse_table():
            n, m = rmqT.shape

            rmqT[0] = np.arange(m, dtype=np.int32)  # base case: length 1
            for j in range(1, n):
                shift = 1 << (j - 1)
                idx1 = rmqT[j - 1, : m - shift]
                idx2 = rmqT[j - 1, shift:m]
                rmqT[j, : m - shift] = np.where(depths[idx1] < depths[idx2], idx1, idx2)

        _tour()
        _sparse_table()
        return cls(tour, depths, pos, rmqT.T)

    def find_lca(self, query: np.ndarray) -> np.int32:
        pos = self.pos[query]
        left, right = np.min(pos), np.max(pos)

        length = right - left + 1
        j = int(np.log2(length))

        i1: np.int32 = self.rmq[left, j]
        i2: np.int32 = self.rmq[right - (1 << j) + 1, j]
        idx = i1 if self.depths[i1] < self.depths[i2] else i2
        return self.tour[idx]

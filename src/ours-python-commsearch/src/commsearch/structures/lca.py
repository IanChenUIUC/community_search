from dataclasses import dataclass
from typing import Self

from math import frexp

import numpy as np
from numba import njit, uint32

from ..graph import NODE_DTYPE

# LCA over a rooted tree, reduced to RMQ over an Euler tour (Bender-Farach-Colton
# sparse table): O(N log N) build, O(1) per query. Node ids and tour positions are
# uint32 -> supports up to ~2^31 tree nodes (same spirit as NODE_DTYPE).


@njit(cache=True)
def _euler_tour(tree_indptr, tree_indices, root, n):
    """Euler tour of the child-CSR tree rooted at ``root`` (``tree_indptr`` int64
    offsets, ``tree_indices`` uint32 child ids). Returns ``(tour, depths, pos)``:
    ``tour``/``depths`` len ``2n-1``, ``pos[v]`` = v's first tour position."""
    length = 2 * n - 1
    tour = np.empty(length, dtype=np.uint32)
    depths = np.empty(length, dtype=np.uint32)
    pos = np.empty(n, dtype=np.uint32)
    st_node = np.empty(n, dtype=np.uint32)
    st_depth = np.empty(n, dtype=np.uint32)
    st_cur = np.empty(n, dtype=np.int64)
    st_node[0], st_depth[0], st_cur[0] = root, 0, 0
    sp, idx = 1, 0
    while sp > 0:
        u, d, c = st_node[sp - 1], st_depth[sp - 1], st_cur[sp - 1]
        if c == 0:
            pos[u] = idx
        tour[idx], depths[idx] = u, d
        idx += 1
        if c < tree_indptr[u + 1] - tree_indptr[u]:
            st_cur[sp - 1] = c + 1
            child = tree_indices[tree_indptr[u] + c]
            st_node[sp], st_depth[sp], st_cur[sp] = child, d + 1, 0
            sp += 1
        else:
            sp -= 1
    return tour, depths, pos


@njit(cache=True)
def _sparse_table(depths):
    """Sparse table over ``depths``: ``rmq[i, j]`` = position of the min-depth
    entry in ``depths[i : i + 2**j]``. Rows filled only where the window fits."""
    m = len(depths)
    logn = frexp(m)[1] - 1  # floor(log2(m))
    rmq = np.empty((m, logn + 1), dtype=np.uint32)
    for i in range(m):
        rmq[i, 0] = i
    for j in range(1, logn + 1):
        half = 1 << (j - 1)
        for i in range(m - (1 << j) + 1):  # only windows [i, i+2^j) that fit
            a, b = rmq[i, j - 1], rmq[i + half, j - 1]
            rmq[i, j] = a if depths[a] < depths[b] else b
    return rmq


@njit
def _lca_one(tour, depths, pos, rmq, nodes, start, end):
    lo = pos[nodes[start]]
    hi = lo
    for i in range(start + 1, end):
        p = pos[nodes[i]]
        if p < lo:
            lo = p
        elif p > hi:
            hi = p
    length = hi - lo + 1
    j = frexp(length)[1] - 1  # floor(log2(length))
    a = rmq[lo, j]
    b = rmq[hi - (1 << j) + 1, j]
    idx = a if depths[a] < depths[b] else b
    return tour[idx]


@njit(cache=True)
def _lca_batch(tour, depths, pos, rmq, q_nodes, q_ptr):
    """LCA of each query's node set. ``q_nodes`` flat, ``q_ptr`` the CSR offsets."""
    out = np.empty(len(q_ptr) - 1, dtype=np.uint32)
    for qi in range(len(q_ptr) - 1):
        out[qi] = _lca_one(tour, depths, pos, rmq, q_nodes, q_ptr[qi], q_ptr[qi + 1])
    return out


@dataclass(frozen=True)
class LCA:
    """O(1) least-common-ancestor over a rooted tree, rebuilt from the tree (never
    persisted). ``tour``/``depths``/``pos``/``rmq`` are uint32."""

    tour: np.ndarray
    depths: np.ndarray
    pos: np.ndarray
    rmq: np.ndarray

    @classmethod
    def build(cls, tree_indptr: np.ndarray, tree_indices: np.ndarray, root: int) -> Self:
        tour, depths, pos = _euler_tour(
            np.ascontiguousarray(tree_indptr, dtype=np.int64),
            np.ascontiguousarray(tree_indices, dtype=NODE_DTYPE),
            uint32(root),
            len(tree_indptr) - 1,
        )
        return cls(tour, depths, pos, _sparse_table(depths))

    def query(self, q_nodes: np.ndarray, q_ptr: np.ndarray) -> np.ndarray:
        return _lca_batch(
            self.tour, self.depths, self.pos, self.rmq,
            np.ascontiguousarray(q_nodes, dtype=NODE_DTYPE),
            np.ascontiguousarray(q_ptr, dtype=np.int64),
        )

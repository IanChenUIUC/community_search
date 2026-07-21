from math import frexp

import numpy as np
from numba import njit
from numba.core import types
from numba.core.extending import overload_method
from numba.experimental import structref

from ..graph import NODE_DTYPE, OFFSET_DTYPE

# LCA over a rooted tree, reduced to RMQ over an Euler tour (Bender-Farach-Colton
# sparse table): O(N log N) build, O(1) per query. `tour` holds tree-node ids
# (NODE_DTYPE); tour positions (`pos`, `rmq`) are signed OFFSET_DTYPE so the
# RMQ window arithmetic (`hi - 2^j + 1`) can never underflow.


@njit(cache=True)
def _euler_tour(tree_csr, root, n):
    """Euler tour of the child-CSR ``tree_csr`` rooted at ``root``. Returns
    ``(tour, depths, pos)``: ``tour``/``depths`` len ``2n-1``, ``pos[v]`` = v's
    first tour position."""
    length = 2 * n - 1
    tour = np.empty(length, dtype=NODE_DTYPE)
    depths = np.empty(length, dtype=np.uint32)
    pos = np.empty(n, dtype=OFFSET_DTYPE)
    st_node = np.empty(n, dtype=NODE_DTYPE)
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
        if c < tree_csr.degree(u):
            st_cur[sp - 1] = c + 1
            child = tree_csr.values[tree_csr.indptr[u] + c]
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
    rmq = np.empty((m, logn + 1), dtype=OFFSET_DTYPE)
    for i in range(m):
        rmq[i, 0] = i
    for j in range(1, logn + 1):
        half = 1 << (j - 1)
        for i in range(m - (1 << j) + 1):  # only windows [i, i+2^j) that fit
            a, b = rmq[i, j - 1], rmq[i + half, j - 1]
            rmq[i, j] = a if depths[a] < depths[b] else b
    return rmq


@structref.register
class LCAType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


@njit(cache=True)
def _ancestor_batch(self, queries):
    out = np.empty(queries.num_rows(), dtype=NODE_DTYPE)
    for qi in range(queries.num_rows()):
        out[qi] = self.ancestor(queries, qi)
    return out


class LCA(structref.StructRefProxy):
    """O(1) least-common-ancestor over a rooted tree, built from its Euler tour +
    RMQ sparse table. A numba ``structref``: usable object-style inside ``@njit``
    kernels (``lca.ancestor(queries, qi)``) and from Python (``ancestor_batch``).
    Rebuilt from the tree via :func:`build_lca` (never persisted)."""

    def ancestor_batch(self, queries):
        """LCA of each query's node set (``queries`` a CSR of tree nodes)."""
        return _ancestor_batch(self, queries)


structref.define_proxy(LCA, LCAType, ["tour", "depths", "pos", "rmq"])


@overload_method(LCAType, "ancestor")
def _ov_ancestor(self, queries, qi):
    def impl(self, queries, qi):
        nodes = queries.neighbors(qi)
        lo = self.pos[nodes[0]]
        hi = lo
        for i in range(1, len(nodes)):
            p = self.pos[nodes[i]]
            if p < lo:
                lo = p
            elif p > hi:
                hi = p
        j = frexp(hi - lo + 1)[1] - 1
        a = self.rmq[lo, j]
        b = self.rmq[hi - (1 << j) + 1, j]
        return self.tour[a if self.depths[a] < self.depths[b] else b]

    return impl


@overload_method(LCAType, "ancestor_batch")
def _ov_ancestor_batch(self, queries):
    def impl(self, queries):
        out = np.empty(queries.num_rows(), dtype=NODE_DTYPE)
        for qi in range(queries.num_rows()):
            out[qi] = self.ancestor(queries, qi)
        return out

    return impl


def build_lca(tree_csr, root):
    """Build the :class:`LCA` structure from a child-CSR tree. A thin Python
    wrapper: the heavy passes (``_euler_tour``/``_sparse_table``) are cached
    kernels; the structref itself is constructed here."""
    tour, depths, pos = _euler_tour(tree_csr, root, tree_csr.num_rows())
    return LCA(tour, depths, pos, _sparse_table(depths))

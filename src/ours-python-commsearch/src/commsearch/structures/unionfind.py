import numba as nb
import numpy as np
from numba import int64
from numba.experimental import jitclass

from ..graph import NODE_DTYPE


def make_unionfind(dtype=NODE_DTYPE):
    """Return a ``UnionFind`` jitclass with full path compression + union-by-size,
    storing its arrays in ``dtype`` (the node-id width). Usable from Python and
    object-style inside ``@njit`` kernels; methods take/return plain ints, so
    callers are dtype-agnostic."""
    nb_ty = nb.from_dtype(np.dtype(dtype))

    @jitclass([("parent", nb_ty[:]), ("size", nb_ty[:])])
    class UnionFind:
        def __init__(self, n):
            self.parent = np.arange(n, dtype=dtype)
            self.size = np.ones(n, dtype=dtype)

        def find(self, x):
            root = x
            while self.parent[root] != root:
                root = self.parent[root]
            while self.parent[x] != root:
                self.parent[x], x = root, self.parent[x]
            return root

        def merge(self, u, v):
            ru, rv = self.find(u), self.find(v)
            if ru == rv:
                return ru
            if self.size[ru] < self.size[rv]:
                ru, rv = rv, ru
            self.parent[rv] = ru
            self.size[ru] += self.size[rv]
            return ru

    return UnionFind


def make_subset_unionfind(dtype=NODE_DTYPE):
    """Return a ``SubsetUnionFind`` jitclass: a :func:`make_unionfind` ``UnionFind``
    (held as a member, not re-implemented) augmented with a circular linked list
    (``nxt``) over each set's members, so ``merge`` stays O(1) and ``subset`` is
    O(size). ``find``/``merge`` delegate to the inner union-find."""
    nb_ty = nb.from_dtype(np.dtype(dtype))
    UnionFind = make_unionfind(dtype)

    @jitclass([("uf", UnionFind.class_type.instance_type), ("nxt", nb_ty[:])])
    class SubsetUnionFind:
        def __init__(self, n):
            self.uf = UnionFind(n)
            self.nxt = np.arange(n, dtype=dtype)

        def find(self, x):
            return self.uf.find(x)

        def merge(self, u, v):
            ru, rv = self.uf.find(u), self.uf.find(v)
            if ru == rv:
                return ru
            root = self.uf.merge(ru, rv)
            # Splicing the two roots' successors fuses the two member cycles into
            # one, independent of which root union-by-size kept.
            self.nxt[ru], self.nxt[rv] = self.nxt[rv], self.nxt[ru]
            return root

        def subset(self, x):
            root = self.uf.find(x)
            out = np.empty(int64(self.uf.size[root]), dtype=dtype)
            out[0] = root
            cur, i = self.nxt[root], 1
            while cur != root:
                out[i] = cur
                cur, i = self.nxt[cur], i + 1
            return out

        def subset_size(self, x):
            return self.uf.size[self.uf.find(x)]

    return SubsetUnionFind


UnionFind = make_unionfind()
SubsetUnionFind = make_subset_unionfind()

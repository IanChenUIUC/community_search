import time

import numpy as np

from commsearch import UnionFind


class RefUF:
    """Naive reference union-find for cross-checking partitions."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _same_partition(uf, ref, n):
    labels_uf = [uf.find(i) for i in range(n)]
    labels_ref = [ref.find(i) for i in range(n)]
    # partitions match iff (i~j in one) == (i~j in the other) for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            assert (labels_uf[i] == labels_uf[j]) == (labels_ref[i] == labels_ref[j])


def test_matches_reference_random():
    rng = np.random.default_rng(0)
    n = 60
    uf, ref = UnionFind(n), RefUF(n)
    for _ in range(120):
        a, b = int(rng.integers(n)), int(rng.integers(n))
        uf.merge(a, b)
        ref.union(a, b)
    _same_partition(uf, ref, n)


def test_union_by_size_root_is_in_larger_set():
    uf = UnionFind(5)
    uf.merge(2, 3)
    uf.merge(2, 4)  # {2,3,4} size 3
    uf.merge(0, 1)  # {0,1} size 2
    root = uf.merge(0, 2)  # attach smaller {0,1} under larger {2,3,4}
    assert root in {2, 3, 4}
    assert uf.find(0) == uf.find(1) == uf.find(2) == uf.find(3) == uf.find(4) == root


def test_find_batch_matches_find():
    rng = np.random.default_rng(1)
    n = 200
    uf = UnionFind(n)
    for _ in range(300):
        uf.merge(int(rng.integers(n)), int(rng.integers(n)))
    xs = np.arange(n, dtype=np.int64)
    batch = uf.find_batch(xs)
    assert batch.tolist() == [uf.find(i) for i in range(n)]


def test_large_batch_runs_quickly():
    n = 1_000_000
    uf = UnionFind(n)
    # chain everything into one set: worst-case depth before compression
    uf.parent[1:] = np.arange(n - 1, dtype=np.int64)
    uf.find_batch(np.array([1], dtype=np.int64))  # warm JIT (compiles _find_batch)
    uf.parent[1:] = np.arange(n - 1, dtype=np.int64)  # restore the chain
    t0 = time.perf_counter()
    roots = uf.find_batch(np.arange(n, dtype=np.int64))
    dt = time.perf_counter() - t0
    assert np.all(roots == 0)
    assert dt < 1.0  # full compression makes this a near-linear single pass

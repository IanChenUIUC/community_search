import numpy as np

from commsearch import SubsetUnionFind


def _reference_sets(n, merges):
    sets = [{i} for i in range(n)]
    pos = list(range(n))  # element -> index in `sets`
    for u, v in merges:
        pu, pv = pos[u], pos[v]
        if pu == pv:
            continue
        sets[pu] |= sets[pv]
        for e in sets[pv]:
            pos[e] = pu
        sets[pv] = set()
    return {frozenset(s) for s in sets if s}


def test_subset_matches_reference():
    rng = np.random.default_rng(3)
    n = 50
    uf = SubsetUnionFind(n)
    merges = [(int(rng.integers(n)), int(rng.integers(n))) for _ in range(80)]
    for u, v in merges:
        uf.merge(u, v)

    got = {frozenset(uf.subset(i).tolist()) for i in range(n)}
    assert got == _reference_sets(n, merges)


def test_subset_contents_and_size():
    uf = SubsetUnionFind(6)
    uf.merge(0, 1)
    uf.merge(1, 2)
    uf.merge(4, 5)
    assert set(uf.subset(0).tolist()) == {0, 1, 2}
    assert set(uf.subset(2).tolist()) == {0, 1, 2}
    assert uf.subset_size(0) == 3
    assert set(uf.subset(4).tolist()) == {4, 5}
    assert uf.subset_size(4) == 2
    assert set(uf.subset(3).tolist()) == {3}
    assert uf.subset_size(3) == 1


def test_subset_size_matches_len():
    rng = np.random.default_rng(4)
    n = 40
    uf = SubsetUnionFind(n)
    for _ in range(60):
        uf.merge(int(rng.integers(n)), int(rng.integers(n)))
    for i in range(n):
        assert uf.subset_size(i) == len(uf.subset(i))

import numpy as np
import pytest

from commsearch.structures.lca import LCA


def _child_csr(parents, root):
    """Child-CSR (parent -> children) from a parent array; root's self-loop skipped."""
    n = len(parents)
    deg = np.zeros(n + 1, dtype=np.int64)
    for v in range(n):
        if v != root:
            deg[parents[v] + 1] += 1
    indptr = np.cumsum(deg)
    indices = np.empty(int(indptr[-1]), dtype=np.uint32)
    cur = indptr[:-1].copy()
    for v in range(n):
        if v != root:
            p = parents[v]
            indices[cur[p]] = v
            cur[p] += 1
    return indptr, indices


def _brute_lca(parents, root, query):
    def ancestors(u):
        chain = []
        while True:
            chain.append(u)
            if u == root:
                return chain
            u = parents[u]

    common = set(ancestors(query[0]))
    for q in query[1:]:
        common &= set(ancestors(q))
    depth = {}
    for v in range(len(parents)):
        depth[v] = len(ancestors(v))
    return max(common, key=lambda v: depth[v])


def _random_tree(n, rng):
    parents = np.empty(n, dtype=np.uint32)
    parents[0] = 0  # root self-loop
    for v in range(1, n):
        parents[v] = rng.integers(0, v)
    return parents, 0


def test_single_node_query():
    parents, root = np.array([0, 0, 1], dtype=np.uint32), 0
    indptr, indices = _child_csr(parents, root)
    lca = LCA.build(indptr, indices, root)
    out = lca.query(np.array([2], dtype=np.uint32), np.array([0, 1], dtype=np.int64))
    assert out[0] == 2


@pytest.mark.parametrize("seed", range(20))
def test_lca_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 40))
    parents, root = _random_tree(n, rng)
    indptr, indices = _child_csr(parents, root)
    lca = LCA.build(indptr, indices, root)

    q_flat, q_ptr = [], [0]
    expected = []
    for _ in range(15):
        size = int(rng.integers(1, min(n, 5) + 1))
        q = rng.choice(n, size=size, replace=True)
        q_flat.extend(int(x) for x in q)
        q_ptr.append(len(q_flat))
        expected.append(_brute_lca(parents, root, q.tolist()))

    out = lca.query(
        np.array(q_flat, dtype=np.uint32), np.array(q_ptr, dtype=np.int64)
    )
    assert out.tolist() == expected

import numpy as np
import pytest

from commsearch import Community, Graph, SteinerKCore


def _graph_from_edges(n, edges):
    src, dst = [], []
    for u, v in edges:
        src += [u, v]
        dst += [v, u]
    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(src, minlength=n))
    return Graph.from_csr(indptr, dst)


def _clique_edges(sizes, bridged):
    offsets = np.cumsum([0, *sizes])
    edges = []
    for i, s in enumerate(sizes):
        start = int(offsets[i])
        for a in range(start, start + s):
            for b in range(a + 1, start + s):
                edges.append((a, b))
        if bridged and i + 1 < len(sizes):
            edges.append((start + s - 1, int(offsets[i + 1])))
    coreness = np.repeat(np.array(sizes) - 1, sizes).astype(np.int64)
    return int(offsets[-1]), edges, coreness


def _clique_chain(sizes):
    n, edges, cor = _clique_edges(sizes, bridged=True)
    return _graph_from_edges(n, edges), cor


def _disjoint_cliques(sizes):
    n, edges, cor = _clique_edges(sizes, bridged=False)
    return _graph_from_edges(n, edges), cor


@pytest.fixture
def chain343():
    g, cor = _clique_chain([3, 4, 3])  # coreness [2,2,2,3,3,3,3,2,2,2]
    return SteinerKCore(g, cor)


def test_single_seed_low_coreness(chain343):
    # seed 0 has coreness 2; the 2-core connects the whole chain
    c = chain343.run([np.array([0])])[0]
    assert c.coreness == 2
    assert set(c.vertices.tolist()) == set(range(10))


def test_single_seed_high_coreness(chain343):
    # seed 4 has coreness 3; the 3-core is just its clique
    c = chain343.run([np.array([4])])[0]
    assert c.coreness == 3
    assert set(c.vertices.tolist()) == {3, 4, 5, 6}


def test_multi_seed(chain343):
    c = chain343.run([np.array([4, 5])])[0]
    assert c.coreness == 3 and set(c.vertices.tolist()) == {3, 4, 5, 6}
    c2 = chain343.run([np.array([0, 6])])[0]
    assert c2.coreness == 2 and set(c2.vertices.tolist()) == set(range(10))


def test_expand_and_query_coreness(chain343):
    eoc = chain343.expand_one_community(np.array([4]))
    assert isinstance(eoc, Community)
    assert eoc.coreness == 3 and set(eoc.vertices.tolist()) == {3, 4, 5, 6}
    assert chain343.query_coreness(np.array([0])) == 2


def test_output_sharing(chain343):
    out = chain343.run([np.array([3]), np.array([4]), np.array([5])])
    assert all(set(c.vertices.tolist()) == {3, 4, 5, 6} for c in out)
    # same community -> the SAME vertices object (one buffer, not recomputed)
    assert out[0].vertices is out[1].vertices
    assert out[1].vertices is out[2].vertices


def test_duplicate_seeds(chain343):
    c = chain343.run([np.array([4, 4, 5])])[0]
    assert c.coreness == 3 and set(c.vertices.tolist()) == {3, 4, 5, 6}


def test_shared_seed_across_queries(chain343):
    out = chain343.run([np.array([4]), np.array([4, 5])])
    assert set(out[0].vertices.tolist()) == {3, 4, 5, 6}
    assert set(out[1].vertices.tolist()) == {3, 4, 5, 6}


def test_disconnected_seeds_raise():
    g, cor = _disjoint_cliques([3, 4])  # {0,1,2} and {3,4,5,6}, no bridge
    sk = SteinerKCore(g, cor)
    with pytest.raises(ValueError):
        sk.run([np.array([0, 3])])


def test_run_parallel_matches_sequential(chain343):
    queries = [np.array([i]) for i in range(10)] * 3  # 30 queries
    seq = chain343.run(queries)
    par = chain343.run_parallel(queries, num_threads=3, max_batch_size=4)
    assert len(par) == len(seq)
    for s, p in zip(seq, par):
        assert s.coreness == p.coreness
        assert set(s.vertices.tolist()) == set(p.vertices.tolist())


def test_run_parallel_short_circuits(chain343):
    assert chain343.run_parallel([]) == []
    par = chain343.run_parallel([np.array([0]), np.array([4])], num_threads=1)
    assert par[0].coreness == 2 and par[1].coreness == 3


def test_warmup_is_classmethod_and_result_transparent(chain343):
    # a classmethod: callable with no instance, for either indptr width
    SteinerKCore.warmup(indptr_dtype=np.dtype(np.uint32))
    SteinerKCore.warmup(indptr_dtype=np.dtype(np.uint64))
    # warming compiles on a throwaway tiny graph -> real results are unchanged
    q = [np.array([0]), np.array([4])]
    before = chain343.run(q)
    SteinerKCore.warmup(indptr_dtype=chain343.graph.indptr.dtype)
    after = chain343.run(q)
    for a, b in zip(before, after):
        assert a.coreness == b.coreness
        assert set(a.vertices.tolist()) == set(b.vertices.tolist())

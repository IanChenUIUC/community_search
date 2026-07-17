import heapq
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from commsearch import Graph, ShellStruct, SteinerKCore

DNC = Path(__file__).parent / "data" / "dnc"


def _graph_from_edges(n, edges):
    src, dst = [], []
    for u, v in edges:
        src += [u, v]
        dst += [v, u]
    order = np.argsort(np.array(src), kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(np.array(src), minlength=n))
    return Graph.from_csr(indptr, np.array(dst)[order])


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
    coreness = np.repeat(np.array(sizes) - 1, sizes).astype(np.uint32)
    return int(offsets[-1]), edges, coreness


def _clique_chain(sizes):
    n, edges, cor = _clique_edges(sizes, bridged=True)
    return _graph_from_edges(n, edges), cor


def _disjoint_cliques(sizes):
    n, edges, cor = _clique_edges(sizes, bridged=False)
    return _graph_from_edges(n, edges), cor


def _core_numbers(g):
    """Reference k-core decomposition (min-degree peeling)."""
    n = g.num_nodes
    d = np.diff(g.indptr).astype(np.int64)
    heap = [(int(d[v]), v) for v in range(n)]
    heapq.heapify(heap)
    removed = np.zeros(n, bool)
    res = np.zeros(n, dtype=np.uint32)
    k = 0
    while heap:
        dv, v = heapq.heappop(heap)
        if removed[v] or dv != d[v]:
            continue
        k = max(k, dv)
        res[v] = k
        removed[v] = True
        for e in range(g.indptr[v], g.indptr[v + 1]):
            w = g.indices[e]
            if not removed[w]:
                d[w] -= 1
                heapq.heappush(heap, (int(d[w]), int(w)))
    return res


@pytest.fixture
def chain343():
    g, cor = _clique_chain([3, 4, 3])  # coreness [2,2,2,3,3,3,3,2,2,2]
    return ShellStruct.build(g, cor)


@pytest.fixture(scope="module")
def dnc():
    g = Graph.load(DNC)
    return g, _core_numbers(g)


def test_single_seed_low_coreness(chain343):
    c = chain343.run([np.array([0])])[0]
    assert c.coreness == 2 and set(c.vertices.tolist()) == set(range(10))


def test_single_seed_high_coreness(chain343):
    c = chain343.run([np.array([4])])[0]
    assert c.coreness == 3 and set(c.vertices.tolist()) == {3, 4, 5, 6}


def test_multi_seed(chain343):
    c = chain343.run([np.array([4, 5])])[0]
    assert c.coreness == 3 and set(c.vertices.tolist()) == {3, 4, 5, 6}
    c2 = chain343.run([np.array([0, 6])])[0]
    assert c2.coreness == 2 and set(c2.vertices.tolist()) == set(range(10))


def test_duplicate_seeds(chain343):
    c = chain343.run([np.array([4, 4, 5])])[0]
    assert c.coreness == 3 and set(c.vertices.tolist()) == {3, 4, 5, 6}


def test_query_coreness(chain343):
    assert chain343.query_coreness(np.array([0])) == 2
    assert chain343.query_coreness(np.array([4, 5])) == 3


def test_output_sharing(chain343):
    out = chain343.run([np.array([3]), np.array([4]), np.array([5])])
    assert all(set(c.vertices.tolist()) == {3, 4, 5, 6} for c in out)
    assert out[0].vertices is out[1].vertices is out[2].vertices


def test_root_is_coreness_zero(chain343):
    assert chain343.node_coreness[chain343.root] == 0
    assert int(np.argmin(chain343.node_coreness)) == chain343.root


def test_cross_component_no_raise():
    # disjoint cliques -> both under the coreness-0 root; no error (unlike Steiner)
    g, cor = _disjoint_cliques([3, 4])
    ss = ShellStruct.build(g, cor)
    c = ss.run([np.array([0, 3])])[0]
    assert c.coreness == 0 and set(c.vertices.tolist()) == set(range(7))


def test_cross_check_steiner_on_dnc(dnc):
    g, cor = dnc
    sk = SteinerKCore(g, cor)
    ss = ShellStruct.build(g, cor)
    rng = np.random.default_rng(0)
    # single seed, or a seed + one of its neighbours -> always connected, so the
    # whole batch runs in one Steiner sweep (no per-query ValueError to skip).
    queries = []
    for _ in range(300):
        v = int(rng.integers(0, g.num_nodes))
        seeds = {v}
        if rng.random() < 0.5:
            nb = g.indices[g.indptr[v] : g.indptr[v + 1]]
            if len(nb):
                seeds.add(int(rng.choice(nb)))
        queries.append(np.array(sorted(seeds)))

    for sc, cc in zip(sk.run(queries), ss.run(queries)):
        assert sc.coreness == cc.coreness
        assert set(sc.vertices.tolist()) == set(cc.vertices.tolist())


def test_save_load_roundtrip(tmp_path, dnc):
    g, cor = dnc
    ss = ShellStruct.build(g, cor)
    cp, tp = tmp_path / "s.components.feather", tmp_path / "s.tree.feather"
    ss.save(cp, tp)
    ss2 = ShellStruct.load(cp, tp)

    for f in ("assign", "node_coreness"):
        assert np.array_equal(getattr(ss, f), getattr(ss2, f)), f
    for f in ("node_vertices", "tree"):
        assert np.array_equal(getattr(ss, f).indptr, getattr(ss2, f).indptr), f + ".indptr"
        assert np.array_equal(getattr(ss, f).values, getattr(ss2, f).values), f + ".values"
    assert ss.root == ss2.root

    rng = np.random.default_rng(1)
    qs = [np.array([int(rng.integers(0, g.num_nodes))]) for _ in range(50)]
    for a, b in zip(ss.run(qs), ss2.run(qs)):
        assert a.coreness == b.coreness
        assert set(a.vertices.tolist()) == set(b.vertices.tolist())


def test_saved_schema_is_icebug(tmp_path, chain343):
    cp, tp = tmp_path / "s.components.feather", tmp_path / "s.tree.feather"
    chain343.save(cp, tp)
    tree = pa.ipc.open_file(pa.memory_map(str(tp), "r")).read_all()
    comp = pa.ipc.open_file(pa.memory_map(str(cp), "r")).read_all()
    assert comp.schema.field("assignment").type == pa.uint64()
    assert tree.schema.field("coreness").type == pa.uint64()
    assert tree.schema.field("vertices").type == pa.large_list(pa.uint64())
    assert tree.schema.field("csr_indptr").type == pa.uint64()
    assert tree.schema.field("csr_indices").type == pa.uint64()
    num = len(chain343.node_coreness)
    assert all(len(tree.column(c)) == num for c in tree.column_names)  # length-matched
    # trailing null on indices, dropped last offset on indptr
    assert tree.column("csr_indices")[-1].as_py() is None

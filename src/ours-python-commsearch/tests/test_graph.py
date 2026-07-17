from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from commsearch import Graph

DNC = Path(__file__).parent / "data" / "dnc"


def test_basic_counts(triangle):
    assert triangle.num_nodes == 3
    assert triangle.num_edges == 3  # a triangle has 3 undirected edges
    assert sorted(triangle.neighbors(0).tolist()) == [1, 2]
    assert sorted(triangle.neighbors(1).tolist()) == [0, 2]


def test_chain_counts(chain_343):
    # cliques of 3,4,3 -> 3+6+3 intra edges + 2 bridges = 14 undirected edges
    assert chain_343.num_nodes == 10
    assert chain_343.num_edges == 14
    assert len(chain_343.indices) == 2 * chain_343.num_edges


def test_neighbors_slice_matches_indptr(chain_343):
    for u in range(chain_343.num_nodes):
        lo, hi = chain_343.indptr[u], chain_343.indptr[u + 1]
        assert np.array_equal(chain_343.neighbors(u), chain_343.indices[lo:hi])


def _write_csr(base, indptr, indices, fmt, indptr_dtype=np.uint32, indices_dtype=np.uint32):
    ip = pa.table({"indptr": np.asarray(indptr, dtype=indptr_dtype)})
    ix = pa.table({"indices": np.asarray(indices, dtype=indices_dtype)})
    for tbl, name in ((ip, "indptr"), (ix, "indices")):
        dst = base.parent / f"{base.name}.{name}.{fmt}"
        if fmt == "feather":
            with pa.ipc.new_file(str(dst), tbl.schema) as writer:
                writer.write_table(tbl)
        else:
            import pyarrow.parquet as pq

            pq.write_table(tbl, str(dst))


@pytest.mark.parametrize("fmt", ["feather", "parquet"])
def test_load_round_trip(tmp_path, chain_343, fmt):
    base = tmp_path / "g"
    _write_csr(base, chain_343.indptr, chain_343.indices, fmt)
    loaded = Graph.load(base, fmt=fmt)
    assert np.array_equal(loaded.indptr, chain_343.indptr)
    assert np.array_equal(loaded.indices, chain_343.indices)
    assert loaded.indptr.dtype == np.uint32 and loaded.indices.dtype == np.uint32


def test_load_accepts_uint64_indptr(tmp_path, chain_343):
    base = tmp_path / "g"
    _write_csr(base, chain_343.indptr, chain_343.indices, "feather", indptr_dtype=np.uint64)
    loaded = Graph.load(base)
    assert loaded.indptr.dtype == np.uint64 and loaded.indices.dtype == np.uint32
    assert np.array_equal(loaded.indices, chain_343.indices)


def test_load_rejects_non_uint32_indices(tmp_path, chain_343):
    base = tmp_path / "g"
    _write_csr(base, chain_343.indptr, chain_343.indices, "feather", indices_dtype=np.uint64)
    with pytest.raises(ValueError, match="indices must be uint32"):
        Graph.load(base)


def test_load_dnc_zero_copy():
    g = Graph.load(DNC)
    assert g.num_nodes == 906
    assert g.num_edges == 10429
    # dtype policy: uint32 columns
    assert g.indptr.dtype == np.uint32 and g.indices.dtype == np.uint32
    # zero-copy: arrays are views over the mmap'd buffers, not owned copies
    assert g.indptr.flags.owndata is False and g.indices.flags.owndata is False
    assert g.indptr.base is not None and g.indices.base is not None

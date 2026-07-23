import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
from numba import njit, uint32
from numba.core import types
from numba.typed import Dict

from .base import (
    Community,
    SelectiveCommunityDetector,
    _assemble_communities,
    _flatten_queries,
)
from .graph import CORE_DTYPE, NODE_DTYPE, Graph, _read_column, _write_column
from .structures.bucketpq import make_bucketpq
from .structures.community import CommunityBuilder
from .structures.csr import NB_CORE, NB_NODE
from .structures.maxheap import make_maxheap
from .structures.unionfind import SubsetUnionFind

# comm_id sentinel for a query whose seeds never connect
_NONE = np.iinfo(NODE_DTYPE).max

_BucketPQ = make_bucketpq(CORE_DTYPE, NODE_DTYPE)  # vertex frontier, keyed by coreness
_ReadyHeap = make_maxheap(NB_CORE, uint32)  # key=coreness, val=qID
_QHIST = types.DictType(uint32, uint32)  # terminals value: qID -> seed count


@njit(cache=True)
def _steiner(gcsr, coreness, queries):
    """Batched multi-set k-core community search over graph adjacency ``gcsr``,
    with ``queries`` a CSR of per-query seed vertices. Returns
    ``(comm_id[Q], comm_cor[C], member_arrays)`` where ``member_arrays`` is a
    ``typed.List`` of ``C`` per-community vertex arrays; ``comm_id[i]`` is the
    community index of query ``i``, or ``UINT32_MAX`` if its seeds never
    connect."""
    n, Q = gcsr.num_rows(), queries.num_rows()
    qlen = np.empty(Q, dtype=np.uint32)
    for i in range(Q):
        qlen[i] = queries.degree(i)

    uf = SubsetUnionFind(n)
    visited = np.zeros(n, dtype=np.bool_)
    comm_id = np.full(Q, _NONE, dtype=NODE_DTYPE)
    vpq, ready_pq = _BucketPQ(n), _ReadyHeap()
    terminals = Dict.empty(NB_NODE, _QHIST)  # component-root -> (qID -> seed count)
    builder = CommunityBuilder()

    def seed(v, q):
        if v not in terminals:
            terminals[v] = Dict.empty(uint32, uint32)
        d = terminals[v]
        cnt = uint32(d[q] + uint32(1)) if q in d else uint32(1)
        d[q] = cnt
        if cnt == qlen[q]:
            ready_pq.push(coreness[v], q)
            d.pop(q)

    def merge_terminals(ru, rv, k):
        has_u, has_v = ru in terminals, rv in terminals
        if (len(terminals[ru]) if has_u else 0) < (len(terminals[rv]) if has_v else 0):
            ru, rv = rv, ru
            has_u, has_v = has_v, has_u
        tu = terminals[ru] if has_u else Dict.empty(uint32, uint32)
        if has_v:
            tv = terminals[rv]
            for q in tv:
                c = uint32(tu[q] + tv[q]) if q in tu else tv[q]
                if c == qlen[q]:
                    ready_pq.push(k, q)
                    if q in tu:
                        tu.pop(q)
                else:
                    tu[q] = c
            terminals.pop(rv)
        if has_u:
            terminals.pop(ru)
        return tu

    def try_merge(u, w, k):
        ru, rv = uf.find(u), uf.find(w)
        if ru == rv:
            return
        if ru in terminals or rv in terminals:
            terminals[uf.merge(u, w)] = merge_terminals(ru, rv, k)
        else:
            uf.merge(u, w)

    def expand(u, k):
        for w in gcsr.neighbors(u):
            cw = coreness[w]
            if cw >= k:
                try_merge(u, w, k)
            if not visited[w] and not vpq.contains(w):
                vpq.insert(min(cw, k), w)

    def finalize(k_next, exhausted):
        # Emit a resolved query.
        curr = Dict.empty(NB_NODE, NB_NODE)
        count = 0
        while len(ready_pq) > 0 and (exhausted or ready_pq.peek_key() > k_next):
            r_res, q = ready_pq.pop()
            root = uf.find(queries.values[queries.indptr[q]])
            if root in curr:
                comm_id[q] = curr[root]
            else:
                idx = builder.append(r_res, uf.subset(root))
                curr[root] = idx
                comm_id[q] = idx
            count += 1
        return count

    for q in range(Q):
        qi = uint32(q)
        for v in queries.neighbors(q):
            seed(v, qi)
            vpq.insert(coreness[v], v)

    nresolved = 0
    while True:
        if len(vpq) > 0:
            k = vpq.peek_key()
            while len(vpq) > 0 and vpq.peek_key() == k:
                _, u = vpq.extract_max()
                visited[u] = True
                expand(u, k)
        exhausted = len(vpq) == 0
        k_next = vpq.peek_key() if not exhausted else 0
        nresolved += finalize(k_next, exhausted)
        if exhausted or nresolved == Q:
            break

    return comm_id, builder.coreness_array(), builder.members


@dataclass
class SteinerKCore(SelectiveCommunityDetector):
    """Batched multi-set k-core community search over a graph with precomputed
    coreness. No index; ``run`` calls an njit kernel."""

    graph: Graph
    coreness: np.ndarray  # per-vertex coreness, len n

    def __post_init__(self):
        self.coreness = np.ascontiguousarray(self.coreness, dtype=CORE_DTYPE)

    def _run(self, queries) -> list[Community]:
        q = _flatten_queries(queries, dedup=True)
        comm_id, comm_cor, member_arrays = _steiner(self.graph.csr, self.coreness, q)
        bad = np.nonzero(comm_id == _NONE)[0]
        if len(bad):
            raise ValueError(f"query {int(bad[0])} seeds are not connected")
        return _assemble_communities(comm_id, comm_cor, member_arrays)

    def _shared_handle(self) -> dict:
        g = self.graph
        src = g.source
        tmpdir = None
        if src is not None and all(str(p).endswith(".feather") for p in src):
            indptr_path, indices_path = src  # re-mmap the real files (no re-save)
        else:
            tmpdir = tempfile.mkdtemp(dir=_shm_dir())
            base = os.path.join(tmpdir, "graph")
            g.save(base, "feather")
            indptr_path = f"{base}.indptr.feather"
            indices_path = f"{base}.indices.feather"
        if tmpdir is None:
            tmpdir = tempfile.mkdtemp(dir=_shm_dir())
        coreness_path = os.path.join(tmpdir, "coreness.feather")
        _write_column(coreness_path, "coreness", self.coreness, "feather")
        return {
            "indptr_path": indptr_path,
            "indices_path": indices_path,
            "coreness": coreness_path,
            "tmpdir": tmpdir,
        }

    @classmethod
    def _from_shared_handle(cls, handle: dict) -> "SteinerKCore":
        # parent already warmed the shared page cache; skip re-scanning per worker
        graph = Graph.load(handle["indptr_path"], handle["indices_path"], warm=False)
        coreness = _read_column(handle["coreness"], "coreness")
        return cls(graph, coreness)

    def _release_handle(self, handle: dict) -> None:
        if handle.get("tmpdir"):
            shutil.rmtree(handle["tmpdir"], ignore_errors=True)


def _shm_dir() -> str | None:
    # /dev/shm is a RAM-backed tmpfs; workers mmap the shared file from there.
    d = "/dev/shm"
    return d if os.path.isdir(d) and os.access(d, os.W_OK) else None

import os
import shutil
import tempfile
from dataclasses import dataclass

import numba as nb
import numpy as np
from numba import njit, uint32
from numba.core import types
from numba.typed import Dict, List

from .base import Community, SelectiveCommunityDetector
from .graph import NODE_DTYPE, Graph, _read_column, _write_column
from .structures.maxheap import make_maxheap
from .structures.unionfind import SubsetUnionFind

NB_NODE = nb.from_dtype(NODE_DTYPE)  # uint32
_UNRESOLVED = np.iinfo(NODE_DTYPE).max  # comm_id sentinel for a query whose seeds never connect

_EdgeHeap = make_maxheap(uint32, nb.types.UniTuple(NB_NODE, 2))  # key=coreness, val=(u,v)
_ReadyHeap = make_maxheap(uint32, uint32)  # key=coreness, val=qID
_INNER = types.DictType(uint32, uint32)  # terminals value: qID -> seed count
_MEMBERS = types.Array(NB_NODE, 1, "C")  # one community's vertices (a subset() output)


@njit(cache=True)
def _steiner(indptr, indices, coreness, q_flat, q_ptr):
    """Batched multi-set k-core community search. Returns
    ``(comm_id[Q], comm_cor[C], member_arrays)`` where ``member_arrays`` is a
    ``typed.List`` of ``C`` per-community vertex arrays; ``comm_id[i]`` is the
    community index of query ``i``, or ``UINT32_MAX`` if its seeds never connect."""
    n, Q = len(indptr) - 1, len(q_ptr) - 1
    qlen = np.empty(Q, dtype=np.uint32)
    for i in range(Q):
        qlen[i] = q_ptr[i + 1] - q_ptr[i]

    uf = SubsetUnionFind(n)
    visited = np.zeros(n, dtype=np.bool_)
    resolved = np.zeros(Q, dtype=np.bool_)
    comm_id = np.full(Q, _UNRESOLVED, dtype=NODE_DTYPE)
    edge_pq, ready_pq = _EdgeHeap(), _ReadyHeap()
    terminals = Dict.empty(NB_NODE, _INNER)  # component-root -> (qID -> seed count)
    member_arrays = List.empty_list(_MEMBERS)  # one array per emitted community
    comm_cor = List.empty_list(uint32)
    ncomm = np.zeros(1, dtype=np.uint32)

    def add_nbrs(v):
        cv = coreness[v]
        for e in range(indptr[v], indptr[v + 1]):
            w = indices[e]
            if not visited[w]:
                cw = coreness[w]
                edge_pq.push(cv if cv < cw else cw, (v, w))

    def seed(v, q):
        if v not in terminals:
            terminals[v] = Dict.empty(uint32, uint32)
        d = terminals[v]
        cnt = uint32(d[q] + uint32(1)) if q in d else uint32(1)
        d[q] = cnt
        if cnt == qlen[q]:
            resolved[q] = True
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
                    resolved[q] = True
                    ready_pq.push(k, q)
                    if q in tu:
                        tu.pop(q)
                else:
                    tu[q] = c
            terminals.pop(rv)
        if has_u:
            terminals.pop(ru)
        return tu

    def process_edge(k, u, v):
        if not visited[v]:
            uf.merge(u, v)
            visited[v] = True
            add_nbrs(v)
            return
        ru, rv = uf.find(u), uf.find(v)
        if ru == rv:
            return
        if ru in terminals or rv in terminals:
            terminals[uf.merge(u, v)] = merge_terminals(ru, rv, k)
        else:
            uf.merge(u, v)

    def finalize(k_next, exhausted):
        # Emit a resolved query only once the edge-PQ's max drops BELOW its
        # coreness: every edge with weight >= that coreness has been processed,
        # so the community is maximal. (When exhausted, nothing is left, so all.)
        curr = Dict.empty(NB_NODE, uint32)
        count = 0
        while len(ready_pq) > 0 and (exhausted or ready_pq.peek_key() > k_next):
            r_res, q = ready_pq.pop()
            root = uf.find(q_flat[q_ptr[q]])
            if root in curr:
                comm_id[q] = curr[root]
            else:
                member_arrays.append(uf.subset(root))
                comm_cor.append(r_res)
                curr[root] = ncomm[0]
                comm_id[q] = ncomm[0]
                ncomm[0] += 1
            count += 1
        return count

    for q in range(Q):
        qi = uint32(q)
        for j in range(q_ptr[q], q_ptr[q + 1]):
            v = q_flat[j]
            seed(v, qi)
            if not visited[v]:
                visited[v] = True
                add_nbrs(v)

    nresolved = 0
    while True:
        if len(edge_pq) > 0:
            k = edge_pq.peek_key()
            while len(edge_pq) > 0 and edge_pq.peek_key() == k:
                kv = edge_pq.pop()
                process_edge(k, kv[1][0], kv[1][1])
        exhausted = len(edge_pq) == 0
        k_next = edge_pq.peek_key() if not exhausted else uint32(0)
        nresolved += finalize(k_next, exhausted)
        if exhausted or nresolved == Q:
            break

    comm_cor_arr = np.empty(len(comm_cor), dtype=np.uint32)
    for i in range(len(comm_cor)):
        comm_cor_arr[i] = comm_cor[i]
    return comm_id, comm_cor_arr, member_arrays


@dataclass
class SteinerKCore(SelectiveCommunityDetector):
    """Batched multi-set k-core community search over a graph with precomputed
    coreness. No index; ``run`` calls an njit kernel."""

    graph: Graph
    coreness: np.ndarray  # per-vertex coreness, len n

    def run(self, queries) -> list[Community]:
        q_flat_list: list[int] = []
        q_ptr = [0]
        for q in queries:
            q_flat_list.extend(dict.fromkeys(int(x) for x in q))
            q_ptr.append(len(q_flat_list))

        comm_id, comm_cor, member_arrays = _steiner(
            np.ascontiguousarray(self.graph.indptr),
            np.ascontiguousarray(self.graph.indices),
            np.ascontiguousarray(self.coreness, dtype=NODE_DTYPE),
            np.asarray(q_flat_list, dtype=NODE_DTYPE),
            np.asarray(q_ptr, dtype=np.int64),
        )

        bad = np.nonzero(comm_id == _UNRESOLVED)[0]
        if len(bad):
            raise ValueError(f"query {int(bad[0])} seeds are not connected")

        # Queries in the same community share ONE vertices array (read-only):
        # mutate/free at your own risk. Materialize to a Python list so repeated
        # indexing yields the same object per community.
        vertices = []
        for a in member_arrays:
            a.setflags(write=False)
            vertices.append(a)
        return [Community(int(comm_cor[c]), vertices[c]) for c in comm_id.tolist()]

    def _shared_handle(self) -> dict:
        g = self.graph
        tmpdir = None
        if g.source is not None and g.source[1] == "feather":
            graph_base, graph_fmt = g.source
        else:
            tmpdir = tempfile.mkdtemp(dir=_shm_dir())
            graph_base, graph_fmt = os.path.join(tmpdir, "graph"), "feather"
            g.save(graph_base, graph_fmt)
        if tmpdir is None:
            tmpdir = tempfile.mkdtemp(dir=_shm_dir())
        coreness_path = os.path.join(tmpdir, "coreness.feather")
        _write_column(coreness_path, "coreness", self.coreness, "feather")
        return {"graph_base": graph_base, "graph_fmt": graph_fmt,
                "coreness": coreness_path, "tmpdir": tmpdir}

    @classmethod
    def _from_shared_handle(cls, handle: dict) -> "SteinerKCore":
        graph = Graph.load(handle["graph_base"], handle["graph_fmt"])
        coreness = _read_column(handle["coreness"], "coreness")
        return cls(graph, coreness)

    def _release_handle(self, handle: dict) -> None:
        if handle.get("tmpdir"):
            shutil.rmtree(handle["tmpdir"], ignore_errors=True)


def _shm_dir() -> str | None:
    # /dev/shm is a RAM-backed tmpfs; workers mmap the shared file from there.
    d = "/dev/shm"
    return d if os.path.isdir(d) and os.access(d, os.W_OK) else None

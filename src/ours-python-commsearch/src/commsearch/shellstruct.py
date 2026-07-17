from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
from numba import njit, uint8, uint32
from numba.core import types
from numba.typed import Dict, List

from .base import Community, SelectiveCommunityDetector
from .graph import NODE_DTYPE, Graph
from .structures.lca import LCA, _lca_one
from .structures.unionfind import UnionFind

NB_NODE = nb.from_dtype(NODE_DTYPE)  # uint32
_SENT = int(np.iinfo(NODE_DTYPE).max)  # "no node yet" marker for node_of / new_id
_MEMBERS = types.Array(NB_NODE, 1, "C")  # one community's vertices


@njit(cache=True)
def _build_shell(indptr, indices, coreness, shell_indptr, shell_nodes):
    """Peel shells in decreasing k, building the component tree. Returns
    ``(parents, node_coreness, assign, num_nodes, root)`` — ``parents`` the tree
    node parent pointers (root self-parented), ``assign[v]`` the leaf node of
    vertex v. Mirrors icebug ``ShellStruct::build``."""
    n = len(coreness)
    uf = UnionFind(n)
    node_of = np.full(n, _SENT, dtype=NODE_DTYPE)  # current uf-root -> its tree node
    new_id = np.full(n, _SENT, dtype=NODE_DTYPE)  # per-shell: uf-root -> node id
    parents = np.full(n + 1, _SENT, dtype=NODE_DTYPE)
    node_coreness = np.empty(n + 1, dtype=NODE_DTYPE)
    assign = np.empty(n, dtype=NODE_DTYPE)
    next_id = 0

    max_k = len(shell_indptr) - 2
    for k in range(max_k, 0, -1):
        lo, hi = shell_indptr[k], shell_indptr[k + 1]
        if lo == hi:
            continue

        # children: higher-k component roots (already have a node) adjacent to V_k
        touched = Dict.empty(NB_NODE, uint8)
        for ii in range(lo, hi):
            u = shell_nodes[ii]
            for e in range(indptr[u], indptr[u + 1]):
                v = indices[e]
                if coreness[v] > k:
                    r = uf.find(v)
                    if node_of[r] != _SENT and r not in touched:
                        touched[r] = uint8(1)

        # merge V_k with its coreness >= k neighbours -> this shell's components
        for ii in range(lo, hi):
            u = shell_nodes[ii]
            for e in range(indptr[u], indptr[u + 1]):
                v = indices[e]
                if coreness[v] >= k:
                    uf.merge(u, v)

        # one new node per resulting component; assign its shell vertices
        for ii in range(lo, hi):
            u = shell_nodes[ii]
            r = uf.find(u)
            if new_id[r] == _SENT:
                new_id[r] = uint32(next_id)
                node_coreness[next_id] = uint32(k)
                next_id += 1
            assign[u] = new_id[r]

        # attach each touched old node as a child of the new node it merged into
        for r_old in touched:
            parents[node_of[r_old]] = new_id[uf.find(r_old)]
        for r_old in touched:  # clear before setting new roots (a winner may reuse r_old)
            node_of[r_old] = _SENT
        for ii in range(lo, hi):
            r = uf.find(shell_nodes[ii])
            nr = new_id[r]
            if nr != _SENT:
                node_of[r] = nr
                new_id[r] = _SENT

    # synthetic coreness-0 root over every remaining forest root + the 0-shell
    root = next_id
    node_coreness[next_id] = 0
    next_id += 1
    for c in range(root):
        if parents[c] == _SENT:
            parents[c] = uint32(root)
    parents[root] = uint32(root)
    for ii in range(shell_indptr[0], shell_indptr[1]):
        assign[shell_nodes[ii]] = uint32(root)

    return parents[:next_id].copy(), node_coreness[:next_id].copy(), assign, next_id, root


@njit(cache=True)
def _query_shell(tour, depths, pos, rmq, tree_indptr, tree_indices,
                 nv_indptr, nv_flat, node_coreness, q_nodes, q_ptr):
    """Per query: LCA of its leaf nodes, then collect the LCA's subtree vertices.
    Returns ``(comm_id[Q], comm_cor[C], member_arrays)`` like ``_steiner``; queries
    sharing an LCA share one vertices array."""
    Q = len(q_ptr) - 1
    N = len(node_coreness)
    comm_id = np.empty(Q, dtype=NODE_DTYPE)
    comm_cor = List.empty_list(uint32)
    member_arrays = List.empty_list(_MEMBERS)
    seen = Dict.empty(NB_NODE, uint32)  # LCA node -> community index
    stack = np.empty(N, dtype=NODE_DTYPE)
    subnodes = np.empty(N, dtype=NODE_DTYPE)

    for qi in range(Q):
        anc = _lca_one(tour, depths, pos, rmq, q_nodes, q_ptr[qi], q_ptr[qi + 1])
        if anc in seen:
            comm_id[qi] = seen[anc]
            continue
        stack[0] = anc
        top, cnt, total = 1, 0, 0
        while top > 0:
            top -= 1
            node = stack[top]
            subnodes[cnt] = node
            cnt += 1
            total += nv_indptr[node + 1] - nv_indptr[node]
            for c in range(tree_indptr[node], tree_indptr[node + 1]):
                stack[top] = tree_indices[c]
                top += 1
        out = np.empty(total, dtype=NODE_DTYPE)
        w = 0
        for t in range(cnt):
            node = subnodes[t]
            for j in range(nv_indptr[node], nv_indptr[node + 1]):
                out[w] = nv_flat[j]
                w += 1
        cidx = uint32(len(member_arrays))
        member_arrays.append(out)
        comm_cor.append(node_coreness[anc])
        seen[anc] = cidx
        comm_id[qi] = cidx

    comm_cor_arr = np.empty(len(comm_cor), dtype=NODE_DTYPE)
    for i in range(len(comm_cor)):
        comm_cor_arr[i] = comm_cor[i]
    return comm_id, comm_cor_arr, member_arrays


def _parents_to_child_csr(parents, root, num):
    nodes = np.arange(num, dtype=np.int64)
    children = nodes[nodes != root].astype(NODE_DTYPE)
    par = parents[nodes != root]
    tree_indices = children[np.argsort(par, kind="stable")]
    tree_indptr = np.zeros(num + 1, dtype=np.int64)
    tree_indptr[1:] = np.cumsum(np.bincount(par, minlength=num))
    return tree_indptr, tree_indices


def _group_by_node(assign, num):
    nv_indptr = np.zeros(num + 1, dtype=np.int64)
    nv_indptr[1:] = np.cumsum(np.bincount(assign, minlength=num))
    nv_flat = np.argsort(assign, kind="stable").astype(NODE_DTYPE)
    return nv_indptr, nv_flat


@dataclass
class ShellStruct(SelectiveCommunityDetector):
    """Indexed k-core community search: an offline component tree with ~O(1)
    LCA queries. Mirrors ``../ours-icebug-commsearch`` ``nk.scd.ShellStruct``.
    Build with :meth:`build`; persist with :meth:`save`/:meth:`load`."""

    assign: np.ndarray  # vertex -> leaf tree node
    node_coreness: np.ndarray  # tree node -> coreness
    nv_flat: np.ndarray  # node vertices (CSR values), grouped by node
    nv_indptr: np.ndarray
    tree_indptr: np.ndarray  # tree child-CSR (parent -> children)
    tree_indices: np.ndarray
    lca: LCA
    root: int

    @classmethod
    def build(cls, graph: Graph, coreness: np.ndarray) -> "ShellStruct":
        coreness = np.ascontiguousarray(coreness, dtype=NODE_DTYPE)
        n = graph.num_nodes
        max_k = int(coreness.max()) if n else 0
        shell_indptr = np.zeros(max_k + 2, dtype=np.int64)
        shell_indptr[1:] = np.cumsum(np.bincount(coreness, minlength=max_k + 1))
        shell_nodes = np.argsort(coreness, kind="stable").astype(NODE_DTYPE)

        parents, node_coreness, assign, num, root = _build_shell(
            np.ascontiguousarray(graph.indptr),
            np.ascontiguousarray(graph.indices),
            coreness, shell_indptr, shell_nodes,
        )
        tree_indptr, tree_indices = _parents_to_child_csr(parents, root, num)
        nv_indptr, nv_flat = _group_by_node(assign, num)
        lca = LCA.build(tree_indptr, tree_indices, root)
        return cls(assign, node_coreness, nv_flat, nv_indptr,
                   tree_indptr, tree_indices, lca, int(root))

    def run(self, queries) -> list[Community]:
        q_flat: list[int] = []
        q_ptr = [0]
        for q in queries:
            q_flat.extend(int(x) for x in q)
            q_ptr.append(len(q_flat))
        if not q_flat:
            return []
        q_nodes = self.assign[np.asarray(q_flat, dtype=NODE_DTYPE)]
        comm_id, comm_cor, member_arrays = _query_shell(
            self.lca.tour, self.lca.depths, self.lca.pos, self.lca.rmq,
            self.tree_indptr, self.tree_indices, self.nv_indptr, self.nv_flat,
            self.node_coreness, np.ascontiguousarray(q_nodes, dtype=NODE_DTYPE),
            np.asarray(q_ptr, dtype=np.int64),
        )
        # Queries sharing a community share ONE read-only vertices array.
        vertices = []
        for a in member_arrays:
            a.setflags(write=False)
            vertices.append(a)
        return [Community(int(comm_cor[c]), vertices[c]) for c in comm_id.tolist()]

    def query_coreness(self, query) -> int:
        q_nodes = self.assign[np.asarray([int(x) for x in query], dtype=NODE_DTYPE)]
        anc = self.lca.query(q_nodes, np.array([0, len(q_nodes)], dtype=np.int64))[0]
        return int(self.node_coreness[anc])

    def save(self, components_path, tree_path, compression: str = "zstd") -> None:
        """Persist to two Arrow-IPC files, byte-identical to icebug: a components
        file (``assignment``) and a tree file (``coreness``, ``vertices`` as a
        ``large_list``, ``csr_indptr``, ``csr_indices``). All uint64. The tree
        columns are length-matched to ``num_nodes`` (indptr drops its last offset;
        indices gets a trailing null) so they share one table."""
        import pyarrow as pa

        num = len(self.node_coreness)
        components = pa.table({"assignment": pa.array(self.assign.astype(np.uint64))})
        vertices = pa.LargeListArray.from_arrays(
            pa.array(self.nv_indptr, pa.int64()),
            pa.array(self.nv_flat.astype(np.uint64)),
        )
        indices = pa.array(
            list(self.tree_indices.astype(np.uint64)) + [None], type=pa.uint64()
        )
        tree = pa.table({
            "coreness": pa.array(self.node_coreness.astype(np.uint64)),
            "vertices": vertices,
            "csr_indptr": pa.array(self.tree_indptr[:-1].astype(np.uint64)),
            "csr_indices": indices,
        })
        assert all(len(tree.column(c)) == num for c in tree.column_names)
        _write_ipc(components, components_path, compression)
        _write_ipc(tree, tree_path, compression)

    @classmethod
    def load(cls, components_path, tree_path) -> "ShellStruct":
        components = _read_ipc(components_path)
        tree = _read_ipc(tree_path)
        assign = components.column("assignment").combine_chunks().to_numpy().astype(NODE_DTYPE)
        node_coreness = tree.column("coreness").combine_chunks().to_numpy().astype(NODE_DTYPE)

        indptr = tree.column("csr_indptr").combine_chunks().to_numpy()
        indices = tree.column("csr_indices").combine_chunks()
        tree_indices = (
            indices.slice(0, len(indices) - 1)  # undo the trailing-null pad
            .to_numpy(zero_copy_only=False)
            .astype(NODE_DTYPE)
        )
        tree_indptr = np.append(indptr, len(tree_indices)).astype(np.int64)

        vv = tree.column("vertices").combine_chunks()
        off = vv.offsets.to_numpy().astype(np.int64)
        nv_indptr = off - off[0]
        nv_flat = vv.values.to_numpy()[off[0] : off[-1]].astype(NODE_DTYPE)

        root = int(np.argmin(node_coreness))  # the unique coreness-0 node
        lca = LCA.build(tree_indptr, tree_indices, root)
        return cls(assign, node_coreness, nv_flat, nv_indptr,
                   tree_indptr, tree_indices, lca, root)


def _write_ipc(table, path, compression: str) -> None:
    import pyarrow as pa

    codec = None if compression.lower() in ("none", "") else compression.lower()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    opts = pa.ipc.IpcWriteOptions(compression=codec)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema, options=opts) as writer:
            writer.write_table(table)


def _read_ipc(path):
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as src:
        return pa.ipc.open_file(src).read_all()

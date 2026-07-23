from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from numba import njit, uint8, uint32
from numba.typed import Dict

from .base import (
    Community,
    SelectiveCommunityDetector,
    _assemble_communities,
    _flatten_queries,
)
from .graph import CORE_DTYPE, NODE_DTYPE, OFFSET_DTYPE, Graph
from .structures.community import CommunityBuilder
from .structures.csr import CSR, NB_NODE, group_csr
from .structures.lca import LCA, build_lca
from .structures.unionfind import UnionFind

_NONE = int(np.iinfo(NODE_DTYPE).max)  # "no node yet" marker for node_of / new_id


@njit(cache=True)
def _build_shell(gcsr, coreness, shell):
    """Peel shells in decreasing k, building the component tree. Returns
    ``(parents, node_coreness, assign, num_nodes, root)`` — ``parents`` the tree
    node parent pointers (root self-parented), ``assign[v]`` the leaf node of
    vertex v. ``shell`` is the coreness-bucket CSR (row k = the vertices of shell
    V_k). Mirrors icebug ``ShellStruct::build``."""
    n = len(coreness)
    uf = UnionFind(n)
    node_of = np.full(n, _NONE, dtype=NODE_DTYPE)  # current uf-root -> its tree node
    new_id = np.full(n, _NONE, dtype=NODE_DTYPE)  # per-shell: uf-root -> node id
    parents = np.full(n + 1, _NONE, dtype=NODE_DTYPE)
    node_coreness = np.empty(n + 1, dtype=CORE_DTYPE)
    assign = np.empty(n, dtype=NODE_DTYPE)
    next_id = np.zeros(1, dtype=OFFSET_DTYPE)  # 1-elem: closures can't rebind a scalar

    def collect_touched(k, touched):
        # higher-k component roots (already have a node) adjacent to shell V_k
        for u in shell.neighbors(k):
            for v in gcsr.neighbors(u):
                if coreness[v] > k:
                    r = uf.find(v)
                    if node_of[r] != _NONE and r not in touched:
                        touched[r] = uint8(1)

    def shell_components(k):
        # merge V_k with its coreness >= k neighbours -> this shell's components
        for u in shell.neighbors(k):
            for v in gcsr.neighbors(u):
                if coreness[v] >= k:
                    uf.merge(u, v)

    def create_new_nodes(k):
        # one new node per resulting component; assign its shell vertices
        for u in shell.neighbors(k):
            r = uf.find(u)
            if new_id[r] == _NONE:
                new_id[r] = uint32(next_id[0])
                node_coreness[next_id[0]] = uint32(k)
                next_id[0] += 1
            assign[u] = new_id[r]

    def attach_childs(k, touched):
        # each touched old node becomes a child of the new node it merged into
        for r_old in touched:
            parents[node_of[r_old]] = new_id[uf.find(r_old)]
        for r_old in touched:
            node_of[r_old] = _NONE
        for u in shell.neighbors(k):
            r = uf.find(u)
            nr = new_id[r]
            if nr != _NONE:
                node_of[r] = nr
                new_id[r] = _NONE

    for k in range(shell.num_rows() - 1, 0, -1):
        if shell.degree(k) == 0:
            continue
        touched = Dict.empty(NB_NODE, uint8)
        collect_touched(k, touched)
        shell_components(k)
        create_new_nodes(k)
        attach_childs(k, touched)

    # synthetic coreness-0 root over every remaining forest root + the 0-shell
    root = next_id[0]
    node_coreness[root] = 0
    next_id[0] += 1
    for c in range(root):
        if parents[c] == _NONE:
            parents[c] = uint32(root)
    parents[root] = uint32(root)
    for u in shell.neighbors(0):
        assign[u] = uint32(root)

    num = next_id[0]
    return parents[:num].copy(), node_coreness[:num].copy(), assign, num, root


@njit(cache=True)
def _query_shell(tree, node_vertices, node_coreness, lca, queries):
    """Per query (``queries`` a CSR of each query's leaf nodes): LCA of its
    nodes, then collect the LCA's subtree vertices. Returns
    ``(comm_id[Q], comm_cor[C], member_arrays)``; queries sharing an LCA share
    one vertices array."""
    Q = queries.num_rows()
    N = tree.num_rows()
    comm_id = np.empty(Q, dtype=NODE_DTYPE)
    builder = CommunityBuilder()
    stack = np.empty(N, dtype=NODE_DTYPE)
    subnodes = np.empty(N, dtype=NODE_DTYPE)

    for qi in range(Q):
        anc = lca.ancestor(queries, qi)
        found, idx = builder.lookup(anc)
        if found:
            comm_id[qi] = idx
            continue
        stack[0] = anc
        top, cnt, total = 1, 0, 0
        while top > 0:
            top -= 1
            node = stack[top]
            subnodes[cnt] = node
            cnt += 1
            total += node_vertices.degree(node)
            for child in tree.neighbors(node):
                stack[top] = child
                top += 1
        out = np.empty(total, dtype=NODE_DTYPE)
        w = 0
        for t in range(cnt):
            for vtx in node_vertices.neighbors(subnodes[t]):
                out[w] = vtx
                w += 1
        comm_id[qi] = builder.add(anc, node_coreness[anc], out)

    return comm_id, builder.coreness_array(), builder.members


@dataclass
class ShellStruct(SelectiveCommunityDetector):
    """Indexed k-core community search: an offline component tree with ~O(1)
    LCA queries. Mirrors ``../ours-icebug-commsearch`` ``nk.scd.ShellStruct``.
    Build with :meth:`build`; persist with :meth:`save`/:meth:`load`."""

    assign: np.ndarray  # vertex -> leaf tree node
    node_coreness: np.ndarray  # tree node -> coreness
    node_vertices: CSR  # tree node -> its vertices
    tree: CSR  # tree child-CSR (parent -> children)
    lca: LCA
    root: int

    @classmethod
    def build(cls, graph: Graph, coreness: np.ndarray) -> "ShellStruct":
        coreness = np.ascontiguousarray(coreness, dtype=CORE_DTYPE)
        n = graph.num_nodes
        max_k = int(coreness.max()) if n else 0
        shell = group_csr(coreness, max_k + 1)  # row k = vertices of shell V_k

        parents, node_coreness, assign, num, root = _build_shell(
            graph.csr, coreness, shell
        )
        num, root = int(num), int(root)
        tree = group_csr(parents, num, drop=root)
        node_vertices = group_csr(assign, num)
        lca = build_lca(tree, root)
        return cls(assign, node_coreness, node_vertices, tree, lca, root)

    def _run(self, queries) -> list[Community]:
        q = _flatten_queries(queries, dedup=False)
        nodes = CSR(q.indptr, self.assign[q.values])  # map seeds -> leaf tree nodes
        comm_id, comm_cor, member_arrays = _query_shell(
            self.tree, self.node_vertices, self.node_coreness, self.lca, nodes
        )
        return _assemble_communities(comm_id, comm_cor, member_arrays)

    def query_coreness(self, query) -> int:
        nodes = self.assign[np.asarray([int(x) for x in query], dtype=NODE_DTYPE)]
        q = CSR(np.array([0, len(nodes)], dtype=OFFSET_DTYPE), nodes)
        anc = self.lca.ancestor_batch(q)[0]
        return int(self.node_coreness[anc])

    def save(self, components_path, tree_path, compression: str = "zstd") -> None:
        components = pa.table({"assignment": pa.array(self.assign.astype(np.uint64))})
        vertices = pa.LargeListArray.from_arrays(
            pa.array(self.node_vertices.indptr, pa.int64()),
            pa.array(self.node_vertices.values.astype(np.uint64)),
        )
        indices = pa.array(
            list(self.tree.values.astype(np.uint64)) + [None], type=pa.uint64()
        )
        tree = pa.table(
            {
                "coreness": pa.array(self.node_coreness.astype(np.uint64)),
                "vertices": vertices,
                "csr_indptr": pa.array(self.tree.indptr[:-1].astype(np.uint64)),
                "csr_indices": indices,
            }
        )
        _write_ipc(components, components_path, compression)
        _write_ipc(tree, tree_path, compression)

    @classmethod
    def load(cls, components_path, tree_path) -> "ShellStruct":
        components = _read_ipc(components_path)
        tree_tbl = _read_ipc(tree_path)
        assign = (
            components.column("assignment")
            .combine_chunks()
            .to_numpy()
            .astype(NODE_DTYPE)
        )
        node_coreness = (
            tree_tbl.column("coreness").combine_chunks().to_numpy().astype(CORE_DTYPE)
        )

        indptr = tree_tbl.column("csr_indptr").combine_chunks().to_numpy()
        indices = tree_tbl.column("csr_indices").combine_chunks()
        tree_indices = (
            indices.slice(0, len(indices) - 1)  # undo the trailing-null pad
            .to_numpy(zero_copy_only=False)
            .astype(NODE_DTYPE)
        )
        tree_indptr = np.append(indptr, len(tree_indices)).astype(OFFSET_DTYPE)
        tree = CSR(tree_indptr, tree_indices)

        vv = tree_tbl.column("vertices").combine_chunks()
        off = vv.offsets.to_numpy().astype(OFFSET_DTYPE)
        nv_indptr = np.ascontiguousarray(off - off[0])
        nv_flat = vv.values.to_numpy()[off[0] : off[-1]].astype(NODE_DTYPE)
        node_vertices = CSR(nv_indptr, nv_flat)

        root = int(np.argmin(node_coreness))  # the unique coreness-0 node
        lca = build_lca(tree, root)
        return cls(assign, node_coreness, node_vertices, tree, lca, root)


def _write_ipc(table, path, compression: str) -> None:
    codec = None if compression.lower() in ("none", "") else compression.lower()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    opts = pa.ipc.IpcWriteOptions(compression=codec)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema, options=opts) as writer:
            writer.write_table(table)


def _read_ipc(path):
    with pa.memory_map(str(path), "r") as src:
        return pa.ipc.open_file(src).read_all()

import dataclasses
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs

from ..ds.auf import AnchoredUnionFind
from ..ds.rmq import LeastCommonAncestor
from .common import MultiSearchOutput, get_data, is_triu, parents_to_tree


@dataclass(frozen=True)
class ShellStruct:
    # mapping back to the user vertices
    vertices: np.ndarray

    # assign[j] = TreeNode of vertex j
    assign: np.ndarray

    # nodes[i] = bitmask of vertices in node i
    # nodes[i, j] = 1 if vertex j in node i
    # find the vertex set using the indices field of csr
    nodes: sp.csr_array

    # parents[i] = parent of node i
    parents: np.ndarray

    # tree = directed rooted tree representing Shells
    tree: sp.csr_array

    # coreness[i] = core number of TreeNode i
    coreness: np.ndarray

    # O(1) access for least-common ancestor in O(N log N) space
    lca: LeastCommonAncestor

    @classmethod
    def build(cls, graph: sp.csr_array, vs: np.ndarray, ks: np.ndarray) -> Self:
        """
        @params
            graph: sparse representation of adjacency list.
                vertices must be arranged in increasing order of coreness.
                graph must be upper-triangular.
            vertices: mapping back to the original vertex IDs
            cores: the coreness of each vertex
            all arrays use dtype=np.int32
        @returns
            ShellStruct
        """
        assert np.all(ks[:-1] <= ks[1:])
        assert is_triu(graph)

        num_vertices = len(vs)
        assign = np.zeros(num_vertices, dtype=np.int32)
        parents = np.full(num_vertices, -1, dtype=np.int32)
        node_rows, node_cols, node_idx = (
            np.empty_like(parents),
            np.empty_like(parents),
            0,
        )
        node_id, node_cores = 0, []

        auf = AnchoredUnionFind.init(num_vertices)
        counts = np.bincount(ks)
        breaks = np.zeros(len(counts) + 1, dtype=np.int32)
        breaks[1:] = np.cumsum(counts)

        def append_node(vertices, children, k):
            nonlocal node_id, node_idx
            assign[vertices] = node_id
            parents[auf.get_roots(children)] = node_id
            auf.reroot(np.concatenate([vertices, children]), node_id)
            node_rows[node_idx : node_idx + len(vertices)] = node_id
            node_cols[node_idx : node_idx + len(vertices)] = vertices
            node_idx += len(vertices)
            node_cores.append(k)
            node_id += 1

        def process_k_shell(k: int):
            # the closed interval representing the left endpoints
            lk_lo, lk_hi, l_sz = breaks[k], breaks[k + 1] - 1, breaks[k + 1] - breaks[k]
            sub_coo = graph[lk_lo : lk_hi + 1].tocoo()
            rows, cols = sub_coo.row, sub_coo.col.copy()

            mask = cols > lk_hi  # the right endpoints that we need to remap
            rk, rinv = np.unique(auf.find(sub_coo.col[mask]), return_inverse=True)
            cols[mask] = rinv + l_sz
            cols[~mask] -= lk_lo
            size = len(rk) + l_sz

            subg = sp.csr_array((sub_coo.data, (rows, cols)), shape=(size, size))
            _, l_cc = cs.connected_components(subg, directed=False)
            order = np.argsort(l_cc, kind="stable")
            viks = np.split(order, np.cumsum(np.bincount(l_cc))[:-1])
            for vik in viks:
                lik = vik[vik < l_sz] + lk_lo
                rik = rk[vik[vik >= l_sz] - l_sz]
                append_node(lik, rik, k)

        for k in reversed(np.unique(ks)):
            process_k_shell(k)

        roots = np.where(parents[:node_id] == -1)[0]
        if len(roots) > 1:
            parents[roots] = node_id
            node_cores.append(0)
            parents[node_id] = node_id
            node_id += 1
            node_idx += len(roots)
        else:
            parents[roots[0]] = roots[0]

        _nodes = sp.csr_array(
            (np.ones_like(node_rows, dtype=np.bool_), (node_rows, node_cols)),
            shape=(node_id, num_vertices),
        )
        _parents = np.array(parents)[:node_id]
        _tree = parents_to_tree(_parents)
        _node_cores = np.array(node_cores)
        _lca = LeastCommonAncestor.build_rmq(_tree)
        return cls(vs, assign, _nodes, _parents, _tree, _node_cores, _lca)

    def save(self, filename: str) -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        d = {}
        for field_name, value in self.__dict__.items():
            if isinstance(value, sp.csr_array):
                d[f"{field_name}_data"] = value.data
                d[f"{field_name}_indices"] = value.indices
                d[f"{field_name}_indptr"] = value.indptr
                d[f"{field_name}_shape"] = value.shape
            elif isinstance(value, LeastCommonAncestor):
                for k, v in value.__dict__.items():
                    d[f"{field_name}_{k}"] = v
            else:
                d[field_name] = value
        np.savez_compressed(filename, **d)

    @classmethod
    def load(cls, filename: str) -> Self:
        with np.load(filename, allow_pickle=True) as loader:
            d = {}
            for field in dataclasses.fields(ShellStruct):
                name = field.name
                if f"{name}_data" in loader:
                    d[name] = sp.csr_array(
                        (
                            loader[f"{name}_data"],
                            loader[f"{name}_indices"],
                            loader[f"{name}_indptr"],
                        ),
                        shape=tuple(loader[f"{name}_shape"]),
                    )
                elif f"{name}_rmq" in loader:
                    d[name] = LeastCommonAncestor(
                        loader[f"{name}_tour"],
                        loader[f"{name}_depths"],
                        loader[f"{name}_pos"],
                        loader[f"{name}_rmq"],
                    )
                elif name in loader:
                    d[name] = loader[name]
                else:
                    raise AssertionError

        return cls(**d)

    def get_vertices(self, node: np.int32) -> np.ndarray:
        data, stack = [], [node]
        while stack:
            ele = stack.pop()
            stack.extend(get_data(self.tree, ele))
            data.extend(get_data(self.nodes, ele))
        return np.array(data)
        # return self.vertices[np.array(data)]

    def draw_tree(self) -> None:
        """
        Print an ASCII representation of the tree
        NOTE: this is LLM generated code
        """

        parents = self.parents
        children = parents_to_tree(parents)
        nodes = self.nodes
        coreness = self.coreness

        visited = set()
        num_nodes = parents.shape[0]

        def _node_label(node: int) -> str:
            vertices = self.vertices[get_data(nodes, node)].tolist()
            return f"Node {node} (k: {coreness[node]}, vertices: {np.sort(vertices)})"

        def _draw(node, prefix="", is_last=True):
            if node in visited:
                return

            connector = "└─ " if is_last else "├─ "
            line = prefix + connector + _node_label(node)
            print(line)

            visited.add(node)

            node_children = get_data(children, node)
            for i, child in enumerate(node_children):
                last = i == len(node_children) - 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                _draw(child, new_prefix, last)

        # Find the root(s) of the tree.
        roots = [i for i in range(num_nodes) if parents[i] == i]

        for root in roots:
            print(_node_label(root))
            visited.add(root)

            node_children = get_data(children, root)
            for i, child in enumerate(node_children):
                _draw(child, "", i == len(node_children) - 1)


def search(
    shell: ShellStruct, queries: list[np.ndarray]
) -> Generator[MultiSearchOutput]:
    curr: dict[np.int32, int] = dict()
    commID = 0

    for qID, query in enumerate(queries):
        lca = shell.lca.find_lca(shell.assign[query])
        if lca in curr:
            vertices = np.array([])
            yield MultiSearchOutput(qID, shell.coreness[lca], curr[lca], vertices)
        else:
            vertices = shell.get_vertices(lca)
            yield MultiSearchOutput(qID, shell.coreness[lca], commID, vertices)
            curr[lca] = commID
            commID += 1

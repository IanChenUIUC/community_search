import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from auf import AnchoredUnionFind, init_auf
from rmq import LeastCommonAncestor, build_rmq, get_data, parents_to_tree


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


def build_shell(
    graph: sp.csr_array,
    vertices: np.ndarray,
    cores: np.ndarray,
) -> ShellStruct:
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
    assert np.all(cores[:-1] <= cores[1:])

    num_vertices = len(vertices)
    assign = np.zeros(num_vertices, dtype=np.int32)
    parents = np.full(num_vertices + 1, -1, dtype=np.int32)
    node_id, node_rows, node_cols, node_cores = 0, [], [], []

    auf: AnchoredUnionFind = init_auf(num_vertices)
    counts = np.bincount(cores)
    breaks = np.zeros(len(counts) + 1, dtype=np.int32)
    breaks[1:] = np.cumsum(counts)

    def process_k_shell(k: int):
        nonlocal node_id

        vk_lo, vk_hi = breaks[k], breaks[k + 1] - 1  # closed interval
        indptr = graph.indptr[vk_lo : vk_hi + 2]
        indices = graph.indices[graph.indptr[vk_lo] : graph.indptr[vk_hi + 1]]

        left = np.arange(vk_lo, vk_hi + 1, dtype=np.int32)
        _right = auf.find(indices[indices > vk_hi])
        right, rinv = np.unique(_right, return_inverse=True)
        l_size, _r_size, size = len(left), len(right), len(left) + len(right)

        rows = np.repeat(left - vk_lo, np.diff(indptr))
        cols = indices - vk_lo
        cols[indices > vk_hi] = rinv + l_size  # relabeling the edges L->R
        data = np.ones_like(rows, dtype=np.bool_)
        bipart = sp.csr_array((data, (rows, cols)), shape=(size, size), dtype=np.bool_)

        n_cc, l_cc = cs.connected_components(bipart, directed=False)
        for i in range(n_cc):
            vik = np.nonzero(l_cc[:l_size] == i)[0] + vk_lo
            rik = right[l_cc[l_size:] == i]

            assign[vik] = node_id
            parents[auf.get_roots(rik)] = node_id
            auf.reroot(np.concatenate([vik, rik]), node_id)
            node_rows.extend([node_id] * len(vik))
            node_cols.extend(vik)
            node_cores.append(k)
            node_id += 1

    for k in reversed(np.unique(cores)):
        process_k_shell(k)

    roots = np.where(parents[:node_id] == -1)[0]
    if len(roots) > 1:
        parents[roots] = node_id
        node_rows.extend([node_id] * len(roots))
        node_cols.extend(roots)
        node_cores.append(0)
        parents[node_id] = node_id
        node_id += 1
    else:
        parents[roots[0]] = roots[0]

    _nodes = sp.csr_array(
        (np.ones_like(node_rows, dtype=np.bool_), (node_rows, node_cols)),
        shape=(node_id, num_vertices),
    )
    _parents = np.array(parents)[:node_id]
    _tree = parents_to_tree(_parents)
    _node_cores = np.array(node_cores)
    _lca = build_rmq(_tree)
    return ShellStruct(vertices, assign, _nodes, _parents, _tree, _node_cores, _lca)


def save_shell(shell: ShellStruct, filename: str):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    d = {}
    for field_name, value in shell.__dict__.items():
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


def load_shell(filename: str):
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

    return ShellStruct(**d)


def get_vertices(shell: ShellStruct, node: int) -> np.ndarray:
    data, stack = [], [node]
    while stack:
        ele = stack.pop()
        stack.extend(get_data(shell.tree, ele))
        data.extend(get_data(shell.nodes, ele))
    return shell.vertices[np.array(data)]


def draw_tree(shell: ShellStruct):
    """
    Print an ASCII representation of the tree
    """

    parents = shell.parents
    children = parents_to_tree(parents)
    nodes = shell.nodes
    coreness = shell.coreness

    visited = set()
    num_nodes = parents.shape[0]

    def _node_label(node: int) -> str:
        vertices = shell.vertices[get_data(nodes, node)].tolist()
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

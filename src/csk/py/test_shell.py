import itertools as it

import numpy as np
import scipy.sparse as sp
from shell import ShellStruct, build_shell, find_community, get_data


def draw_tree(shell: ShellStruct):
    """
    Print an ASCII representation of the entire tree (or forest)
    """

    parents = shell.parents
    children = shell.children
    nodes = shell.nodes
    coreness = shell.coreness

    visited = set()
    num_nodes = parents.shape[0]

    def _node_label(node: int) -> str:
        vertices = shell.vertices[get_data(nodes, node)].tolist()
        return f"Node {node} (k: {coreness[node]}, vertices: {vertices})"

    def _draw(node, prefix="", is_last=True):
        connector = "└─ " if is_last else "├─ "
        line = prefix + connector + _node_label(node)
        print(line)

        if node in visited:
            # indicate already-printed node to avoid infinite loops
            print(prefix + ("    " if is_last else "│   ") + "[already shown]")
            return

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


def get_clique_chain(clique_sizes: list[int]) -> sp.coo_array:
    assert min(clique_sizes) >= 3

    edges = []
    offsets = np.cumsum([0] + clique_sizes[:-1])

    for i, size in enumerate(clique_sizes):
        start = offsets[i]
        # Intra-clique edges
        for u in range(start, start + size):
            for v in range(u + 1, start + size):
                edges.append([v, u])
                edges.append([u, v])

        # Inter-clique bridge (last node of C_i to first node of C_i+1)
        if i < len(clique_sizes) - 1:
            u = start + size - 1
            v = offsets[i + 1]
            edges.append([u, v])
            edges.append([v, u])

    edges = np.array(edges).T
    num_nodes = np.sum(clique_sizes)
    rows, cols = edges[0], edges[1]
    data = np.ones_like(rows)
    return sp.coo_array((data, (rows, cols)), shape=(num_nodes, num_nodes))


def find_gt_comm(vertices, cliques, query):
    vlist = vertices.tolist()
    qlist = [vlist.index(q) for q in query]

    boundaries = np.cumsum(cliques)
    c_indices = np.searchsorted(boundaries, qlist, side="right")
    c_min, c_max = np.min(c_indices), np.max(c_indices)
    bottleneck_size = np.min(cliques[c_min : c_max + 1])

    left, right = c_min, c_max
    while left > 0 and cliques[left - 1] >= bottleneck_size:
        left -= 1
    while right < len(cliques) - 1 and cliques[right + 1] >= bottleneck_size:
        right += 1

    v_beg = boundaries[left - 1] if left > 0 else 0
    v_end = boundaries[right]
    return vertices[v_beg:v_end]


def assert_permutationally_same(a, b):
    assert a.shape == b.shape
    assert np.all(np.sort(a) == np.sort(b))


def main():
    cliques = [3, 6, 5, 3, 5, 6, 12, 3, 7, 8, 7, 7, 8, 6, 4, 3, 4, 4, 8]
    # print(np.cumsum(cliques))
    # print(cliques)

    num_nodes = np.sum(cliques)
    edges = get_clique_chain(cliques)
    vertices = np.arange(num_nodes)
    cores = np.repeat(cliques, cliques) - 1

    shell = build_shell(edges, vertices, cores)
    draw_tree(shell)

    for v in vertices:
        query = np.array([v])
        et_comm = find_community(shell, query, 0)
        gt_comm = find_gt_comm(vertices, cliques, query)
        assert_permutationally_same(et_comm, gt_comm)

    for vs in it.combinations(vertices, 2):
        query = np.array(list(vs))
        et_comm = find_community(shell, query, 0)
        gt_comm = find_gt_comm(vertices, cliques, query)
        assert_permutationally_same(et_comm, gt_comm)


if __name__ == "__main__":
    main()

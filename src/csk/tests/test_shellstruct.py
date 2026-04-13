import itertools as it

import networkit as nk
import numpy as np
import scipy.sparse as sp
from kcore_shellstruct import AdvancedIndexBuilder


def get_root(indexer):
    node = indexer.v_to_treenode[0]
    while node.parent is not None:
        node = node.parent
    return node


def draw_subtree(root, stream=None):
    def _node_label(node):
        verts = ",".join(str(v) for v in sorted(node.vertex_set))
        return f"CLTreeNode(core={node.core_num}, vertices={{ {verts} }})"

    def _sorted_children(node):
        # deterministic ordering: higher core first, then smallest vertex id
        def key(c):
            minv = min(c.vertex_set) if c.vertex_set else float("inf")
            return (-c.core_num, minv)

        return sorted(node.children, key=key)

    """
    Print an ASCII representation of the subtree rooted at `root`.
    Handles cycles by tracking visited nodes.
    """
    if stream is None:
        import sys

        stream = sys.stdout

    visited = set()

    def _draw(node, prefix="", is_last=True):
        nid = id(node)
        connector = "└─ " if is_last else "├─ "
        line = prefix + connector + _node_label(node)
        print(line, file=stream)
        if nid in visited:
            # indicate already-printed node to avoid infinite loops
            print(
                prefix + ("    " if is_last else "│   ") + "[already shown]",
                file=stream,
            )
            return
        visited.add(nid)

        children = _sorted_children(node)
        for i, child in enumerate(children):
            last = i == len(children) - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            _draw(child, new_prefix, last)

    # root printed without leading connector
    print(_node_label(root), file=stream)
    visited.add(id(root))
    children = _sorted_children(root)
    for i, child in enumerate(children):
        _draw(child, "", i == len(children) - 1)


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


def find_community(indexer, query):
    def ancestors(node):
        """Return the set of all ancestors (including the node itself)."""
        result = set()
        while node is not None:
            result.add(node)
            node = node.parent
        return result

    def find_kcore(queries, k=-1):
        """Returns (resolved_k, sorted_vertices). resolved_k is the actual
        core number used (matters when k=-1)."""
        nodes = []
        for q in queries:
            if q not in indexer.v_to_treenode:
                return k, []
            node = indexer.v_to_treenode[q]
            if k == -1:
                # just stay at the node - this is the highest k possible
                pass
            elif node.core_num < k:
                return k, []
            else:
                # walk up to the first ancestor with core_num <= k
                while node.parent and node.parent.core_num >= k:
                    node = node.parent
            nodes.append(node)

        # intersecting ancestor sets to find LCA
        lca_candidates = ancestors(nodes[0])
        for node in nodes[1:]:
            lca_candidates &= ancestors(node)

        if not lca_candidates:
            return k, []

        # LCA is the deepest common ancestor (highest core_num)
        lca = max(lca_candidates, key=lambda n: n.core_num)
        while lca.parent is not None and lca.core_num == lca.parent.core_num:
            lca = lca.parent

        if k != -1 and lca.core_num < k:
            return k, []

        # collect all vertices from the LCA and its descendants
        vertices = set()
        stack = [lca]
        while stack:
            cur = stack.pop()
            vertices.update(cur.vertex_set)
            stack.extend(cur.children)

        return np.sort(np.array(list(vertices)))

    return find_kcore(query)


def main():
    cliques = [3, 6, 5, 3, 5, 6, 12, 3, 7, 8, 7, 7, 8, 6, 4, 3, 4, 4, 8]
    # print(np.cumsum(cliques))
    # print(cliques)

    num_nodes = np.sum(cliques)
    edges = get_clique_chain(cliques)
    vertices = np.arange(num_nodes)
    # cores = np.repeat(cliques, cliques) - 1

    # graph = nk.GraphFromCoo(edges.coords)
    graph = nk.Graph(num_nodes)
    for u, v in zip(*edges.coords):
        if u < v:
            graph.addEdge(u, v)

    index = AdvancedIndexBuilder(graph)
    index.build()

    draw_subtree(get_root(index))

    for v in vertices:
        query = np.array([v])
        et_comm = find_community(index, query)
        gt_comm = find_gt_comm(vertices, cliques, query)
        # print(et_comm)
        # print(gt_comm)
        assert_permutationally_same(et_comm, gt_comm)

    for vs in it.combinations(vertices, 2):
        query = np.array(list(vs))
        et_comm = find_community(index, query)
        gt_comm = find_gt_comm(vertices, cliques, query)
        assert_permutationally_same(et_comm, gt_comm)


if __name__ == "__main__":
    main()

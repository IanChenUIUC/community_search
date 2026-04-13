import itertools as it

import numpy as np
import scipy.sparse as sp
from rmq import parents_to_tree
from shell import ShellStruct, build_shell, get_data, get_vertices


# def get_clique_chain(clique_sizes: list[int]) -> sp.coo_array:
#     assert min(clique_sizes) >= 3

#     edges = []
#     offsets = np.cumsum([0] + clique_sizes[:-1])

#     for i, size in enumerate(clique_sizes):
#         start = offsets[i]
#         # Intra-clique edges
#         for u in range(start, start + size):
#             for v in range(u + 1, start + size):
#                 edges.append([v, u])
#                 edges.append([u, v])

#         # Inter-clique bridge (last node of C_i to first node of C_i+1)
#         if i < len(clique_sizes) - 1:
#             u = start + size - 1
#             v = offsets[i + 1]
#             edges.append([u, v])
#             edges.append([v, u])

#     edges = np.array(edges).T
#     num_nodes = np.sum(clique_sizes)
#     rows, cols = edges[0], edges[1]
#     data = np.ones_like(rows)
#     return sp.coo_array((data, (rows, cols)), shape=(num_nodes, num_nodes))


# def find_gt_comm(vertices, cliques, query):
#     vlist = vertices.tolist()
#     qlist = [vlist.index(q) for q in query]

#     boundaries = np.cumsum(cliques)
#     c_indices = np.searchsorted(boundaries, qlist, side="right")
#     c_min, c_max = np.min(c_indices), np.max(c_indices)
#     bottleneck_size = np.min(cliques[c_min : c_max + 1])

#     left, right = c_min, c_max
#     while left > 0 and cliques[left - 1] >= bottleneck_size:
#         left -= 1
#     while right < len(cliques) - 1 and cliques[right + 1] >= bottleneck_size:
#         right += 1

#     v_beg = boundaries[left - 1] if left > 0 else 0
#     v_end = boundaries[right]
#     return vertices[v_beg:v_end]


# def assert_permutationally_same(a, b):
#     assert a.shape == b.shape
#     assert np.all(np.sort(a) == np.sort(b))


# def find_community(shell, query):
#     return get_vertices(shell, find_lca(shell, query))


# def main():
#     cliques = [3, 6, 5, 3, 5, 6, 12, 3, 7, 8, 7, 7, 8, 6, 4, 3, 4, 4, 8]
#     # print(np.cumsum(cliques))
#     # print(cliques)

#     num_nodes = np.sum(cliques)
#     edges = get_clique_chain(cliques)
#     vertices = np.arange(num_nodes)
#     cores = np.repeat(cliques, cliques) - 1

#     shell = build_shell(edges, vertices, cores)
#     draw_tree(shell)

#     for v in vertices:
#         query = np.array([v])
#         et_comm = find_community(shell, query)
#         gt_comm = find_gt_comm(vertices, cliques, query)
#         assert_permutationally_same(et_comm, gt_comm)

#     for vs in it.combinations(vertices, 2):
#         query = np.array(list(vs))
#         et_comm = find_community(shell, query)
#         gt_comm = find_gt_comm(vertices, cliques, query)
#         assert_permutationally_same(et_comm, gt_comm)


def main():
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

    cliques = np.array([3, 4, 3, 4, 3, 5, 5, 4, 3, 5, 4])
    # cliques = np.array([3, 4, 3])
    graph = get_clique_chain(cliques.tolist())
    cores = np.repeat(cliques - 1, cliques)

    # Sort nodes by increasing coreness
    order = np.argsort(cores)
    rorder = np.argsort(order)
    cores = cores[order]
    graph.coords = rorder[graph.coords[0]], rorder[graph.coords[1]]

    # num_nodes = np.sum(cliques)
    graph = sp.triu(graph, format="csr")

    shell = build_shell(graph, order, cores)
    draw_tree(shell)


if __name__ == "__main__":
    main()

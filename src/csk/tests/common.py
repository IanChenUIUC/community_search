import numpy as np
import scipy.sparse as sp


def get_clique_chain(clique_sizes: list[int]) -> sp.coo_array:
    """
    returns a 2xN array of all the edges (only one direction is specified)
    the vertices will have contiguous IDs only if contiguous=True, else randomized
    """

    assert min(clique_sizes) >= 3

    num_vertices = np.sum(clique_sizes)
    offsets = np.cumsum([0] + clique_sizes[:-1])
    edges = []

    for i, size in enumerate(clique_sizes):
        start = offsets[i]
        # Intra-clique edges
        for u in range(start, start + size):
            for v in range(u + 1, start + size):
                edges.append([v, u])

        # Inter-clique bridge (last node of C_i to first node of C_i+1)
        if i < len(clique_sizes) - 1:
            u = start + size - 1
            v = offsets[i + 1]
            edges.append([u, v])

    edges = np.array(edges, dtype=np.int32)
    rows = edges[:, 0]
    cols = edges[:, 1]
    data = np.ones_like(rows, dtype=np.bool_)
    return sp.coo_array((data, (rows, cols)), shape=(num_vertices, num_vertices))


def find_gt_comm(cliques: np.ndarray, query: np.ndarray):
    boundaries = np.cumsum(cliques)
    c_indices = np.searchsorted(boundaries, query, side="right")
    c_min, c_max = np.min(c_indices), np.max(c_indices)
    bottleneck_size = np.min(cliques[c_min : c_max + 1])

    left, right = c_min, c_max
    while left > 0 and cliques[left - 1] >= bottleneck_size:
        left -= 1
    while right < len(cliques) - 1 and cliques[right + 1] >= bottleneck_size:
        right += 1

    v_beg = boundaries[left - 1] if left > 0 else 0
    v_end = boundaries[right]
    return np.arange(v_beg, v_end)


def to_triu_adj(
    edges: sp.coo_array, cores: np.ndarray
) -> tuple[sp.csr_array, np.ndarray, np.ndarray]:
    """
    @params
        edges: sparse coo array representation of the edgelist
            assumes that the vertices are from 0 to n-1 (contiguous)
        cores: the core numbers of each vertex (contiguous)
    @returns
        the first element of output is an triu adjlist, sorted by increasing coreness
        the second element is the remapping of nodes from newIDs to origIDs
        the third element is the remapping of nodes from origIDs to newIDs
    """

    n = cores.size
    orig = np.arange(n, dtype=np.int64)
    order = np.lexsort((orig, cores))  # order[newID] -> originalID

    rank_map = np.empty(n, dtype=np.int64)
    rank_map[order] = np.arange(n, dtype=np.int64)

    # remap edges to new ids
    unew = rank_map[edges.row]
    vnew = rank_map[edges.col]

    u = np.minimum(unew, vnew)
    v = np.maximum(unew, vnew)

    data = np.ones_like(u, dtype=np.bool_)
    graph = sp.csr_array((data, (u, v)), shape=(n, n))
    graph.sum_duplicates()
    graph.eliminate_zeros()
    graph.sort_indices()

    return graph, order.copy(), rank_map.copy()


def assert_permutationally_same(a, b):
    assert a.shape == b.shape
    assert np.all(np.sort(a) == np.sort(b))

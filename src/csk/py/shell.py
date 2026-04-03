from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as cs


@dataclass(frozen=True)
class ShellStruct:
    # mapping back to the user vertices
    vertices: np.ndarray

    # assign[j] = node index of vertex j
    assign: np.ndarray

    # nodes[i] = bitmask of vertices in node i
    # nodes[i, j] = 1 if vertex j in node i
    # find the vertex set using the indices field of csr
    nodes: sp.csr_array

    # children[i] = bitmask of children of node i
    # children[i, j] = 1 if node j is child of node i
    # find the children set using the indices field of csr
    children: sp.csr_array

    # parents[i] = parent of node i
    parents: np.ndarray
    # coreness[i] = core number of vertex i
    coreness: np.ndarray

    # rmq[i][j] = ...
    rmq_table: np.ndarray[tuple[int, int], np.dtype[np.int32]]
    rmq_pos: np.ndarray[tuple[int, int], np.dtype[np.int32]]
    rmq_tour: np.ndarray
    rmq_depths: np.ndarray


@dataclass(slots=True)
class AnchoredUnionFind:
    parents: np.ndarray
    sizes: np.ndarray
    roots: np.ndarray

    def __repr__(self):
        return f"""AUF=
{np.array([self.parents, self.sizes, self.roots])}
"""

    def _find(self, xs: np.ndarray):
        curr = np.array(xs, copy=True)
        while not np.all(self.parents[curr] == curr):
            self.parents[curr] = self.parents[self.parents[curr]]
            curr = self.parents[curr]
        return curr

    def get_roots(self, xs: np.ndarray) -> np.ndarray:
        reprs = self._find(xs)
        return self.roots[reprs]

    def merge(self, xs: np.ndarray) -> int:
        reprs = np.unique(self._find(xs))
        merge_to = reprs[np.argmax(self.sizes[reprs])]
        self.parents[reprs] = merge_to
        self.sizes[merge_to] = np.sum(self.sizes[reprs])
        self.roots[reprs] = self.roots[merge_to]
        return merge_to

    def reroot(self, xs: np.ndarray, new_root: int):
        repr = self.merge(xs)
        self.roots[repr] = new_root


def build_rmq(tree: sp.csr_array) -> np.ndarray:
    """
    tree is a csr_matrix representing adjacency list of directed (rooted) tree
    @returns
        rmq: 2k x int(log_2(2k)) array, representing RMQ of the Euler tour
            of the tree, where rmq[i, j] = argmin_{v}( d(v) : v in Tour[i:i+2^j] )
    """
    num_nodes = tree.shape[0]
    euler: np.ndarray  # the Euler tour of the tree
    depths: np.ndarray  # the depths of each node in the Euler tour
    pos: np.ndarray  # the first (arbitrary) index where node shows up in tour
    rmq: np.ndarray  # the sparse table representation of the LCA structure

    # NOTE: this can probably be just-in-time compiled to make it faster
    def _euler():
        nonlocal euler, depths, pos
        l_euler, l_depths, l_pos = [], [], np.empty(num_nodes, dtype=np.int32)

        # node, depth, c = counting how many times we visited r
        root = num_nodes - 1
        stack = [(root, 0, 0)]

        while stack:
            u, d, c = stack.pop()

            if c == 0:
                l_pos[u] = len(l_euler)
            l_euler.append(u)
            l_depths.append(d)

            adj = get_data(tree, u)
            if c < len(adj):
                stack.append((u, d, c + 1))
                stack.append((adj[c], d + 1, 0))

        euler = np.array(l_euler, dtype=np.int32)
        depths = np.array(l_depths, dtype=np.int32)
        pos = l_pos

    def _sparse_table():
        nonlocal euler, depths, rmq
        assert len(euler) == 2 * num_nodes - 1

        m = 2 * num_nodes - 1
        n = int(np.log2(m)) + 1

        rmqT = np.zeros((n, m), dtype=np.int32)
        rmqT[0] = np.arange(m)  # base case: length 1
        for j in range(1, n):
            shift = 1 << (j - 1)
            idx1 = rmqT[j - 1, : m - shift]
            idx2 = rmqT[j - 1, shift:m]
            rmqT[j, : m - shift] = np.where(depths[idx1] < depths[idx2], idx1, idx2)
        rmq = rmqT.T

    _euler()
    _sparse_table()
    return rmq, pos, euler, depths


def build_shell(
    edges: sp.coo_array, vertices: np.ndarray, cores: np.ndarray
) -> ShellStruct:
    """
    @params
        edgelist representation of the adjacency list
        vertices are the list of vertex IDs (for each core)
        cores[i] = coreness of vertices[i]
    @returns
        ShellStruct
    """

    ident = np.argsort(cores)
    rident = np.empty_like(ident)
    rident[ident] = np.arange(len(ident))

    cores = cores[ident]
    vertices = vertices[ident]
    new_vertices = np.arange(len(vertices))
    rows, cols = edges.coords[0], edges.coords[1]

    # evals = np.minimum(cores[rows], cores[cols])
    # e_indices = np.cumsum(np.bincount(evals))
    # ordering = np.argpartition(evals, e_indices[:-1])
    v_indices = np.cumsum(np.bincount(cores))
    rows, cols = rident[rows], rident[cols]

    num_nodes = len(cores)
    graph = sp.coo_array(
        (np.ones_like(rows, dtype=np.bool_), (rows, cols)),
        shape=(num_nodes, num_nodes),
    ).tocsr()

    node_id = 0
    assign = np.empty_like(vertices, dtype=np.int32)
    parents = np.full_like(vertices, -1, dtype=np.int32)
    seen = np.zeros_like(vertices, dtype=np.bool_)
    node_rows, node_cols = [], []
    child_r, child_c = [], []
    coreness = []

    auf = AnchoredUnionFind(
        parents=np.arange(num_nodes, dtype=np.int32),
        sizes=np.ones(num_nodes, dtype=np.int32),
        roots=np.full(num_nodes, -1, dtype=np.int32),
    )

    # 1. iterate bottom up (from highest coreness down)
    for k in reversed(np.unique(cores)):
        # 2. find the vertices of each node in this shell
        v_beg, v_end = v_indices[k - 1], v_indices[k]
        shell = graph[v_beg:v_end, v_beg:v_end]  # FIXME: this is probably a bottleneck
        n_cc, l_cc = cs.connected_components(shell)

        for cc in range(n_cc):
            # 3. create the new node
            nodes = new_vertices[v_beg:v_end][l_cc == cc]
            new_nodes = nodes[~seen[nodes]]
            node_rows.extend(np.full_like(nodes, node_id))
            node_cols.extend(nodes)
            coreness.append(k)
            assign[nodes] = node_id
            auf.reroot(new_nodes, node_id)

            # 4. add links between all adjacent TreeNodes
            adjac = np.unique(graph[nodes].indices)
            adjac = adjac[seen[adjac]]
            child = np.unique(auf.get_roots(adjac))

            if len(child) > 0:
                auf.merge(adjac)
                auf.reroot(np.array([adjac[0], nodes[0]]), node_id)

                child_r.extend(np.full_like(child, node_id))
                child_c.extend(child)
                parents[child] = node_id

            # 5. mark as seen
            seen[new_nodes] = True
            node_id += 1

    # 5. add a super-root to make it a single tree
    coreness.append(0)
    child = np.unique(auf.get_roots(new_vertices))
    child_r.extend(np.full_like(child, node_id))
    child_c.extend(child)
    parents[child] = node_id
    parents[node_id] = node_id
    node_id += 1

    # 6. create the final shellstruct
    shp = (node_id, num_nodes)
    nodes = sp.coo_array((np.ones_like(node_rows), (node_rows, node_cols)), shp).tocsr()
    children = sp.coo_array((np.ones_like(child_r), (child_r, child_c)), shp).tocsr()
    rmq = build_rmq(children)
    return ShellStruct(
        vertices,
        assign,
        nodes,
        children,
        parents[:node_id],
        np.array(coreness),
        *rmq,
    )


def get_data(matrix: sp.csr_array, node: int) -> np.ndarray:
    return matrix.indices[matrix.indptr[node] : matrix.indptr[node + 1]]


def get_indexer(shell) -> pd.Index:
    return pd.Index(shell.vertices)


def find_community(
    shell: ShellStruct, query: np.ndarray, k_min: int, indexer=None
) -> np.ndarray:
    """
    @params
        shell: computed ShellStruct
        query: a set of nodes (original IDs)
        kMin: the minimum coreness of community to return
        indexer: optional precomputed structure to convert from user to internal queries
    @returns
        the node list (original IDs) solving the k-core query problem
        or an empty list, if coreness(Q) <= kmin
    """

    def get_vertices(node: int) -> np.ndarray:
        """all vertices in subtree of node"""
        data, stack = [], [node]
        while stack:
            ele = stack.pop()
            stack.extend(get_data(shell.children, ele))
            data.extend(get_data(shell.nodes, ele))
        return np.array(data)

    if indexer is None:
        indexer = get_indexer(shell)

    query = indexer.get_indexer(query)
    pos = shell.rmq_pos[shell.assign[query]]
    left, right = np.min(pos), np.max(pos)

    j = int(np.log2(right - left + 1))
    idx = shell.rmq_table[[left, right - (1 << j) + 1], j]
    lca = shell.rmq_tour[idx[np.argmin(shell.rmq_depths[idx])]]
    kval = shell.coreness[lca]

    if kval < k_min:
        return np.array([])

    parent = shell.parents[lca]
    while parent != lca and shell.coreness[parent] == kval:
        lca, parent = parent, shell.parents[parent]

    return shell.vertices[get_vertices(lca)]

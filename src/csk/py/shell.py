import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as cs

global start, end


@dataclass(frozen=True)
class ShellStruct:
    # mapping back to the user vertices
    vertices: np.ndarray
    # coreness[i] = core number of vertex i
    coreness: np.ndarray

    # assign[j] = node index of vertex j
    assign: np.ndarray

    # nodes[i] = bitmask of vertices in node i
    # nodes[i, j] = 1 if vertex j in node i
    # find the vertex set using the indices field of csr
    nodes: sp.csr_array

    # parents[i] = parent of node i
    parents: np.ndarray

    # O(1) access for least-common ancestor in O(N log N) space
    rmq_table: np.ndarray[tuple[int, int], np.dtype[np.int32]]
    rmq_pos: np.ndarray[tuple[int, int], np.dtype[np.int32]]
    rmq_tour: np.ndarray
    rmq_depths: np.ndarray


def save_shell(shell: ShellStruct, filename: str):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    d = {}
    for field_name, value in shell.__dict__.items():
        if isinstance(value, sp.csr_array):
            d[f"{field_name}_data"] = value.data
            d[f"{field_name}_indices"] = value.indices
            d[f"{field_name}_indptr"] = value.indptr
            d[f"{field_name}_shape"] = value.shape
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
            elif name in loader:
                d[name] = loader[name]
            else:
                raise AssertionError

    return ShellStruct(**d)


@dataclass(slots=True)
class AnchoredUnionFind:
    parents: np.ndarray
    sizes: np.ndarray
    roots: np.ndarray

    def __repr__(self):
        return f"""AUF=\n\t{np.array([self.parents, self.sizes, self.roots])}"""

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


def process_k_shell(
    k: int,
    start_k: int,
    end_k: int,
    A_upper: sp.csr_matrix,
    uf: AnchoredUnionFind,
    current_node_id: int,
) -> tuple[list[TreeNode], int]:
    """
    Extracts the bipartite subgraph for a given k-shell, finds connected components,
    and updates the hierarchical tree.
    """
    N = A_upper.shape[0]
    Vk_nodes = np.arange(start_k, end_k)

    # 2. Slice the csgraph by rows to find incident edges
    row_start = A_upper.indptr[start_k]
    row_end = A_upper.indptr[end_k]

    # Localize indptr to the slice
    indptr_slice = A_upper.indptr[start_k : end_k + 1] - row_start
    indices_slice = A_upper.indices[row_start:row_end]

    # Separate indices into V_k (internal) and V_{>k} (bipartite)
    mask_same = indices_slice < end_k
    mask_higher = ~mask_same

    # Expand indptr into row indices for easy filtering
    row_indices = np.repeat(np.arange(end_k - start_k), np.diff(indptr_slice))

    # --- Step A: Internal Components (C_{ik}) ---
    same_indices = indices_slice[mask_same] - start_k
    same_rows = row_indices[mask_same]

    # A_same is upper-triangular, but `connected_components` with directed=False handles it natively
    A_same = sp.csr_matrix(
        (np.ones(len(same_indices), dtype=bool), (same_rows, same_indices)),
        shape=(end_k - start_k, end_k - start_k),
    )

    n_components, labels = cs.connected_components(A_same, directed=False)

    # --- Step B: Bipartite Graph to V_{>k} ---
    higher_indices = indices_slice[mask_higher]
    higher_rows = row_indices[mask_higher]

    # 3. Remap targets using UF representatives
    higher_reprs = uf._find(higher_indices)

    # Map from C_{ik} component labels to V_{>k} UF representatives
    label_sources = labels[higher_rows]

    # Rebuild CSR to deduplicate endpoints automatically
    B_data = np.ones(len(label_sources), dtype=bool)
    B = sp.csr_matrix((B_data, (label_sources, higher_reprs)), shape=(n_components, N))
    B.sum_duplicates()

    # --- Step C: Tree Building and UF Update ---
    new_nodes = []
    for i in range(n_components):
        comp_nodes = Vk_nodes[labels == i]

        # Unique UF representatives in V_{>k} that this component connects to
        adj_reprs = B.indices[B.indptr[i] : B.indptr[i + 1]]

        # Translate UF representatives to their actual TreeNode IDs
        adj_roots = uf.roots[adj_reprs]

        node = TreeNode(
            id=current_node_id,
            k=k,
            nodes=comp_nodes.tolist(),
            children=adj_roots.tolist(),
        )
        new_nodes.append(node)

        # 4. Merge the new V_k component nodes WITH the components they attach to,
        # and assign the new TreeNode ID as the root
        xs = np.concatenate([comp_nodes, adj_reprs])
        uf.reroot(xs, current_node_id)

        current_node_id += 1

    return new_nodes, current_node_id


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
        shell_vertices = new_vertices[v_beg:v_end]
        shell = graph[v_beg:v_end, v_beg:v_end]
        n_cc, l_cc = cs.connected_components(shell)

        for cc in range(n_cc):
            # 3. create the new node
            nodes = shell_vertices[l_cc == cc]
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

    global start, end
    end = time.perf_counter()

    print(end - start)

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


def find_lca(shell: ShellStruct, query: np.ndarray, indexer=None) -> np.ndarray:
    if indexer is None:
        indexer = get_indexer(shell)

    query = indexer.get_indexer(query)
    pos = shell.rmq_pos[shell.assign[query]]
    left, right = np.min(pos), np.max(pos)

    j = int(np.log2(right - left + 1))
    idx = shell.rmq_table[[left, right - (1 << j) + 1], j]
    lca = shell.rmq_tour[idx[np.argmin(shell.rmq_depths[idx])]]
    kval = shell.coreness[lca]

    parent = shell.parents[lca]
    while parent != lca and shell.coreness[parent] == kval:
        lca, parent = parent, shell.parents[parent]
    return lca


def get_vertices(shell: ShellStruct, node: int) -> np.ndarray:
    """all vertices in subtree of node"""
    data, stack = [], [node]
    while stack:
        ele = stack.pop()
        stack.extend(get_data(shell.children, ele))
        data.extend(get_data(shell.nodes, ele))
    return shell.vertices[np.array(data)]


@click.group()
def kcore():
    pass


@kcore.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def index(edgelist, output):
    from networkit.centrality import CoreDecomposition
    from networkit.graphio import EdgeListReader

    graph = EdgeListReader("\t", 0).read(edgelist)
    core = CoreDecomposition(graph).run()

    data = [[node, int(core.score(node))] for node in range(graph.numberOfNodes())]
    df = pd.DataFrame(data, columns=["node", "core"])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, header=False, sep="\t")


@kcore.command()
@click.option("--coreslist", required=True, type=click.Path(exists=True))
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def build(coreslist, edgelist, output):
    global start
    start = time.perf_counter()

    df_values = pd.read_csv(coreslist, sep="\\s+", header=None)
    vertices = df_values[0].values
    cores = df_values[1].values
    length = len(vertices)

    df_edges = pd.read_csv(edgelist, sep="\\s+", header=None)
    rows = df_edges[0].values
    cols = df_edges[1].values
    data = np.ones(len(rows), dtype=np.int32)
    graph = sp.coo_array((data, (rows, cols)), shape=(length, length), dtype=np.int32)

    shell = build_shell(graph, vertices, cores)
    save_shell(shell, output)


@kcore.command()
@click.option("--shell_file", required=True, type=click.Path(exists=True))
@click.option("--queries", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def search(shell_file, queries, outputdir):
    shell = load_shell(shell_file)
    indexer = get_indexer(shell)

    mapping: dict[int, Path] = dict()  # mapping from community IDs to files
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)

    with open(queries) as f:
        for spec, querylist in zip(f, f):
            spec = spec.strip().split(" ")
            assert len(spec) == 1 or len(spec) == 2

            name = spec[0]
            kmin = 0 if len(spec) == 1 else int(spec[1])

            query = np.fromstring(querylist, sep=" ", dtype=np.int32)
            lca = find_lca(shell, query, indexer)
            kval = shell.coreness[lca]

            path = output / f"{name}_k{kval}.txt"
            if lca in mapping:
                if path.exists(follow_symlinks=False):
                    path.unlink()
                path.symlink_to(mapping[lca].name)
            elif kval >= kmin:
                mapping[lca] = path
                vertices = get_vertices(shell, lca)
                np.savetxt(path, vertices, fmt="%d")
            else:
                np.savetxt(path, np.array([]))


def main():
    import matplotlib.pyplot as plt

    shell = load_shell("shell.txt.npz")
    sizes = shell.nodes.indptr[1:] - shell.nodes.indptr[:-1]
    print(sizes.shape)

    q = np.quantile(sizes, np.linspace(0, 1, 10))
    print(q)

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(sizes)), np.sort(sizes))
    plt.show()


if __name__ == "__main__":
    # main()
    kcore()

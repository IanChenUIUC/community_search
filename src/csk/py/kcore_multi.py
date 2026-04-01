from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit


@dataclass(slots=True)
class ShellGraph:
    """
    indices: as in csr_matrix.indices for an adjacency matrix
    start_indptr and end_indptr: subslices of each row to study
    start_node:
    """

    # num_nodes == len(start_indptr) == len(end_indptr)
    # num_edges == len(indices)
    start_indptr: np.ndarray
    end_indptr: np.ndarray
    indices: np.ndarray


# @njit
def shell_bfs(graph: ShellGraph, start_node: int, cores: np.ndarray, kmin: int):
    num_nodes = len(graph.start_indptr)
    visited = np.zeros(num_nodes, dtype=np.bool_)
    queue = np.empty(num_nodes, dtype=np.int32)
    bfs_order = np.empty(num_nodes, dtype=np.int32)

    queue[0] = start_node
    visited[start_node] = True

    read_ptr = 0
    write_ptr = 1
    order_ptr = 0

    while read_ptr < write_ptr:
        curr_node = queue[read_ptr]
        read_ptr += 1

        bfs_order[order_ptr] = curr_node
        order_ptr += 1

        beg, end = graph.start_indptr[curr_node], graph.end_indptr[curr_node]
        end = beg + np.searchsorted(-cores[graph.indices[beg:end]], -kmin, side="right")
        graph.start_indptr[curr_node] = end  # remove these edges from the graph

        neighbors = graph.indices[beg:end]
        neighbors = neighbors[~visited[neighbors]]

        visited[neighbors] = True
        queue[write_ptr : write_ptr + len(neighbors)] = neighbors
        write_ptr += len(neighbors)

    return bfs_order[:order_ptr]


def get_community(
    communities: dict[int, np.ndarray], labels: np.ndarray, idx: int
) -> np.ndarray:
    """
    Finds the community of a node given labels and past communities.

    @returns
        an array of the community for the current index.
    """

    output = []
    seen = np.zeros_like(labels, dtype=np.bool_)
    nodes = np.arange(len(labels))[labels == idx]  # get the nodes with idx as label
    for node in nodes:
        if seen[node]:
            continue
        comm = communities.get(node, np.array([node]))
        seen[comm] = True
        output.extend(comm)

    return np.array(output)


def run_queries(
    graph: sp.csr_matrix, queries: list[np.ndarray], cores: np.ndarray
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Runs an work-efficient algorithm for solving multiple queries.
    Yields each community incrementally, which reduces memory allocations.

    Assumes that all vertices are in the range 0 to n-1, sorted by decreasing coreness.
    Assumes that vertices and adjacency are sorted according to decreasing coreness.

    @returns
        the index of the query and its community (may not return queries in order)
    """

    assert graph.has_sorted_indices  # adjacency is sorted order
    assert np.all(cores[:-1] >= cores[1:])  # the vertices are sorted correctly

    # 0. initialize data structures for later use
    subg: ShellGraph = ShellGraph(
        start_indptr=graph.indptr[:-1].copy(),
        end_indptr=graph.indptr[1:].copy(),
        indices=graph.indices.copy(),
    )
    labels = np.full_like(cores, -1, dtype=np.int32)
    visited = np.zeros_like(cores, dtype=np.bool_)
    pqueue: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for idx, query in enumerate(queries):
        pqueue[np.min(cores[query])].append((idx, query))
    communities: dict[int, np.ndarray] = {}

    # 1. iterate in decreasing priority; priority >= core(Q)
    for k in reversed(range(1, np.max(cores) + 1)):
        while pqueue[k]:
            idx, query = pqueue[k].pop()

            # 2. find the reachable nodes in the subgraph
            reach = shell_bfs(subg, query[0], cores, k)
            reach = reach[~visited[reach]]

            # 3. update the labels to the query node
            labels[reach] = idx
            visited[reach] = True
            query = query[~visited[query]]

            # 4. decrement the k if there are still nodes left
            if len(query) > 0:
                pqueue[k - 1].append((idx, query))
                continue

            # 5. print the community if found all nodes
            comms = get_community(communities, labels, idx)
            communities[idx] = comms
            yield idx, comms


def to_graph(
    edges: np.ndarray, cores: dict[int, int]
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    @params
        edges: a 2xN matrix of edges; only one direction of edge should be specified
        cores: the core numbers of each vertex (not necessarily contiguous)
    @returns
        the first element of output is a graph that can be fed into get_community
        the second element is the remapping of nodes from newIDs to origIDs
    """

    codes, uniques = pd.factorize(edges.ravel())

    v_cores = np.array([cores[u] for u in uniques])
    rank_map = np.argsort(np.argsort(-v_cores))
    new_edges = rank_map[codes].reshape(edges.shape)
    original_ids = np.empty_like(uniques)
    original_ids[rank_map] = uniques

    u, v = new_edges[0], new_edges[1]
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])

    num_nodes = len(uniques)
    data = np.ones(len(rows), dtype=np.int32)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    graph.sort_indices()

    return graph, original_ids


def main():
    pass


if __name__ == "__main__":
    main()

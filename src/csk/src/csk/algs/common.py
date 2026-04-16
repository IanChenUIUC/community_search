import numpy as np
import scipy.sparse as sp


def get_data(matrix: sp.csr_array, node: int | np.int32) -> np.ndarray:
    return matrix.indices[matrix.indptr[node] : matrix.indptr[node + 1]]


def parents_to_tree(parents: np.ndarray) -> sp.csr_array:
    n = parents.size
    mask = parents != np.arange(len(parents))

    cols = np.nonzero(mask)[0]
    rows = parents[mask]
    data = np.ones_like(rows, dtype=np.bool_)
    return sp.csr_array((data, (rows, cols)), shape=(n, n))


def is_triu(graph: sp.csr_array):
    coo = graph.tocoo()
    return np.all(coo.row <= coo.col)

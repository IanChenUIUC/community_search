import numpy as np
import scipy.sparse as sp


def get_data(matrix: sp.csr_array, node: int | np.int32) -> np.ndarray:
    return matrix.indices[matrix.indptr[node] : matrix.indptr[node + 1]]

import numpy as np
from numba import njit

from commsearch.structures import CSR, group_csr


def _numpy_csr(rows):
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for i, r in enumerate(rows):
        indptr[i + 1] = indptr[i] + len(r)
    values = np.array([x for r in rows for x in r], dtype=np.uint32)
    return indptr, values


def test_accessors_from_python():
    rows = [[1, 2, 3], [], [7], [4, 5]]
    indptr, values = _numpy_csr(rows)
    csr = CSR(indptr, values)
    assert csr.num_rows() == 4
    for i, r in enumerate(rows):
        assert csr.degree(i) == len(r)
        assert csr.neighbors(i).tolist() == r


def test_wrap_is_zero_copy():
    indptr = np.array([0, 2, 3], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.uint32)
    csr = CSR(indptr, values)
    assert csr.indptr.ctypes.data == indptr.ctypes.data
    assert csr.values.ctypes.data == values.ctypes.data


@njit
def _sum_via_csr(csr):
    s = 0
    for i in range(csr.num_rows()):
        for v in csr.neighbors(i):
            s += v
    return s


def test_used_inside_kernel():
    indptr = np.array([0, 2, 3, 5], dtype=np.int64)
    values = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    assert _sum_via_csr(CSR(indptr, values)) == int(values.sum())


def test_group_csr_buckets_by_key():
    key = np.array([2, 0, 2, 1, 0, 2], dtype=np.uint32)
    csr = group_csr(key, 3)
    assert csr.indptr.dtype == np.int64 and csr.values.dtype == np.uint32
    for k in range(3):
        assert sorted(csr.neighbors(k).tolist()) == np.nonzero(key == k)[0].tolist()


def test_group_csr_drop_excludes_index():
    parents = np.array([3, 3, 4, 4, 4], dtype=np.uint32)  # root=4, self-parented
    csr = group_csr(parents, 5, drop=4)
    assert csr.neighbors(3).tolist() == [0, 1]
    assert csr.neighbors(4).tolist() == [2, 3]  # the root itself is not its own child
    assert 4 not in csr.values.tolist()

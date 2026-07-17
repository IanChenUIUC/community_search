import numba as nb
import numpy as np
from numba.experimental import jitclass

from ..graph import CORE_DTYPE, NODE_DTYPE, OFFSET_DTYPE

# Semantic numba scalar aliases (defined once, imported by the kernels).
NB_NODE = nb.from_dtype(NODE_DTYPE)  # vertex & tree-node ids (uint32)
NB_CORE = nb.from_dtype(CORE_DTYPE)  # coreness / k (uint32)
NB_OFFSET = nb.from_dtype(OFFSET_DTYPE)  # CSR offsets (int64)


def make_csr(offset_dtype=OFFSET_DTYPE, value_dtype=NODE_DTYPE):
    """Return a ``CSR`` jitclass wrapping ``(indptr, values)`` arrays — the
    in-kernel adjacency abstraction (parent->children, node->vertices, graph
    adjacency). Like :func:`make_unionfind`, it's numba's template idiom: one
    concrete class per ``(offset_dtype, value_dtype)`` pair, usable from Python
    and object-style inside ``@njit`` kernels.

    Construction **wraps** the given arrays (no copy); ``neighbors(i)`` returns a
    **view** into ``values`` — callers must not free/mutate the backing arrays
    while the CSR (or a returned neighbor slice) is alive.
    """
    off_ty = nb.from_dtype(np.dtype(offset_dtype))
    val_ty = nb.from_dtype(np.dtype(value_dtype))

    @jitclass([("indptr", off_ty[:]), ("values", val_ty[:])])
    class CSR:
        def __init__(self, indptr, values):
            self.indptr = indptr
            self.values = values

        def num_rows(self):
            return len(self.indptr) - 1

        def degree(self, i):
            return self.indptr[i + 1] - self.indptr[i]

        def neighbors(self, i):
            return self.values[self.indptr[i] : self.indptr[i + 1]]

    return CSR


CSR = make_csr()  # (int64 offsets, uint32 values): every structure we construct

_GRAPH_CSR: dict = {}  # graph indptr numpy dtype -> its CSR jitclass (uint32|uint64)


def graph_csr(indptr: np.ndarray, indices: np.ndarray):
    """Wrap a graph's native-width CSR arrays (zero-copy). ``indptr`` keeps its
    on-disk width (uint32|uint64) so the mmap stays zero-copy; the matching
    jitclass variant is built once per width and cached."""
    cls = _GRAPH_CSR.get(indptr.dtype)
    if cls is None:
        cls = make_csr(indptr.dtype, NODE_DTYPE)
        _GRAPH_CSR[indptr.dtype] = cls
    return cls(indptr, indices)


def group_csr(key: np.ndarray, num: int, drop: int | None = None):
    """Group indices ``0..len(key)-1`` into ``num`` buckets by ``key[i]``, as a
    :data:`CSR` (int64 offsets, uint32 values) via a stable counting sort.
    ``drop`` excludes that single index (used to drop the tree root's
    self-parent when turning parent-pointers into child lists)."""
    key = np.asarray(key)
    if drop is None:
        idx = np.arange(len(key), dtype=NODE_DTYPE)
        bucket = key
    else:
        idx = np.delete(np.arange(len(key), dtype=NODE_DTYPE), drop)
        bucket = np.delete(key, drop)
    values = idx[np.argsort(bucket, kind="stable")]
    indptr = np.zeros(num + 1, dtype=OFFSET_DTYPE)
    indptr[1:] = np.cumsum(np.bincount(bucket, minlength=num))
    return CSR(indptr, values)

import numba as nb
import numpy as np
from numba import njit
from numba.core import types
from numba.core.extending import overload_method
from numba.experimental import structref

from ..graph import CORE_DTYPE, NODE_DTYPE, OFFSET_DTYPE

# Semantic numba scalar aliases (defined once, imported by the kernels).
NB_NODE = nb.from_dtype(NODE_DTYPE)  # vertex & tree-node ids (uint32)
NB_CORE = nb.from_dtype(CORE_DTYPE)  # coreness / k (uint32)
NB_OFFSET = nb.from_dtype(OFFSET_DTYPE)  # CSR offsets (int64)


@structref.register
class CSRType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


@njit(cache=True)
def _num_rows(self):
    return len(self.indptr) - 1


@njit(cache=True)
def _degree(self, i):
    return self.indptr[i + 1] - self.indptr[i]


@njit(cache=True)
def _neighbors(self, i):
    return self.values[self.indptr[i] : self.indptr[i + 1]]


@njit(cache=True)
def _indptr(self):
    return self.indptr


@njit(cache=True)
def _values(self):
    return self.values


class CSR(structref.StructRefProxy):
    """A ``(indptr, values)`` CSR — the in-kernel adjacency abstraction
    (parent->children, node->vertices, graph adjacency). A numba ``structref``:
    usable from Python and object-style inside ``@njit`` kernels
    (``csr.neighbors(i)``). Unlike a jitclass, its numba type is keyed by its
    field dtypes alone (no per-process identity), so kernels taking a ``CSR`` can
    reuse numba's on-disk cache across processes.

    Construction **wraps** the given arrays (no copy); ``neighbors(i)`` and the
    ``.indptr`` / ``.values`` accessors return **views** into the backing arrays —
    callers must not free/mutate them while the ``CSR`` (or a returned slice) is
    alive. numba specializes+caches one concrete type per ``(offset, value)``
    dtype pair automatically, so any offset width (uint32|uint64|int64) works
    zero-copy.
    """

    @property
    def indptr(self):
        return _indptr(self)

    @property
    def values(self):
        return _values(self)

    def num_rows(self):
        return _num_rows(self)

    def degree(self, i):
        return _degree(self, i)

    def neighbors(self, i):
        return _neighbors(self, i)


structref.define_proxy(CSR, CSRType, ["indptr", "values"])


@overload_method(CSRType, "num_rows")
def _ov_num_rows(self):
    def impl(self):
        return len(self.indptr) - 1

    return impl


@overload_method(CSRType, "degree")
def _ov_degree(self, i):
    def impl(self, i):
        return self.indptr[i + 1] - self.indptr[i]

    return impl


@overload_method(CSRType, "neighbors")
def _ov_neighbors(self, i):
    def impl(self, i):
        return self.values[self.indptr[i] : self.indptr[i + 1]]

    return impl


def graph_csr(indptr: np.ndarray, indices: np.ndarray) -> CSR:
    """Wrap a graph's native-width CSR arrays (zero-copy). ``indptr`` keeps its
    on-disk width (uint32|uint64) so the mmap stays zero-copy; numba specializes
    the ``CSR`` type per width on demand."""
    return CSR(indptr, indices)


def group_csr(key: np.ndarray, num: int, drop: int | None = None) -> CSR:
    """Group indices ``0..len(key)-1`` into ``num`` buckets by ``key[i]``, as a
    :class:`CSR` (int64 offsets, uint32 values) via a stable counting sort.
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

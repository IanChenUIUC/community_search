from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NODE_DTYPE = np.dtype(np.uint32)  # vertex & tree-node ids (max ~4B nodes)
CORE_DTYPE = np.dtype(np.uint32)  # coreness / k values
OFFSET_DTYPE = np.dtype(np.int64)  # CSR offsets (signed -> no unsigned underflow in numba)


@dataclass(frozen=True)
class Graph:
    """Undirected graph in CSR form, vertices ``0..n-1``.

    Stored symmetric: every undirected edge ``{u, v}`` appears as both ``(u, v)``
    and ``(v, u)`` in ``indices``. Fields are plain numpy arrays so they can be
    passed to / closed over by numba ``njit`` code. Node ids (``indices``) are
    ``NODE_DTYPE`` (uint32); when loaded from feather the arrays are **zero-copy
    views over the memory-mapped file**, so loading adds ~0 anonymous memory.
    """

    indptr: np.ndarray  # uint32 or uint64, len n+1 (offsets)
    indices: np.ndarray  # NODE_DTYPE, neighbor ids (symmetric)
    # (base, fmt) set by load(). Excluded from equality.
    source: tuple[str, str] | None = field(default=None, compare=False)

    @property
    def num_nodes(self) -> int:
        return len(self.indptr) - 1

    @property
    def num_edges(self) -> int:
        return len(self.indices) // 2  # undirected

    def neighbors(self, u: int) -> np.ndarray:
        return self.indices[self.indptr[u] : self.indptr[u + 1]]

    @property
    def csr(self):
        """A ``CSR`` jitclass wrapping this graph's adjacency (zero-copy), for
        passing into ``@njit`` kernels. The offset width (uint32|uint64) is kept
        native, so the wrap is a view over the memory-mapped buffers."""
        from .structures.csr import graph_csr  # deferred: breaks graph<->csr cycle

        return graph_csr(self.indptr, self.indices)

    @classmethod
    def from_csr(cls, indptr: np.ndarray, indices: np.ndarray) -> Self:
        """Build a Graph from CSR arrays. Node ids are enforced to ``NODE_DTYPE``
        (uint32); offsets are kept as uint32 (when they fit) or uint64. When the
        inputs already have those dtypes (the ``load`` path) this is a no-op, so
        zero-copy is preserved; otherwise the ids are cast (a copy)."""
        indices = np.ascontiguousarray(indices, dtype=NODE_DTYPE)
        indptr = np.asarray(indptr)
        if indptr.dtype not in (np.dtype(np.uint32), np.dtype(np.uint64)):
            fits = len(indices) < 2**32
            indptr = indptr.astype(np.uint32 if fits else np.uint64)
        indptr = np.ascontiguousarray(indptr)
        assert indptr.ndim == 1 and indices.ndim == 1
        assert len(indptr) >= 1 and int(indptr[0]) == 0
        assert int(indptr[-1]) == len(indices)
        return cls(indptr, indices)

    @classmethod
    def load(cls, base: str | Path, fmt: str = "feather") -> Self:
        """Load CSR from the sibling files ``<base>.indptr.<fmt>`` and
        ``<base>.indices.<fmt>`` (each a single-column Arrow table).

        For ``feather`` the arrays are zero-copy views over the mmap'd file.
        Enforces the dtype policy: ``indices`` must be ``NODE_DTYPE`` (uint32),
        ``indptr`` must be ``uint32`` or ``uint64`` (offsets) — the file's dtype
        is taken as-is (no cast) so the read stays zero-copy.
        """
        base = Path(base)
        indptr = _read_column(base.parent / f"{base.name}.indptr.{fmt}", "indptr")
        indices = _read_column(base.parent / f"{base.name}.indices.{fmt}", "indices")
        if indices.dtype != NODE_DTYPE:
            raise ValueError(f"indices must be {NODE_DTYPE}, got {indices.dtype}")
        if indptr.dtype not in (np.dtype(np.uint32), np.dtype(np.uint64)):
            raise ValueError(f"indptr must be uint32 or uint64, got {indptr.dtype}")
        graph = cls.from_csr(indptr, indices)  # no-op casts -> zero-copy preserved
        object.__setattr__(graph, "source", (str(base), fmt))  # frozen: bypass
        return graph

    def save(self, base: str | Path, fmt: str = "feather") -> None:
        """Write CSR to the sibling files ``<base>.indptr.<fmt>`` /
        ``<base>.indices.<fmt>`` (single-chunk Arrow tables), re-loadable via
        :meth:`load`."""
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        _write_column(
            base.parent / f"{base.name}.indptr.{fmt}", "indptr", self.indptr, fmt
        )
        _write_column(
            base.parent / f"{base.name}.indices.{fmt}", "indices", self.indices, fmt
        )


def _read_column(path: str | Path, column: str) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".feather":  # feather V2 == the Arrow IPC file format
        source = pa.memory_map(str(path), "r")
        table = pa.ipc.open_file(source).read_all()
        col = table.column(column)
        assert col.num_chunks == 1, f"{path} must be stored as a single chunk"
        # zero-copy view over the mmap; the returned array keeps the buffer alive
        return col.chunk(0).to_numpy(zero_copy_only=True)
    elif suffix == ".parquet":  # encoded -> a decoded copy (not zero-copy)
        table = pq.read_table(path)
        return table[column].combine_chunks().to_numpy(zero_copy_only=False)
    raise ValueError(f"unsupported format: {path.suffix} (use feather or parquet)")


def _write_column(path: str | Path, column: str, arr: np.ndarray, fmt: str) -> None:
    table = pa.table({column: np.ascontiguousarray(arr)})
    if fmt == "feather":
        with pa.ipc.new_file(str(path), table.schema) as writer:
            writer.write_table(table)
    elif fmt == "parquet":
        pq.write_table(table, str(path))
    else:
        raise ValueError(f"unsupported format: {fmt} (use feather or parquet)")

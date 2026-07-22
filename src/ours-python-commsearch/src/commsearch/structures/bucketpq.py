import numba as nb
import numpy as np
from numba import boolean, int64
from numba.experimental import jitclass


def make_bucketpq(key_dtype, val_dtype):
    """Template factory: return a ``BucketPQ`` jitclass — an addressable bucket
    priority queue over vertex ids ``0..n-1`` keyed by integer priorities in
    ``[0, n)``. ``insert`` / ``extract_max`` / ``contains`` are O(1); ``peek_key``
    is O(1) amortized over a top-down drain.
    """
    key_nb = nb.from_dtype(np.dtype(key_dtype))
    val_nb = nb.from_dtype(np.dtype(val_dtype))
    SENTINEL = np.iinfo(np.dtype(val_dtype)).max

    spec = [
        ("head", val_nb[:]),
        ("nxt", val_nb[:]),
        ("in_pq", boolean[:]),
        ("cur_max", int64),
        ("count", int64),
    ]

    @jitclass(spec)
    class BucketPQ:
        def __init__(self, n):
            self.head = np.full(n, SENTINEL, dtype=val_dtype)
            self.nxt = np.full(n, SENTINEL, dtype=val_dtype)
            self.in_pq = np.zeros(n, dtype=np.bool_)
            self.cur_max = -1
            self.count = 0

        def __len__(self):
            return self.count

        def contains(self, v):
            return self.in_pq[v]

        def insert(self, key, v):
            if self.in_pq[v]:
                return
            self.nxt[v] = self.head[key]
            self.head[key] = v
            self.in_pq[v] = True
            self.count += 1
            if key > self.cur_max:
                self.cur_max = key

        def peek_key(self):
            """Largest non-empty key (caller must ensure non-empty)."""
            while self.head[self.cur_max] == SENTINEL:
                self.cur_max -= 1
            return key_nb(self.cur_max)

        def extract_max(self):
            """Remove and return the ``(key, vertex)`` with the largest key."""
            k = self.peek_key()
            v = self.head[k]
            self.head[k] = self.nxt[v]
            self.in_pq[v] = False
            self.count -= 1
            return k, v

    return BucketPQ

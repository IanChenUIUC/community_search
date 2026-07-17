import numba as nb
from numba.experimental import jitclass
from numba.typed import List


def make_maxheap(key_ty, val_ty):
    """Template factory: return a ``MaxHeap`` jitclass specialized to ``key_ty``
    (an orderable numba scalar type) and ``val_ty`` (the payload type).

    This is numba's equivalent of a C++ template — call it once per (key, value)
    type pair to stamp out a concrete jitclass. The resulting class can be
    instantiated and used inside ``@njit`` code (keys ordered as a max-heap).
    """
    spec = [("_keys", nb.types.ListType(key_ty)), ("_vals", nb.types.ListType(val_ty))]

    @jitclass(spec)
    class MaxHeap:
        def __init__(self):
            self._keys = List.empty_list(key_ty)
            self._vals = List.empty_list(val_ty)

        def __len__(self):
            return len(self._keys)

        def peek_key(self):
            """Largest key without popping (caller must ensure non-empty)."""
            return self._keys[0]

        def push(self, key, val):
            self._keys.append(key)
            self._vals.append(val)
            i = len(self._keys) - 1
            while i > 0:
                parent = (i - 1) // 2
                if self._keys[i] > self._keys[parent]:
                    self._swap(i, parent)
                    i = parent
                else:
                    break

        def pop(self):
            """Remove and return the ``(key, val)`` with the largest key."""
            last = len(self._keys) - 1
            self._swap(0, last)
            max_key = self._keys.pop()
            max_val = self._vals.pop()
            i, n = 0, len(self._keys)
            while True:
                left, right, largest = 2 * i + 1, 2 * i + 2, i
                if left < n and self._keys[left] > self._keys[largest]:
                    largest = left
                if right < n and self._keys[right] > self._keys[largest]:
                    largest = right
                if largest != i:
                    self._swap(i, largest)
                    i = largest
                else:
                    break
            return max_key, max_val

        def _swap(self, i, j):
            self._keys[i], self._keys[j] = self._keys[j], self._keys[i]
            self._vals[i], self._vals[j] = self._vals[j], self._vals[i]

    return MaxHeap

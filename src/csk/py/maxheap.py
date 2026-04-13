import numba as nb
from numba.experimental import jitclass
from numba.typed import List

value_tuple_type = nb.types.UniTuple(nb.types.int64, 2)

spec = [
    ("_keys", nb.types.ListType(nb.types.int64)),
    ("_vals", nb.types.ListType(value_tuple_type)),
]


@jitclass(spec)
class MaxHeap:
    def __init__(self):
        # Initialize typed lists
        self._keys = List.empty_list(nb.types.int64)
        self._vals = List.empty_list(value_tuple_type)

    def __len__(self):
        return len(self._keys)

    def is_empty(self):
        return len(self._keys) == 0

    def push(self, key, val):
        """Push a new key and value tuple onto the heap."""
        self._keys.append(key)
        self._vals.append(val)
        self._siftup(len(self._keys) - 1)

    def pop(self):
        """Pop and return the (key, value) with the LARGEST key."""
        if len(self._keys) == 0:
            raise IndexError("pop from empty heap")

        last_idx = len(self._keys) - 1
        self._swap(0, last_idx)

        max_key = self._keys.pop()
        max_val = self._vals.pop()

        if len(self._keys) > 0:
            self._siftdown(0)

        return max_key, max_val

    def peek(self):
        """Return the (key, value) with the LARGEST key without popping."""
        if len(self._keys) == 0:
            raise IndexError("peek from empty heap")
        return self._keys[0], self._vals[0]

    def _siftup(self, idx):
        while idx > 0:
            parent_idx = (idx - 1) // 2
            if self._keys[idx] > self._keys[parent_idx]:
                self._swap(idx, parent_idx)
                idx = parent_idx
            else:
                break

    def _siftdown(self, idx):
        n = len(self._keys)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            largest = idx

            if left < n and self._keys[left] > self._keys[largest]:
                largest = left
            if right < n and self._keys[right] > self._keys[largest]:
                largest = right

            if largest != idx:
                self._swap(idx, largest)
                idx = largest
            else:
                break

    def _swap(self, i, j):
        temp_key = self._keys[i]
        self._keys[i] = self._keys[j]
        self._keys[j] = temp_key

        temp_val = self._vals[i]
        self._vals[i] = self._vals[j]
        self._vals[j] = temp_val

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AnchoredUnionFind:
    parents: np.ndarray
    sizes: np.ndarray
    roots: np.ndarray

    def find(self, xs: np.ndarray):
        curr = np.array(xs, copy=True)
        while not np.all(self.parents[curr] == curr):
            self.parents[curr] = self.parents[self.parents[curr]]
            curr = self.parents[curr]
        return curr

    def merge(self, xs: np.ndarray) -> int:
        reprs = np.unique(self.find(xs))
        merge_to = reprs[np.argmax(self.sizes[reprs])]
        self.parents[reprs] = merge_to
        self.sizes[merge_to] = np.sum(self.sizes[reprs])
        self.roots[reprs] = self.roots[merge_to]
        return merge_to

    def get_roots(self, xs: np.ndarray) -> np.ndarray:
        reprs = self.find(xs)
        return self.roots[reprs]

    def reroot(self, xs: np.ndarray, new_root: int):
        repr_id = self.merge(xs)
        if repr_id != -1:
            self.roots[repr_id] = new_root


def init_auf(size: int) -> AnchoredUnionFind:
    return AnchoredUnionFind(
        np.arange(size, dtype=np.int32),
        np.ones(size, dtype=np.int32),
        np.full(size, -1, dtype=np.int32),
    )

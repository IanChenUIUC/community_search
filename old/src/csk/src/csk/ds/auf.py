from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass(slots=True)
class UnionFind:
    parents: np.ndarray
    sizes: np.ndarray

    @classmethod
    def init(cls, size: int) -> Self:
        """Creates union-find with singleton sets for 0, 1, ..., size - 1"""
        return cls(np.arange(size, dtype=np.int32), np.ones(size, dtype=np.int32))

    def find(self, xs: np.ndarray) -> np.ndarray:
        """
        @params
            xs: np.ndarray, the elements to find the roots of (between 0 to n-1)
        @returns the representatives of all elements in xs, in the same order of xs.
        """
        curr = xs.copy()
        mask = self.parents[curr] != curr
        while np.any(mask):
            active = curr[mask]
            self.parents[active] = self.parents[self.parents[active]]
            curr[mask] = self.parents[active]
            mask[mask] = self.parents[curr[mask]] != curr[mask]
        return curr

    def merge(self, xs: np.ndarray) -> np.int32:
        """
        merges all elements in xs into the same set. returns the new representative.
        """
        assert xs.size != 0

        reprs = np.unique(self.find(xs))
        merge_to = reprs[np.argmax(self.sizes[reprs])]
        self.parents[reprs] = merge_to
        self.sizes[merge_to] = np.sum(self.sizes[reprs])
        return merge_to


@dataclass(slots=True)
class AnchoredUnionFind:
    uf: UnionFind
    roots: np.ndarray

    @classmethod
    def init(cls, size: int) -> Self:
        """Creates union-find with singleton sets for 0, 1, ..., size - 1"""
        return cls(UnionFind.init(size), np.full(size, -1, dtype=np.int32))

    def find(self, xs: np.ndarray) -> np.ndarray:
        return self.uf.find(xs)

    def merge(self, xs: np.ndarray) -> np.int32:
        return self.uf.merge(xs)

    def get_roots(self, xs: np.ndarray) -> np.ndarray:
        reprs = self.find(xs)
        return self.roots[reprs]

    def reroot(self, xs: np.ndarray, new_root: int) -> None:
        assert xs.size != 0

        repr_id = self.merge(xs)
        self.roots[repr_id] = new_root

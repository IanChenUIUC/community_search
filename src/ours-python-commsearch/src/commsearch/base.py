import math
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class Community(NamedTuple):
    coreness: int
    vertices: np.ndarray


class SelectiveCommunityDetector(ABC):
    """Interface for k-core community-search methods (cf. networkit ``scd``).

    Subclasses implement :meth:`run`; they inherit :meth:`expand_one_community`,
    :meth:`query_coreness`, :meth:`warmup`, and a default parallel driver
    :meth:`run_parallel`. To get zero-copy sharing across worker processes,
    override :meth:`_shared_handle` / :meth:`_from_shared_handle` (and
    :meth:`_release_handle` for cleanup); the defaults just ship ``self`` by value.
    """

    @abstractmethod
    def run(self, queries) -> list[Community]:
        """Detect one community per query (each query is a set of seed vertices)."""

    def warmup(self) -> None:
        """Force njit compilation (and prime the on-disk cache) before heavy use."""
        self.run([np.array([0], dtype=np.int64)])

    def expand_one_community(self, query) -> Community:
        """Community for a single query (an array of one or more seed vertices)."""
        return self.run([query])[0]

    def query_coreness(self, query) -> int:
        return self.expand_one_community(query).coreness

    # --- zero-copy sharing hooks (default: share self by value / pickle-copy) ---

    def _shared_handle(self):
        return self

    @classmethod
    def _from_shared_handle(cls, handle):
        return handle

    def _release_handle(self, handle) -> None:
        """Clean up anything _shared_handle allocated (e.g. temp files)."""

    def run_parallel(
        self, queries, num_threads: int | None = None, max_batch_size: int = 0
    ) -> list[Community]:
        """Run many queries across processes, one contiguous batch per worker.

        Groups the queries into ``<= num_threads`` batches (``max_batch_size > 0``
        caps a batch, forcing more/smaller ones) and runs each with ``self.run``
        in a worker. Workers reconstruct a zero-copy detector via the sharing
        hooks. Results are returned in input order.

        Workers are **spawned** (not forked), so a program that calls this from a
        script must guard its entry point with ``if __name__ == "__main__":``
        (the standard multiprocessing requirement).
        """
        queries = list(queries)
        n = len(queries)
        if n == 0:
            return []
        if num_threads is None:
            num_threads = _available_cpus()
        num_threads = max(1, min(num_threads, n))

        batch_size = math.ceil(n / num_threads)
        if max_batch_size > 0:
            batch_size = min(batch_size, max_batch_size)
        batches = [queries[i : i + batch_size] for i in range(0, n, batch_size)]

        if num_threads == 1 or len(batches) == 1:
            return self.run(queries)

        # spawn (not fork): avoids fork-in-multithreaded-process deadlocks; workers
        # reopen the graph zero-copy from the shared handle instead of inheriting it.
        handle = self._shared_handle()
        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                num_threads, initializer=_init_worker, initargs=(type(self), handle)
            ) as pool:
                results = pool.map(_run_batch, batches)
        finally:
            self._release_handle(handle)

        out: list[Community] = []
        for res in results:
            out.extend(res)
        return out


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on this platform
        return os.cpu_count() or 1


# --- multiprocessing worker plumbing (module-global detector per worker) ---

_WORKER_DETECTOR: SelectiveCommunityDetector | None = None


def _init_worker(cls, handle) -> None:
    global _WORKER_DETECTOR
    _WORKER_DETECTOR = cls._from_shared_handle(handle)


def _run_batch(batch) -> list[Community]:
    assert _WORKER_DETECTOR is not None
    return _WORKER_DETECTOR.run(batch)

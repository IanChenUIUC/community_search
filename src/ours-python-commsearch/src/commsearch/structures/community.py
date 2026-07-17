import numpy as np
from numba.core import types
from numba.experimental import jitclass
from numba.typed import Dict, List

from .csr import NB_CORE, NB_NODE

_MEMBERS = types.Array(NB_NODE, 1, "C")  # one community's vertex array


def make_community_builder():
    """Return a ``CommunityBuilder`` jitclass: the emit accumulator shared by the
    search kernels. It owns the two arrays that must stay in sync — the
    per-community vertex arrays and their coreness — plus an optional
    ``key -> index`` dedup map, so callers never hand-sync them.

    ``append`` is the primitive (register a new community, no dedup). For
    whole-batch dedup keep vertex materialization lazy::

        found, idx = b.lookup(key)
        if not found:
            idx = b.add(key, coreness, verts)   # compute verts only when new

    Callers needing a *narrower* dedup scope (e.g. Steiner's per-finalize batch)
    keep their own scratch map and call ``append`` directly, leaving ``seen``
    unused. Appended vertex arrays are stored by reference (shared buffers); the
    read-only contract carries through to :class:`Community.vertices`.
    """

    @jitclass(
        [
            ("members", types.ListType(_MEMBERS)),
            ("cor", types.ListType(NB_CORE)),
            ("seen", types.DictType(NB_NODE, NB_NODE)),
        ]
    )
    class CommunityBuilder:
        def __init__(self):
            self.members = List.empty_list(_MEMBERS)
            self.cor = List.empty_list(NB_CORE)
            self.seen = Dict.empty(NB_NODE, NB_NODE)

        def append(self, coreness, verts):
            """Register a new community (no dedup) and return its index."""
            idx = NB_NODE(len(self.members))
            self.members.append(verts)
            self.cor.append(coreness)
            return idx

        def lookup(self, key):
            """``(found, index)`` for ``key`` in the dedup map; ``index`` is
            meaningful only when ``found`` (lets the caller skip building vertices
            on a dedup hit)."""
            if key in self.seen:
                return True, self.seen[key]
            return False, NB_NODE(0)

        def add(self, key, coreness, verts):
            """``append`` + record ``key`` in the dedup map; returns the index."""
            idx = self.append(coreness, verts)
            self.seen[key] = idx
            return idx

        def coreness_array(self):
            out = np.empty(len(self.cor), dtype=np.uint32)
            for i in range(len(self.cor)):
                out[i] = self.cor[i]
            return out

    return CommunityBuilder


CommunityBuilder = make_community_builder()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`commsearch` is a standalone, uv-managed, src-layout Python package for **k-core community search** —
a library users import (no CLI). It's a pure-Python (numpy + numba) port that mirrors the API of the
sibling `../ours-icebug-commsearch` (a C++/Cython NetworKit fork; community search under `nk.scd`).

## Commands
- `uv sync` — install the package + dev deps.
- `uv run pytest -q` — run the full test suite.
- `uv run pytest tests/test_steiner.py -q` — a single test file.
- Datasets: CSR feather at `../../input/*.{indptr,indices}.feather` (uint64, two single-column files);
  smallest is `dnc` (n=906). A uint32 copy of `dnc` is vendored at `tests/data/`.
- Memory/time profiling: `../utilities/mytime.py` (reports `peak_anon_kb`; always pass `-o OUT`).

## Architecture
- `graph.py` — `Graph`: undirected CSR, vertices `0..n-1`. Zero-copy mmap feather `load`/`save`;
  node ids are `NODE_DTYPE` (uint32), `indptr` uint32|uint64 (kept native, zero-copy). `.csr` wraps the
  adjacency in a `CSR` structref for kernels. Dtype homes: `NODE_DTYPE`/`CORE_DTYPE` (uint32) and
  `OFFSET_DTYPE` (int64). A `source` breadcrumb enables zero-copy reopen (e.g. across processes).
- `structures/` — numba data structures for kernels, in **two flavors by boundary role** (see
  Conventions): `CSR` and `LCA` are **structrefs** (passed as `@njit(cache=True)` arguments, so their
  numba type must be stable across processes for the on-disk cache to hit — methods via
  `@overload_method` + Python proxy methods); `UnionFind`/`SubsetUnionFind` (factories mirror
  `make_unionfind`), `CommunityBuilder`, and `make_maxheap` (templated PQ) are **jitclasses** (only ever
  constructed *inside* kernels). `CSR` wraps `(indptr, values)` zero-copy — the in-kernel adjacency for
  graph/tree/node-vertices/queries; `neighbors(i)` is a **view**; `graph_csr` is a thin wrapper and
  `group_csr` a counting-sort → CSR; numba specializes one CSR type per `(offset, value)` dtype pair
  automatically (no per-width factory). `CommunityBuilder` is the emit/dedup accumulator (members/coreness
  lists + dedup map); `LCA` is an Euler-tour + RMQ sparse table (~O(1) `ancestor`), built by `build_lca`.
  Derived offsets are `OFFSET_DTYPE` (int64 — signed, so RMQ/degree subtraction can't underflow in
  numba); the `NB_*` numba aliases live at the top of `structures/csr.py`.
- `base.py` — `SelectiveCommunityDetector` ABC (`run` / `expand_one_community` / `query_coreness` /
  `warmup` / `run_parallel`) + `Community` NamedTuple. Shared `run` plumbing: `_flatten_queries(queries,
  dedup)` → a query `CSR`, `_assemble_communities(comm_id, comm_cor, member_arrays)` (shared read-only
  vertex buffers). `run_parallel` batches queries across **spawn** worker processes and shares the graph
  zero-copy via Arrow re-mmap (`_shared_handle` / `_from_shared_handle`); needs `if __name__=="__main__"`.
- `steiner.py` — `SteinerKCore`: batched multi-set k-core search over an `@njit` kernel (`_steiner`),
  consuming `graph.csr` + a query `CSR` and a `CommunityBuilder` (per-finalize dedup via a local scratch
  map). `coreness` normalized to `CORE_DTYPE` once in `__post_init__`.
- `shellstruct.py` — `ShellStruct`: **indexed** variant. Offline `build(graph, coreness)` peels shells
  (decreasing k) into a component tree via `@njit(cache=True)` `_build_shell` (inner closures per phase;
  plain `UnionFind` as the "which component" oracle + a `node_of[root]->node` side array; no anchored UF).
  Fields are CSRs: `tree` (child-CSR) + `node_vertices`, plus the `LCA`. Queries: `_query_shell` = batched
  `lca.ancestor` + a downward subtree traversal (whole-batch dedup via the builder's `seen`). `save`/`load`
  **format-identical to icebug** (two Arrow-IPC files: components `assignment`; tree `coreness`/`vertices`
  large_list/`csr_indptr`/`csr_indices`, uint64, ZSTD; int64↔uint64 cast at the I/O boundary; LCA rebuilt
  at load, root = coreness-0 node). Semantically equivalent but **not byte-identical**: `group_csr`'s stable
  counting sort canonicalizes each node's children into ascending id order, so `tree.csr_indices` may list
  siblings in a different order than icebug's peeling order — same child *sets*, so LCA/subtree queries are
  unaffected (verified: identical coreness+membership over the dnc component tree).
- Steiner vs ShellStruct are cross-checked to agree on dnc (`tests/test_shellstruct.py`). Old `algs/`
  and `csk/` are reference-only (superseded by `structures/lca.py` + `shellstruct.py`); remove when ready.

## Conventions (important — apply by default, don't wait to be asked)
- **Kernel data structures: structref vs jitclass by boundary role.** A struct **passed as an argument**
  to a `@njit(cache=True)` kernel must be a **`structref`** (`CSR`, `LCA`) — a jitclass's numba type name
  embeds `id()`, which changes each process and defeats numba's on-disk cache, so kernels taking a
  jitclass never hit the cache cross-process; structref type names are structural, so they do. A struct
  only **constructed inside** kernels stays a **jitclass** (`UnionFind`/`SubsetUnionFind`/`MaxHeap`/
  `CommunityBuilder`) — it compiles into the enclosing kernel's cache entry, costing nothing extra, and
  reads more cleanly. Both are usable from Python *and* object-style in kernels (`uf.merge(u, v)`), not
  free functions over raw arrays. (structref methods: `@overload_method` for njit + thin Python proxy
  methods; jitclasses: a `make_*` factory.)
- **Keep njit kernels flat**: extract helper `@njit` functions with early returns; avoid deep nesting.
- **Reuse, don't duplicate** (one mmap reader/writer, etc.).
- **Dtypes: ids `NODE_DTYPE` + coreness `CORE_DTYPE` (uint32); constructed CSR offsets `OFFSET_DTYPE`
  (int64, signed — never uint on the njit side).** The graph `indptr` keeps its native on-disk width
  (uint32|uint64), zero-copy — never cast on the hot path. All graph I/O is zero-copy Arrow feather.
  Coreness is an external input array (never computed here). No scipy.
- **Mirror `../ours-icebug-commsearch` API/naming; flag any deviation and say why.**
- **Comments describe interfaces (docstrings), not behavior.** Don't narrate what code does; the only
  comment exception is behavior affecting the caller (e.g. same-community `Community.vertices` share
  one buffer — callers must not mutate/free it).
- De-risk njit/pyarrow assumptions with a quick experiment before building.

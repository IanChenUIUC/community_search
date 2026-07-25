# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`commsearch-abm272` is a uv-managed **consumer/experiment package**, not a library. It drives the two
k-core community-search engines against the `abm272` dataset (an agent-based-model-generated citation
network). It has no algorithms of its own — those live in the sibling packages it depends on.

## Engines it drives (source of truth for behavior)
- **`commsearch`** — our pure-Python (numpy + numba) engine, a **git dependency** pinned in
  `pyproject.toml` to `src/ours-python-commsearch` of the same GitHub repo. See
  `../ours-python-commsearch/CLAUDE.md` for its architecture (`SteinerKCore`, `ShellStruct`, CSR structrefs).
- **`icebug` (`>=12.9`)** — a C++/Cython NetworKit fork; imports as `networkit as nk`, community search
  under `nk.scd` (e.g. `nk.scd.ShellStruct`). The consumer scripts here (`build_shellstruct.py`, and any
  `query_*` / `core_decomposition` added later) **mirror the drivers in `../ours-icebug-commsearch/`** —
  treat those as the reference and keep signatures/CLI in sync with them.

## Commands
- `uv sync` — install the package + both engines.
- `uv run build_shellstruct.py INDPTR_PATH INDICES_PATH SHELL_BASE_PATH [CORES]` — build an icebug
  ShellStruct index from CSR feather (writes `<base>.components.feather` + `<base>.tree.feather`).
- `uv run <script>.py --help` — every driver is a `click` CLI; check `--help` for the exact argument order.
- Memory/time profiling: `../utilities/mytime.py` (reports `peak_anon_kb`; always pass `-o OUT`).

## Data
- `edgelist.parquet` / `nodelist.csv` live here but their contents are **stored remotely** (not in git) —
  don't assume they're present locally. `nodelist.csv` carries rich ABM node attributes (`type`, `year`,
  `alpha`, `pa_weight`, `fit_weight`, degrees, generator provenance); the canonical edgelist is
  `source,target`, undirected, each edge once.
- **Pipeline shape:** csv edgelist → CSR (two single-column feather/parquet files, `indptr` + `indices`)
  via `../utilities/csv2csr.py`; core decomposition produces the `CORES` scores; ShellStruct / Steiner
  drivers consume CSR (+ scores/queries). CSR reads are zero-copy memory-mapped feather.

## Scope
Only this package, the `../ours-*` engine packages, and `../utilities` are relevant here; the other `src/`
directories (core-decomposition variants, `gullo-shellstruct`, `csk-commsearch`, `gen-query`) belong to
the broader experiment and are out of scope for work in this package.

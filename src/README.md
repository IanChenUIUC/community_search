## Source code for all methods in experiment

Many of these are gitmodules, so they have to be initalized with.

```
git submodule update --init --recursive
```

To build each repository, see the command in `[recipe.build]` of the `slurm/pipeline.toml` orchestrator.

## Core Decomposition

### gbbs-core-decomposition

Ligra's parallel bucketed core decomposition.

### networkit-core-decomposition

Networkit's ParK implementation.

### ucrparlay-core-decomposition

Parallel peeling with vertical granularity control.

### pkc-core-decomposition

PKC algorithm.

### lbug-core-decomposition

Implementation of Ligra's core decomposition, in the LadybugDB graph database.

## Community Search

### gullo-shellstruct

Shellstruct, as well as implementations of local and global search.

### csk-commsearch

Single-vertex k-core community search.

### ours-icebug-commsearch

Our novel methods and implementations.
Implemented in Icebug, a fork of NetworKit.

### ours-python-commsearch

Our novel methods and implementations.
Implemented with JIT compilation.


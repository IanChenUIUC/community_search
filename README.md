# About

Master repository for `Improved Methods for k-Core Community Search`.

# Directory Structure

## Methods

All software is in `src/`.

### New Methods

#### Icebug community search

This contains a parallel implementation of `ShellStruct`, and `SteinerKCore` and `LocalKCore`, exposed in the `nk.scd` namespace.
Not currently merged with the upstream Icebug yet.

#### Python community search

Python implementations of the same algorithms as in Icebug, but using JIT-compilation.

### External Methods

#### GBBS core decomposition

#### UCR core decomposition

#### NetworKit ParK core decomposition

#### PKC core decomposition

#### Ladybug core decomposition

#### ShellStruct community search

## Running Experiments

`input/` and `output/` directories.

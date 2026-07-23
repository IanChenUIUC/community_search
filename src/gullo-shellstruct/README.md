# Source Code for Efficient and Effective Community Search

## Building and running

```
mvn package
chmod +x run.sh
./run.sh <class> <arguments>
```

`mvn package` downloads Kryo, compiles everything into `target/classes`, and copies Kryo's jar into `target/dependency`.
`run.sh` invokes java on the above format.

## Directory structure

```
.
├── README.md
├── input
│   ├── Datasets
│   │   └── dnc.csv
│   └── Queries
│       ├── dnc-set.csv
│       └── dnc-singleton.csv
├── pom.xml
├── run.sh
└── src
    └── main
        └── java
            ├── cocktailParty
            │   ├── CocktailParty.java
            │   └── ...
            ├── csm
            │   ├── CSM.java
            │   └── ...
            ├── index
            │   ├── TreeIndex.java
            │   └── ...
            ├── main
            │   ├── BuildIndex_PKDDJ.java
            │   ├── MyMain.java
            │   └── ...
            └── ...
                └── ...
```

In which the `src` directory contains the implementations for global search, local search, shellstruct respectively, as well as the driver code.

## Changes to Code

1. Added `pom.xml` Maven configuration file, and `run.sh`.
2. Changed `src/main/java/main/Graph.java:readFromFile` to read edgelists by csv with a header, according to [data-specification](https://github.com/illinois-or-research-analytics/data-specification).
    In both cases, it adds both edge directions.
3. Changed `src/main/java/fromResults/ResultsReader.java:queriesReader` to read queries as a csv.
4. Added `src/main/java/main/MyMain.java` to stop using hard-coded paths, and persist the communities.
5. Added example network `dnc` and queries.

## Example pipeline (dnc)

### Inputs

The input network is a CSV edgelist, e.g. `input/Datasets/dnc.csv`.
This should be a simple, unweighted, undirected graph, with a header.

For each method, it requires a list of queries (each query is run independently), e.g. `input/Queries/dnc-*.csv`.
This should have a single line for each query, with each node in that query comma-separated.

### Outputs

```
# build ShellStruct
mkdir -p output/dnc
./run.sh main.BuildIndex_PKDDJ input/Datasets/dnc.csv output/dnc/ShellStruct.bin

# run Algorithms
mkdir -p output/dnc/query-singleton
./run.sh main.MyMain \
  -n input/Datasets/dnc.csv \
  -s output/dnc/ShellStruct \
  -sk ss \
  -q input/Queries/dnc-singleton.csv \
  -a grcon-gs-ls-core \
  -o output/dnc/query-singleton

mkdir -p output/dnc/query-set
./run.sh main.MyMain \
  -n input/Datasets/dnc.csv \
  -s output/dnc/ShellStruct \
  -sk ss \
  -q input/Queries/dnc-set.csv \
  -a grcon-gs-core \
  -o output/dnc/query-set
```

## Reference

The original code is uploaded to [dropbox](http://bit.ly/1b6WbSQ).

```bibtex
@article{barbieri2015efficient,
  title={Efficient and effective community search},
  author={Barbieri, Nicola and Bonchi, Francesco and Galimberti, Edoardo and Gullo, Francesco},
  journal={Data mining and knowledge discovery},
  volume={29},
  number={5},
  pages={1406--1433},
  year={2015},
  publisher={Springer}
}
```

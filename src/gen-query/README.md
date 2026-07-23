# About

This is the code for generating queries for k-core community search.
Looking at `src/networkit-core-decomposition/centrality.py`, we found that sampling nodes by degree gets the most cohesive (by coreness) communities.
Thus, this code samples accordingly.

The `py_v_ib.py` script was for the training experiment comparing icebug and python implementations.
`gen_query.py` is the more general script,

```
> uv run gen_query.py --help
Usage: gen_query.py [OPTIONS] INDPTR_PATH INDICES_PATH OUT_DIRECTORY

Options:
  -n, --query_size INTEGER
  -b, --batch_size INTEGER
  -r, --reps INTEGER
  --help                    Show this message and exit.
```

where it uses the Icebug code to generate a query by sampling nodes (without replacement) in the top 1% of degree centrality.
Each file in the output will be a single batch of queries of a fixed size, with each query sampled independently, of filename `query{rep}.csv`.

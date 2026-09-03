# abm272-scaling

The size-scaling experiment for k-core community search, run on CANTATAS (descibed at [zenodo](https://zenodo.org/records/21513973)).

## `prep_year.py`

```
> uv run prep_year.py --help
Usage: prep_year.py [OPTIONS] EDGELIST_PATH NODELIST_PATH YEAR OUTPUT_CSR_BASE

Options:
  --threads INTEGER
  --help             Show this message and exit.
```

Writes `<base>.indptr.parquet` and `<base>.indices.parquet` for the subgraph induced by the nodes
whose `year` is at most `YEAR`, and prints `nodes=<n>`.

mkdir -p tmp \
  && uv run csk compact \
    --edgelist ../data/dnc.tsv \
    --out_edges tmp/dnc_compact.csv \
    --out_mapping tmp/dnc_map.csv \
    --sep $'\t' --head False \
  && uv run csk coreness \
    --edgelist tmp/dnc_compact.csv --output tmp/cores.csv \
  && uv run csk steiner-search \
    --edges tmp/dnc_compact.csv \
    --nodemap tmp/dnc_map.csv \
    --coreslist tmp/cores.csv \
    --nodelist query.txt \
    --outputdir tmp/

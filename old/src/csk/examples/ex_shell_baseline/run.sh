mkdir -p tmp \
  && uv run csk compact \
    --edgelist ../data/dnc.tsv \
    --out_edges tmp/dnc_compact.csv \
    --out_mapping tmp/dnc_map.csv \
    --sep $'\t' --head False \
  && uv run csk shell-baseline-index \
    --edgelist tmp/dnc_compact.csv \
    --output tmp/shell.bin \
  && uv run csk shell-baseline-search \
    --index tmp/shell.bin \
    --nodelist query.txt \
    --nodemap tmp/dnc_map.csv \
    --outputdir tmp

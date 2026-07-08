mkdir -p tmp \
  && uv run csk compact \
    --edgelist ../data/dnc.tsv \
    --out_edges tmp/dnc_compact.csv \
    --out_mapping tmp/dnc_map.csv \
    --sep $'\t' --head False \
  && uv run csk coreness \
    --edgelist tmp/dnc_compact.csv --output tmp/cores.csv \
  && uv run csk shell-compressed-index \
    --coreslist tmp/cores.csv \
    --edgelist tmp/dnc_compact.csv \
    --output tmp/shell.bin.npz \
  && uv run csk shell-compressed-search \
    --shell_file tmp/shell.bin.npz \
    --nodemap tmp/dnc_map.csv \
    --queries query.txt \
    --outputdir tmp

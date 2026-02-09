# generate index
/usr/bin/time -v ~/ianchen3/csearch/scripts/gen_kcore_idx.sh \
  output/kcore/$dataset/ \
  data/processed/$dataset/edges.tsv 

# get query nodes
~/ianchen3/csearch/core_decomp/index.py \
  index \
  --edgelist data/processed/$dataset/edges.tsv \
  --output tmp
~/ianchen3/csearch/core_decomp/index.py \
  query \
  --index tmp \
  --output tmp
nodes=$(cat tmp)
rm tmp

# run csk on each of the thingies
/usr/bin/time -v ~/ianchen3/csearch/scripts/run_max_kcore.sh \
  output/kcore/11_24/$dataset/ \
  data/processed/$dataset/edges.tsv \
  output/kcore/$dataset/kcore_index.txt \
  $nodes

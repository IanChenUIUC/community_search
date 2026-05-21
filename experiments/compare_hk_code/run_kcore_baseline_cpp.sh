#!/usr/bin/bash

export CONTAINER=/u/ianchen3/venv/python_bootstrap.sif
export PYTHON=/u/ianchen3/venv/myenv/bin/python
export SOURCE=/u/ianchen3/community_search/src/csk/cpp
export INPUT=/u/ianchen3/scratch/csearch-comparison-baseline/input
export OUTPUT=/u/ianchen3/scratch/csearch-comparison-baseline/output

mkdir -p $OUTPUT
cd /scratch/ianchen3/csearch-comparison-baseline

dataset=$1 # e.g. cit_hepph, orkut, cen

/usr/bin/time -v 2> >(tee $OUTPUT/${dataset}_csk_index.err) \
  apptainer exec $CONTAINER \
  $SOURCE/main -i \
  $OUTPUT/${dataset}_csk/ \
  $INPUT/${dataset}.tsv

queries=('high_1' 'high_10' 'high_100' 'rand_1' 'rand_10' 'rand_100')
for query in ${queries[@]}; do
  mkdir -p $OUTPUT/${dataset}_csk
  /usr/bin/time -v 2> >(tee $OUTPUT/${dataset}_csk_${query}.err) \
    apptainer exec $CONTAINER \
    $SOURCE/main -s \
    $OUTPUT/${dataset}_csk/ \
    $INPUT/${dataset}.tsv \
    $OUTPUT/${dataset}_index.tsv \
    $INPUT/${dataset}_queries/${query}.txt    
done

echo done


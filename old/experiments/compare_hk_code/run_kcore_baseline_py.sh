#!/usr/bin/bash

export CONTAINER=/u/ianchen3/venv/python_bootstrap.sif
export PYTHON=/u/ianchen3/venv/myenv/bin/python
export SOURCE=/u/ianchen3/community_search/src/csk/py
export INPUT=/u/ianchen3/scratch/csearch-comparison-baseline/input
export OUTPUT=/u/ianchen3/scratch/csearch-comparison-baseline/output

mkdir -p $OUTPUT
cd /scratch/ianchen3/csearch-comparison-baseline

dataset=$1 # e.g. dnc, cit_hepph, orkut, cen

/usr/bin/time -v 2> >(tee $OUTPUT/${dataset}_compact.err) \
  apptainer exec $CONTAINER \
  $PYTHON $SOURCE/compact.py \
  --edgelist $INPUT/${dataset}_cleaned.tsv \
  --output $INPUT/${dataset}.tsv
 
/usr/bin/time -v 2> >(tee $OUTPUT/${dataset}_genquery.err) \
  apptainer exec $CONTAINER \
  $PYTHON $SOURCE/genquery_singleton.py \
  --edgelist $INPUT/${dataset}.tsv \
  --outdir $INPUT/${dataset}_queries \
  --header False --sep '	'

/usr/bin/time -v 2> >(tee $OUTPUT/${dataset}_index.err) \
  apptainer exec $CONTAINER \
  $PYTHON $SOURCE/kcore.py index \
  --edgelist $INPUT/${dataset}.tsv \
  --output $OUTPUT/${dataset}_index.tsv

queries=('high_1' 'high_10' 'high_100' 'rand_1' 'rand_10' 'rand_100')
for query in ${queries[@]}; do
  /usr/bin/time -v 2> >(tee $OUTPUT/${dataset}_search_${query}.err) \
    apptainer exec $CONTAINER \
    $PYTHON $SOURCE/kcore.py search \
    --edgelist $INPUT/${dataset}.tsv \
    --nodelist $INPUT/${dataset}_queries/${query}.txt \
    --index $OUTPUT/${dataset}_index.tsv \
    --outputdir $OUTPUT/${dataset}_search/$query
done

echo done


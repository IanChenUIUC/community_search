#!/bin/bash

source $MYENV/bin/activate
# == ARGUMENTS ==
# $1 - the percentile to run it on

# == PIPELINE TO RUN THE CSK vs IKC experiment on CEN ==
mkdir -p input/cen_percentiles
mkdir -p output/kcore/cen/$1
mkdir -p output/stats/
mkdir -p output/mem
mkdir -p output/ikc

# python analysis/summary.py \
#    --csk_dir output/kcore/cit-hepph/run2 \
#    --ikc_com output/ikc/cit-hepph/com.tsv \
#    --csk_stat output/stats/11_11_2025/cit-hepph/run2/stats.csv \
#    --ikc_stat output/stats/11_11_2025/cit-hepph/ikc/stats.csv \
#    --output output/stats/11_11_2025/cit-hepph/combined.csv

# rm -r output/kcore/cit-hepph/run1
# mkdir -p  output/kcore/cit-hepph/run1
# /usr/bin/time -v \
#    python ./repos/kcsearch/kcore.py \
#    --edgelist data/processed/cit-hepph/edges.tsv \
#    --outdir output/kcore/cit-hepph/run1 \
#    --min-k 20 \
#    2> output/kcore/cit-hepph/run1/stderr.txt
 
# rm -r output/kcore/cit-hepph/run2
# mkdir -p  output/kcore/cit-hepph/run2
# /usr/bin/time -v \
#    python ./repos/kcsearch/kcore.py \
#    --edgelist data/processed/cit-hepph/edges.tsv \
#    --outdir output/kcore/cit-hepph/run2 \
#    --min-k 20 \
#    2> output/kcore/cit-hepph/run2/stderr.txt

# rm -rf output/stats/11_11_2025/cit-hepph/run2
# mkdir -p output/stats/11_11_2025/cit-hepph/run2
# /usr/bin/time -v \
#    python ./repos/kcsearch/stats.py \
#    --edgelist data/processed/cit-hepph/edges.tsv \
#    --clustdir output/kcore/cit-hepph/run2 \
#    --outfile output/stats/11_11_2025/cit-hepph/run2/stats.csv \
#    2> output/stats/11_11_2025/cit-hepph/run2/stderr.txt

# generate the index file
# /usr/bin/time -v ~/ianchen3/csearch/scripts/gen_kcore_idx.sh \
#   output/kcore/cen/ \
#   data/processed/cen/edges.tsv 

# sample some query nodes from percentile of degree
# apptainer exec $MYCONTAINER \
#   ~/ianchen3/csearch/scripts/gen_percentiles.py \
#   --edgelist data/processed/cen/edges.tsv \
#   --output input/cen_percentiles/$1.tsv \
#   --percentile $1 \
#   --number 100

# run community search
# /usr/bin/time -v 2> "output/mem/csk$1.txt" \
#    apptainer exec $MYCONTAINER \
#    ~/ianchen3/csearch/scripts/run_max_kcore.sh \
#    output/kcore/cen/$1/ \
#    data/processed/cen/edges.tsv \
#    output/kcore/cen/kcore_index.txt \
#    input/cen_percentiles/$1.tsv \
 
# ~/ianchen3/csearch/scripts/run_max_kcore.sh \
#    output/kcore/cen/test/ \
#    data/processed/cen/edges.tsv \
#    output/kcore/cen/kcore_index.txt \
#    input/rep_cores/in.tsv
   
# compute cluster stats
# python ~/ianchen3/csearch/scripts/ikc_csk_stats.py \
#    --edgelist data/processed/cen/edges.tsv \
#    --csk_com output/kcore/cen/$1/max_kcore \
#    --ikc_com output/ikc/cen_com.tsv \
#    --output tmp.csv \
#    --csk_id $2
   # --output output/stats/ikc_csk${1}_${2}.csv \

# == SCRIPT TO RUN IKC
# apptainer exec $MYCONTAINER \
# rm -rf ouput/ikc/cit-hepph
# mkdir -p output/ikc/cit-hepph
# /usr/bin/time -v \
#    ~/ianchen3/csearch/binaries/ikc \
#   -e data/processed/cit-hepph/edges.tsv \
#   -o output/ikc/cit-hepph/com.tsv \
#   -k 20 \
#   2> output/ikc/cit-hepph/stderr.txt

# rm -rf output/stats/11_10_2025/cit-hepph/ikc
# mkdir -p output/stats/11_10_2025/cit-hepph/ikc
# /usr/bin/time -v \
#    python analysis/stats.py \
#    --edgelist data/processed/cit-hepph/edges.tsv \
#    --comlist output/ikc/cit-hepph/com.tsv \
#    --outfile output/stats/11_10_2025/cit-hepph/ikc/stats.csv \
#    2> output/stats/11_10_2025/cit-hepph/ikc/stderr.txt

# == SCRIPT TO COMPACTIFY NODE-IDS ==
# mkdir -p data/processed/cit-patents
# python scripts/preprocess_edgelist.py \
#    --input-edgelist data/raw/cit-patents.tsv \
#    --output-edgelist data/processed/cit-patents/edges.tsv \
#    --output-mapping data/processed/cit-patents/map.tsv

# == SCRIPT TO RUN CSK ==
# input_dirs=input/academia_edu_percentiles/
# output_dir=output/kcore/95
# edges_file=data/processed/academia_edu/edges.tsv
# index_file=output/kcore/kcore_index.txt
# query_file=${input_dirs}95.tsv

# ./scripts/gen_kcore_idx.sh $output_dir $edges_file
# ./scripts/run_max_kcore.sh $output_dir $edges_file $index_file $query_file

# == SCRIPT TO RUN IKC ==
# apptainer exec $MYCONTAINER ~/ianchen3/csearch/binaries/ikc \
#   -e data/processed/academia_edu/edges.tsv \
#   -o output/ikc/academia_edu_com.tsv \
#   -k 2

# == SCRIPT TO GEN PERCENTILES ==
# apptainer exec $MYCONTAINER ~/ianchen3/csearch/scripts/gen_percentiles.py \
#   --edgelist data/processed/academia_edu/edges.tsv \
#   --output input/academia_edu_percentiles/$1.tsv \
#   --percentile $1 \
#   --number 250
 
# == SCRIPT TO RUN GLOBAL ==
# # TODO: make the remapping so that nodes are properly indexed
# ./scripts/run_query.sh $output_dir $edges_file $query_file

deactivate

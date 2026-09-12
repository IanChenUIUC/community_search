#!/bin/bash
#SBATCH --job-name=make-input-abm272mf
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=secondary
#SBATCH --mem=256GB
#SBATCH --output=/u/ianchen3/scratch/slurm/slurm-%A.out

set -euo pipefail
CONTAINER=/u/ianchen3/venv/python_bootstrap-sandbox
UTIL=/u/ianchen3/community_search/src/utilities
INPUT=/u/ianchen3/community_search/input

exec apptainer exec -B /scratch:/scratch -B /projects:/projects "$CONTAINER" bash -c "
  set -euo pipefail
  uv run --project $UTIL $UTIL/pq2csv.py \
    $INPUT/abm272.edgelist.parquet \
    $INPUT/abm272.raw \
    --threads 4
  uniq $INPUT/abm272.raw.csv > $INPUT/abm272.tmp.csv
  mv $INPUT/abm272.tmp.csv $INPUT/abm272.csv
  rm $INPUT/abm272.raw.csv
"

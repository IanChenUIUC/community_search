#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=secondary
#SBATCH --mem=256GB
#SBATCH --output=/u/ianchen3/scratch/slurm/slurm-%A.out

set -euo pipefail
CONTAINER=/u/ianchen3/venv/python_bootstrap-sandbox

exec apptainer exec "$CONTAINER" bash -c "
  cd /u/ianchen3/community_search/make-input
  source .venv/bin/activate
  python netzschleuder.py
  uv run --project /u/ianchen3/community_search/src/utilities databank.py cen abm14
"

echo "done"

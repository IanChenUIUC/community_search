#!/usr/bin/sh

# sbatch ./scripts/run.sbatch.sh

sbatch ./scripts/run.sbatch.sh 79 0

# for j in $(seq 0 10);
# do
#   for i in $(seq 3 4 99);
#   do
#     sbatch ./scripts/run.sbatch.sh $i $j
#   done
# done

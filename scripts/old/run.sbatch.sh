#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=MissionControl
#SBATCH --output=/u/ianchen3/slurm/slurm-%j.out

export MYCONTAINER=$HOME/venv/python_bootstrap.sif

# export MYENV="$HOME/ianchen3/csearch/env/${SLURM_JOB_ID}"
# apptainer exec -B /projects:/projects $MYCONTAINER $HOME/scripts/create_venv $MYENV
# export MYENV="$HOME/venv/myenv"
# apptainer exec -B /projects:/projects $MYCONTAINER ./scripts/run.sh
# rm -rf $MYENV
 
export MYENV="$HOME/venv/myenv"
apptainer exec -B /projects:/projects $MYCONTAINER ./scripts/run.sh

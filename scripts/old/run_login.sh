export MYENV="$HOME/venv/myenv"

# apptainer exec /u/ianchen3/venv/python_bootstrap.sif \
# 	./scripts/run.sh
 
export dataset=$1
apptainer exec /u/ianchen3/venv/python_bootstrap.sif \
	./scripts/11-24.sh $1

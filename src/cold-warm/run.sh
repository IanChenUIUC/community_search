mkdir -p ${output}/${dataset}/cold-warm

PY_UTIL=$(uv run --project ${root}/src/utilities  python -c 'import sys; print(sys.executable)')
PY_CW=$(  uv run --project ${root}/src/cold-warm  python -c 'import sys; print(sys.executable)')

export LD_LIBRARY_PATH=$(${PY_CW} -c "import pyarrow; print(pyarrow.get_library_dirs()[0])"):$LD_LIBRARY_PATH

MYTIME="${PY_UTIL} ${root}/src/utilities/mytime.py"
INDPTR=${input}/${dataset}.indptr.feather
INDICES=${input}/${dataset}.indices.feather
CORES=${output}/${dataset}/cold-warm/cores.npy

evict_inputs() { vmtouch -e ${INDPTR} ${INDICES}; }
evict_cores()  { sync; [ -f ${CORES} ] && vmtouch -e ${CORES}; }

## cold coreness
evict_inputs
${MYTIME} -o ${timing}-coldcores.txt -- \
  ${PY_CW} ${root}/src/cold-warm/coreness.py ${INDPTR} ${INDICES} ${CORES} \
  1> >(tee ${stdout}) 2> >(tee ${stderr})

## cold query
evict_inputs
evict_cores
${MYTIME} -o ${timing}-coldquery.txt -- \
  ${PY_CW} ${root}/src/cold-warm/query.py ${INDPTR} ${INDICES} ${CORES} \
  1> >(tee -a ${stdout}) 2> >(tee -a ${stderr})

## simult coreness
evict_inputs
${MYTIME} -o ${timing}-simult1coreness.txt -- \
  ${PY_CW} ${root}/src/cold-warm/coreness.py ${INDPTR} ${INDICES} \
    ${output}/${dataset}/cold-warm/simult1-cores.npy \
  1> >(tee -a ${stdout}) 2> >(tee -a ${stderr}) &
${MYTIME} -o ${timing}-simult2coreness.txt -- \
  ${PY_CW} ${root}/src/cold-warm/coreness.py ${INDPTR} ${INDICES} \
    ${output}/${dataset}/cold-warm/simult2-cores.npy \
  1> >(tee -a ${stdout}) 2> >(tee -a ${stderr}) &
wait

## simult query
evict_inputs
evict_cores
${MYTIME} -o ${timing}-simult1query.txt -- \
  ${PY_CW} ${root}/src/cold-warm/query.py ${INDPTR} ${INDICES} ${CORES} \
  1> >(tee -a ${stdout}) 2> >(tee -a ${stderr}) &
${MYTIME} -o ${timing}-simult2query.txt -- \
  ${PY_CW} ${root}/src/cold-warm/query.py ${INDPTR} ${INDICES} ${CORES} \
  1> >(tee -a ${stdout}) 2> >(tee -a ${stderr}) &
wait

## warm coreness
vmtouch -t ${INDPTR} ${INDICES}
${MYTIME} -o ${timing}-warmcoreness.txt -- \
  ${PY_CW} ${root}/src/cold-warm/coreness.py ${INDPTR} ${INDICES} ${CORES} \
  1> >(tee -a ${stdout}) 2> >(tee -a ${stderr})

## warm query
vmtouch -t ${INDPTR} ${INDICES} ${CORES}
${MYTIME} -o ${timing}-warmquery.txt -- \
  ${PY_CW} ${root}/src/cold-warm/query.py ${INDPTR} ${INDICES} ${CORES} \
  1> >(tee -a ${stdout}) 2> >(tee -a ${stderr})

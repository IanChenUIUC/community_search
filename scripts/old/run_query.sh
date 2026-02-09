#!/bin/bash

mkdir -p $1
python3 ./binaries/global --edgelist=$2 --querylist=$3 --weighted=False

#!/bin/bash

PROJECT_DIR="set/path/"

seed=$1
ssr=$2

source # set your venv

python3 ${PROJECT_DIR}/tedsim_fit.py --seed ${seed} --ssr ${ssr}
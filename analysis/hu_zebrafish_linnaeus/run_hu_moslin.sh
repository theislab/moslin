#!/bin/bash

PROJECT_DIR="set/dir"

alpha=$1
epsilon=$2
beta=$3
tau_a=$4
tau_b=$5
save=$6


source #activate venv

python3 ${PROJECT_DIR}/hu_moslin_couplings_transient_fibro_prob.py --alpha ${alpha} --epsilon ${epsilon} --beta ${beta} --tau_a ${tau_a} --tau_b ${tau_b} --save ${save}

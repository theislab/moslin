#!/bin/bash

PROJECT_DIR=""

alpha=$1
epsilon=$2
beta=$3
tau_a=$4

# activte env

python3 ${PROJECT_DIR}/hu_moslin_fit_couplings.py --alpha ${alpha} --epsilon ${epsilon} --beta ${beta} --tau_a ${tau_a}
#!/bin/bash

PROJECT_DIR="set/path/"


epsilon=$1
tau_a=$2
save=$3

source # activate local env

python3 ${PROJECT_DIR}/hu_lot_couplings_transient_fibro_prob.py --epsilon ${epsilon} --tau_a ${tau_a} --save ${save}
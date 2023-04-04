#!/bin/bash

PROJECT_DIR="/cs/labs/mornitzan/zoe.piran/research/projects/moslin/analysis/simulations/forrow_2gene"

flow_type=$1

module load torch/1.11.0-cuda11.3
source /cs/labs/mornitzan/zoe.piran/venvzp/bin/activate

python3 ${PROJECT_DIR}/utils.py --flow_type ${flow_type}
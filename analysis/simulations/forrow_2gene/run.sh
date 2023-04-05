#!/bin/bash

PROJECT_DIR="/moslin/analysis/simulations/forrow_2gene"

flow_type=$1

python3 ${PROJECT_DIR}/utils.py --flow_type ${flow_type}
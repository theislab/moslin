#!/bin/bash

PROJECT_DIR="set/path/"

flow_type=$1

source # set your venv

python3 ${PROJECT_DIR}/utils_ot.py --flow_type ${flow_type}
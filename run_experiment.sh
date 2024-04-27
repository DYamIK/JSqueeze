#!/usr/bin/env bash
DATASET=${1}
SETTING=${2}
NUM_WORKER=20
python run_algorithm.py --name ${SETTING} --input-path ${DATASET} --output-path output/${DATASET}/ --num-workers ${NUM_WORKER}
python run_evaluation.py -i ${DATASET}/${SETTING}/injection_info.csv -p output/${SETTING}.json -c columns.json

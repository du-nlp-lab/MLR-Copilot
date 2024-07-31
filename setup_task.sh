#!/bin/bash
task=$1

cd "/app/benchmarks/$task/scripts"

if [[ -f prepare.py ]]; then
    python3 prepare.py
fi

cd /app
./copy_task.sh $task

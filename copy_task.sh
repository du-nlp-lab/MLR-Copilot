#!/bin/bash
task=$1

if [[ -e "/app/workspaces/$task" ]]; then
    rm -rf "/app/workspaces/$task"
fi

cp -r "/app/benchmarks/$task/env" "/app/workspaces/$task"
cp "/app/benchmarks/$task/scripts/research_problem.txt" "/app/problems/$task"

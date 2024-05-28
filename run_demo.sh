#!/bin/bash

prob=$1
name=$2

ts=$(date +%s)

if [ -d $name/$ts ]; then
    echo "Folder $name/$ts already exists. removing it"
    rm -rf $name/$ts
fi
mkdir -p $name/$ts

python -u -m MLAgentBench.runner --research-problem $prob --log-dir $name/$ts --work-dir workspaces/$name/$ts > $name/$ts/log 2>&1
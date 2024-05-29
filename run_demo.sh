#!/bin/bash

prob=$1
name=$2

ts=$(date -I) 

logdir="logs/$name/$ts"

if [ -d $logdir]; then
    echo "Folder $logdir already exists. removing it"
    rm -rf $logdir
fi
mkdir -p $logdir

python -u -m MLAgentBench.runner --research-problem $prob --log-dir $logdir --work-dir workspaces/$name/$ts > $logdir/log 2>&1

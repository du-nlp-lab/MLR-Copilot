#!/bin/bash

prob=$1
name=$2

logdir="logs/$name"
workdir="workspaces/$name"

if [ -d $logdir ]; then
    echo "Folder $logdir already exists. removing it"
    rm -rf $logdir
fi
mkdir -p $logdir

python -u -m MLAgentBench.runner --research-problem $prob --log-dir $logdir --work-dir $workdir --retrieval

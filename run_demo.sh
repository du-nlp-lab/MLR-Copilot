#!/bin/bash

name=$1

prob="/app/problems/$name"
logdir="/app/logs/$name"
workdir="/app/workspaces/$name"

if [ -d $logdir ]; then
    echo "Folder $logdir already exists. removing it"
    rm -rf $logdir
fi
mkdir -p $logdir

python -u -m reactagent/runner.py --research-problem "$(< $prob)" --log-dir $logdir --work-dir $workdir

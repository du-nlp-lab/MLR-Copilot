#!/bin/bash

name=$1
mode=$2

prob="problems/$name"
logdir="logs/$name"
workdir="workspaces/$name"

if [ -d $logdir ]; then
    echo "Folder $logdir already exists. removing it"
    rm -rf $logdir
fi
mkdir -p $logdir

python -u -m reactagent.runner --research-problem "$(< $prob)" --log-dir $logdir --work-dir $workdir

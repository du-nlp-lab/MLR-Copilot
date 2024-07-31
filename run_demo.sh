#!/bin/bash

name=$1
model=$2

prob="/app/problems/$name"
logdir="/app/logs/$name"
workdir="/app/workspaces/$name"

if [[ $model = llama ]]; then
    llm='Meta-Llama-3.1-8B-Instruct'
    fastllm=$llm
elif [[ $model = claude ]]; then
    llm='claude-3-5-sonnet-20240620'
    fastllm='claude-3-haiku-20240307'
elif [[ $model = gpt4 ]]; then
    llm='gpt-4o-2024-05-13'
    fastllm='gpt-4o-mini-2024-07-18'
fi

echo using llm = $llm
echo using fastllm = $fastllm

if [ -d $logdir ]; then
    echo "Folder $logdir already exists. removing it"
    rm -rf $logdir
fi
mkdir -p $logdir

python -u -m reactagent.runner \
--research-problem "$(< $prob)" \
--log-dir $logdir \
--work-dir $workdir \
--llm-name $llm \
--edit-script-llm-name $llm \
--fast-llm-name $fastllm

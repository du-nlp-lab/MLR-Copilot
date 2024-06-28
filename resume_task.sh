#!/bin/bash
task=$1
step=$2

python -u -m MLAgentBench.runner --research-problem MLAgentBench/benchmarks/$task/scripts/research_problem.txt --work-dir MLAgentBench/benchmarks/$task/env --log-dir logs/$task --resume logs/$task --resume-step $step --retrieval

#!/bin/bash
task=$1

python -u -m MLAgentBench.runner --research-problem MLAgentBench/benchmarks/$1/scripts/research_problem.txt --work-dir MLAgentBench/benchmarks/$1/env --log-dir logs/$1

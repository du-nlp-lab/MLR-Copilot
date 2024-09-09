# MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aMh94R1Nl6r0wTzRVJFzsx-S3pwadmFD?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2310.01783-b31b1b.svg)](https://www.arxiv.org/abs/2408.14033)

This repo provides the source code of our paper: [MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents.](https://www.arxiv.org/abs/2408.14033)
[[PDF]](https://www.arxiv.org/pdf/2408.14033)[[Twitter]](https://x.com/Xinya16/status/1829962980999774481)[[Demo]](https://huggingface.co/spaces/du-lab/MLR-Copilot) If you discuss or use MLR-Copilot in your research, please cite us!
```bibtex
@misc{li2024mlrcopilotautonomousmachinelearning,
      title={MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents}, 
      author={Ruochen Li and Teerth Patel and Qingyun Wang and Xinya Du},
      year={2024},
      eprint={2408.14033},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.14033}, 
}
```

##

MLR-Copilot is a framework where LLMs mimic researchers’ thought processes, designed to enhance the productivity of machine learning research by automating the generation and implementation of research ideas.

It begins with a research paper, autonomously generating and validating these ideas, while incorporating human feedback to help reach executable research outcomes.



## Demo  ([Link](https://huggingface.co/spaces/du-lab/MLR-Copilot))

https://github.com/user-attachments/assets/1259e2ad-efc8-4a3c-bd4d-d604c46ebd55


## Framework Overview

MLR-Copilot operates in three integrated phases:

1. **Research Idea Generation**: LLM-powered agents generate research hypotheses and experimental plans based on existing research papers.
2. **Experiment Implementation**: Translates experimental plans into executable experiments using retrieved prototype code and models.
3. **Implementation Execution**: Runs the experiments with mechanisms for human feedback and iterative debugging.
![MLR-Copilot Framework](gif/figure1.gif)
*Figure 1: The autonomous machine learning research task. We take the research paper as input and output the research idea (i.e., research hypothesis and experiment plan) with execution results.*

![MLR-Copilot Process](gif/figure2.gif)
*Figure 2: Our MLR-Copilot Framework. LLM IdeaAgent (leftmost grey component) performs research idea generation, including hypothesis and experimental design (Stage 1). ExperimentAgent implements and executes the experiments.*

## Quick Start

[Open in Colab](https://colab.research.google.com/drive/1aMh94R1Nl6r0wTzRVJFzsx-S3pwadmFD?usp=sharing)


### Setup

Begin by cloning this repository.

#### LLM Configuration

- Place the following in a `.env` file at the root of this project:
  - `CLAUDE_API_KEY`
  - `OPENAI_API_KEY`

- Configure the Hugging Face Token as needed so that `huggingface_hub.login()` works if you intend to use Llama.

#### Local Version

- Install requirements: `pip install -r requirements.txt`

#### Docker Version

- Obtain the Docker image `tortcode/nlp-coresearcher`:
  - Build: `docker build . -t 'tortcode/nlp-coresearcher'`
  - Or pull from Docker Hub: `docker pull 'tortcode/nlp-coresearcher'`
- Run `bash container.sh` to start the container.

## Experimentation

### Task Creation

- Place the research idea in the file `problems/<task_name>`.
- Run any preparation scripts as needed.
- Place all starter code in the directory `workspaces/<task_name>`.

### Task Execution

- To run the agent with a specific task and LLM (Claude, GPT-4, or Llama), execute `bash run_demo.sh <task_name> <llm_name>`.
  - You must have access to the Meta Llama 3.1 models in Hugging Face to run Llama.
- To ignore error logging, redirect stderr to `/dev/null`: `bash run_demo.sh <task_name> <llm_name> 2>/dev/null`.

### Task Logs

- Full logs are under `logs/<task_name>/<start_timestamp>/agent_log/full_log.jsonl`.
- Other logs are under `logs/<task_name>/<start_timestamp>/env_log/`.

## License

MLR-Copilot is adapted from [MLAgentBench](https://github.com/snap-stanford/MLAgentBench), under the [MIT License](https://github.com/snap-stanford/MLAgentBench/blob/main/LICENSE).

Some components are adapted from [Prompt2Model](https://github.com/neulab/prompt2model/), under the [Apache License 2.0](https://github.com/neulab/prompt2model/blob/main/LICENSE). Files utilizing API calls have been modified.

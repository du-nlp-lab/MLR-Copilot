# Title Here - Conduct Experimentation Based on Research Idea

## Setup
Begin by cloning this repo.
### LLM Configuration
- Place the following as needed into a .env file at the root of this project:
  - `CLAUDE_API_KEY`
  - `OPENAI_API_KEY`
- Configure the Hugging Face Token as needed so that `huggingface_huh.login()` works if you intend to use Llama.
### Local version
- Install requirements: `pip install -r requirements.txt`
### Docker version
- Obtain the docker image `tortcode/nlp-coresearcher` in either way:
  - Build: `docker build . -t 'tortcode/nlp-coresearcher'`, or
  - Pull from Docker Hub: `docker pull 'tortcode/nlp-coresearcher'`
- Run `bash container.sh` to start container

## Experimentation
### Task Creation
- Place the research idea in the file problems/<task_name>
- Run any preparation scripts as needed
- Place all starter code in the directory workspaces/<task_name>
### Task Execution
- To run the agent with a specific task and llm (claude, gpt4, or llama), run `bash run_demo.sh <task_name> <llm_name>`
  - You must be granted access to the Meta Llama 3.1 models in Hugging Face to run llama
- You can ignore the error logging by redirecting stderr to the null file: `bash run_demo.sh <task_name> <llm_name> 2>/dev/null`
### Task Logs
- The full log can be found under logs/<task_name>/<start_timestamp>/agent_log/full_log.jsonl
- Various other logs are under logs/<task_name>/<start_timestamp>/env_log/

## License
reactagent is adapted from [MLAgentBench](https://github.com/snap-stanford/MLAgentBench), license [MIT License](https://github.com/snap-stanford/MLAgentBench/blob/main/LICENSE).

prompt2model components have been adapted from original [prompt2model](https://github.com/neulab/prompt2model/), license [Apache License 2.0](https://github.com/neulab/prompt2model/blob/main/LICENSE).
Any files utilizing API calls have been modified.

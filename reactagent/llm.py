""" This file contains the code for calling all LLM APIs. """

from pathlib import Path
from .schema import TooLongPromptError, LLMError
from functools import partial
from transformers import AutoTokenizer
import transformers
import torch
import os
import time


try:
    import os
    from huggingface_hub import login
    login(os.environ["HF_TOKEN"])
except Exception as e:
    print(e)
    print("Could not load hugging face token HF_TOKEN from environ")

try:
    import anthropic
    # setup anthropic API key
    anthropic_client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'])
except Exception as e:
    print(e)
    print("Could not load anthropic API key CLAUDE_API_KEY from environ")

try:
    import openai
    openai_client = openai.OpenAI()
except Exception as e:
    print(e)
    print("Could not load OpenAI API key i_key.txt.")

class LlamaAgent:
    def __init__(
        self,
        model_name,
        temperature: float = 0.5,
        top_p: float = None,
        max_batch_size: int = 1,
        max_gen_len = 2000,
    ):
        model = f"meta-llama/{model_name}"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len

    def complete_text(
        self,
        prompts: list[str],
        max_gen_len=None,
        temperature=None,
        top_p=None,
        num_responses=1,
    ) -> str:
        if max_gen_len is None:
            max_gen_len = self.max_gen_len
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        results = []
        for prompt in prompts:
            seqs = self.pipeline(
                [{"role": "user", "content": prompt}],
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_responses,
                max_new_tokens=max_gen_len,
            )
            seqs = [s["generated_text"][-1]["content"] for s in seqs]
            results += seqs
        return results

agent_cache = {}

def complete_text_openai(prompt, stop_sequences=[], model="gpt-3.5-turbo", max_tokens_to_sample=500, temperature=0.2):
    """ Call the OpenAI API to complete a prompt."""
    raw_request = {
          "model": model,
          "temperature": temperature,
          "max_tokens": max_tokens_to_sample,
          "stop": stop_sequences or None,  # API doesn't like empty list
    }
    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(messages=messages, **raw_request)
    completion = response["choices"][0]["message"]["content"]
    return completion

def complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5):
    """ Call the Claude API to complete a prompt."""

    ai_prompt = anthropic.AI_PROMPT
    try:
        while True:
            try:
                message = anthropic_client.messages.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model,
                    stop_sequences=stop_sequences,
                    temperature=temperature,
                    max_tokens=max_tokens_to_sample,
                )
            except anthropic.RateLimitError:
                time.sleep(0.1)
                continue
            except anthropic.InternalServerError as e:
                pass
            try:
                completion = message.content[0].text
                break
            except:
                print("end_turn???")
                pass
    except anthropic.APIStatusError as e:
        print(e)
        raise TooLongPromptError()
    except Exception as e:
        raise LLMError(e)

    return completion

def complete_multi_text(
    prompts: str, model: str, 
    max_tokens_to_sample=None, 
    temperature=0.5,
    top_p=None,
    responses_per_request=1,
) -> list[str]:
    """ Complete text using the specified model with appropriate API. """
    if model.startswith("claude"):
        completions = []
        for prompt in prompts:
            for _ in range(responses_per_request):
                completion = complete_text_claude(
                    prompt,
                    stop_sequences=[anthropic.HUMAN_PROMPT, "Observation:"],
                    temperature=temperature,
                    model=model,
                )
                completions.append(completion)
        return completions
    elif model.startswith("gpt"):
        completions = []
        for prompt in prompts:
            for _ in range(responses_per_request):
                completion = complete_text_openai(
                    prompt,
                    stop_sequences=[anthropic.HUMAN_PROMPT, "Observation:"],
                    temperature=temperature,
                    model=model,
                )
                completions.append(completion)
        return completions
    else: #llama
        if model not in agent_cache:
            agent_cache[model] = LlamaAgent(model_name=model)

        completions = []
        try:
            completions = agent_cache[model].complete_text(
                prompts=prompts,
                num_responses=responses_per_request,
                max_gen_len=max_tokens_to_sample,
                temperature=temperature,
                top_p=top_p
            )
            for _ in range(responses_per_request):
                completions += agent_cache[model].complete_text(
                    prompts=prompts,
                )
        except Exception as e:
            raise LLMError(e)

        return completions

def complete_text(
    prompt: str, model: str, 
    max_tokens_to_sample=2000, 
    temperature=0.5,
    top_p=None,
) -> str:
    completion = complete_multi_text(
        prompts=[prompt],
        model=model,
        max_tokens_to_sample=max_tokens_to_sample,
        temperature=temperature,
        top_p=top_p,
    )[0]

    return completion

# specify fast models for summarization etc
FAST_MODEL = "Meta-Llama-3.1-8B-Instruct"
def complete_text_fast(prompt, *args, **kwargs):
    return complete_text(prompt, model=FAST_MODEL, *args, **kwargs)


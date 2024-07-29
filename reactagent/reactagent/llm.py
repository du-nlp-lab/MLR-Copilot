""" This file contains the code for calling all LLM APIs. """

from pathlib import Path
from .schema import TooLongPromptError, LLMError
from functools import partial
from transformers import AutoTokenizer
import transformers
import torch


class LlamaAgent:
    def __init__(
        self,
        model_name = "CodeLlama-13b-Python",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_batch_size: int = 8,
        max_gen_len = 2000,
    ):
        model = f"meta-llama/{model_name}-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
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
            results += self.pipeline(
                prompt,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_responses,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=max_gen_len,
            )
        return [r["generated_text"] for r in results]

agent_cache = {}

def complete_multi_text(
    prompts: str, model: str, 
    max_tokens_to_sample=None, 
    temperature=None,
    top_p=None,
    responses_per_request=1,
) -> list[str]:
    """ Complete text using the specified model with appropriate API. """
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
    max_tokens_to_sample=None, 
    temperature=None,
    top_p=None,
) -> str:
    """ Complete text using the specified model with appropriate API. """
    if model not in agent_cache:
        agent_cache[model] = LlamaAgent(model_name=model)

    try:
        completion = agent_cache[model].complete_text(
            prompts=[prompt],
            max_gen_len=max_tokens_to_sample,
            temperature=temperature,
            top_p=top_p
        )[0]
    except Exception as e:
        raise LLMError(e)

    return completion

# specify fast models for summarization etc
FAST_MODEL = "CodeLlama-13b-Python"
complete_text_fast = partial(complete_text, model=FAST_MODEL)


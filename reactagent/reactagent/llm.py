""" This file contains the code for calling all LLM APIs. """

from pathlib import Path
from llama import Llama
import tiktoken
import anthropic
from .schema import TooLongPromptError, LLMError
from functools import partial

enc = tiktoken.get_encoding("cl100k_base")

class LlamaAgent:
    def __init__(
        self,
        model_name = "CodeLlama-70b-Python",
        temperature: float = 0.4,
        top_p: float = 0.95,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len = 2000,
    ):
        ckpt_dir = Path("codellama", model_name)
        tokenizer_path = ckpt_dir / "tokenizer.model"
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
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
    ) -> str:
        if max_gen_len is None:
            max_gen_len = self.max_gen_len
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        results = self.generator(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return [r["generation"] for r in results]

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
        for _ in range(responses_per_request):
            completions += agent_cache[model].complete_text(
                prompts=prompts,
                max_gen_len=max_tokens_to_sample,
                temperature=temperature,
                top_p=top_p
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
FAST_MODEL = ""
complete_text_fast = partial(complete_text, model=FAST_MODEL)


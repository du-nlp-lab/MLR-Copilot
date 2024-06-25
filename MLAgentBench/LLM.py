""" This file contains the code for calling all LLM APIs. """

import os
from functools import partial
import tiktoken
from .schema import TooLongPromptError, LLMError

enc = tiktoken.get_encoding("cl100k_base")

try:   
    import anthropic
    # setup anthropic API key
    anthropic_client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
except Exception as e:
    print(e)
    print("Could not load anthropic API key")
    
def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")


def complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, messages=None, **kwargs):
    """ Call the Claude API to complete a prompt."""

    ai_prompt = anthropic.AI_PROMPT
    if "ai_prompt" in kwargs is not None:
        ai_prompt = kwargs["ai_prompt"]

    
    try:
        if model == "claude-3-opus-20240229":
            while True:
                try:
                    message = anthropic_client.messages.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ] if messages is None else messages,
                        model=model,
                        stop_sequences=stop_sequences,
                        temperature=temperature,
                        max_tokens=max_tokens_to_sample,
                        **kwargs
                    )
                except anthropic.InternalServerError as e:
                    pass
                try:
                    completion = message.content[0].text
                    break
                except:
                    print("end_turn???")
                    pass
        else:
            rsp = anthropic_client.completions.create(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {ai_prompt}",
                stop_sequences=stop_sequences,
                model=model,
                temperature=temperature,
                max_tokens_to_sample=max_tokens_to_sample,
                **kwargs
            )
            completion = rsp.completion
        
    except anthropic.APIStatusError as e:
        print(e)
        raise TooLongPromptError()
    except Exception as e:
        raise LLMError(e)

    
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def complete_text(prompt, log_file, model, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    
    if model.startswith("claude"):
        # use anthropic API
        completion = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
    else:
        raise ValueError(f"Model {model} not supported. Only Claude available.")
    return completion

# specify fast models for summarization etc
FAST_MODEL = "claude-v1"
def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, **kwargs)


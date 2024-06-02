from importlib import reload

import ipdb
import persist_to_disk as ptd
import torch

from ._load_model import (
    _load_pretrained_model,
    _load_pretrained_tokenizer,
    load_tokenizer_with_prefix_space,
)
from .openai_models import openai_query


def _normalize_model_name(model_name):
    if model_name == "llama2-13b":
        return "meta-llama/Llama-2-13b-hf"
    if model_name in {"gemma-2b", "gemma-7b"}:
        return f"google/{model_name}"
    if model_name == "mistral-7b":
        return "mistralai/Mistral-7B-v0.1"
    return model_name


def load_model_and_tokenizer(model_name="llama2-13b", device="cuda:0", **kwargs):
    model_name = _normalize_model_name(model_name)
    if model_name is None:
        return None, None
    return _load_pretrained_model(
        model_name, device, **kwargs
    ), _load_pretrained_tokenizer(model_name)


def load_tokenizer(model_name="llama2-13b"):
    model_name = _normalize_model_name(model_name)
    if model_name is None:
        return None
    return _load_pretrained_tokenizer(model_name)


@ptd.persistf(switch_kwarg="cache", hashsize=10000)
def llama2_completion(input_text, model_name="meta-llama/Llama-2-70b-hf"):
    from transformers import GenerationConfig

    generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        model_name, do_sample=True, return_unused_kwargs=True
    )
    model, tokenizer = load_model_and_tokenizer(model_name, None)
    # tokenizer = load_tokenizer(model_name)
    eos_token_id = [tokenizer(_)["input_ids"][-1] for _ in ["\n", "."]]
    generation_config.eos_token_id = eos_token_id + [generation_config.eos_token_id]

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        outputs = model.generate(generation_config=generation_config, **inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text) :]


@torch.no_grad()
def model_completion(input_text, model_name, device="cuda:0", **kwargs):
    model_name = _normalize_model_name(model_name)
    from transformers import GenerationConfig

    generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        model_name, do_sample=True, return_unused_kwargs=True, **kwargs
    )
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    # tokenizer = load_tokenizer(model_name)
    eos_token_id = [tokenizer(_)["input_ids"][-1] for _ in ["\n", "."]]
    generation_config.eos_token_id = eos_token_id + [generation_config.eos_token_id]

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        outputs = model.generate(generation_config=generation_config, **inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text) :]


def get_tokens_as_list(word_list, model_name, add_prefix_space=True):
    # https://huggingface.co/docs/transformers/en/internal/generation_utils
    # "Converts a sequence of words into a list of tokens"
    model_name = _normalize_model_name(model_name)
    if add_prefix_space:
        tokenizer = load_tokenizer_with_prefix_space(model_name)
    else:
        tokenizer = load_tokenizer(model_name)
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list


@torch.no_grad()
def retrieve_choice_helper(
    prompt: str, model, tokenizer, choices=["A", "B"], logsm=False
):
    assert prompt.endswith("(")
    toks = [tokenizer.encode(f"({_}")[-1] for _ in choices]
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    res = model(input_ids, output_hidden_states=True)
    if logsm:
        logits = torch.nn.functional.log_softmax(res["logits"][0][-1], 0)
    else:
        logits = res["logits"][0][-1]
    ret = logits[toks].detach().cpu()

    _sorted_logits = logits.sort()
    if ret.min() < _sorted_logits.values[-30].detach().cpu():
        print(
            f"Warning: {choices} are not the most likely: {tokenizer.convert_ids_to_tokens(_sorted_logits.indices[-30:])}"
        )
    return ret

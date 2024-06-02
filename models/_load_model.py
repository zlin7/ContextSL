# This script exists just to load models faster
import functools
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@functools.lru_cache(maxsize=1)
def _load_pretrained_model(model_name, device, torch_dtype=torch.float16):
    print(f"Reading model={model_name} pid={os.getpid()} on device={device}")
    if model_name in {
        "microsoft/deberta-large-mnli",
        "cross-encoder/stsb-roberta-large",
    }:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif model_name == "roberta-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-mnli"
        )  # , torch_dtype=torch_dtype)
    elif model_name == "meta-llama/Llama-2-13b-hf" or model_name == "llama2-13b-hf":
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-hf", cache_dir=None, torch_dtype=torch_dtype
        )
    elif model_name.startswith("meta-llama"):
        if "70b" in model_name:
            assert device is None
            return AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map="auto"
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
    elif model_name.startswith("google/"):
        assert model_name in {"google/gemma-7b", "google/gemma-2b"}
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
    elif model_name == "mistralai/Mistral-7B-v0.1":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}")

        if device is None or device == "auto":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        model.eval()
    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name in {
        "microsoft/deberta-large-mnli",
        "cross-encoder/stsb-roberta-large",
    }:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_name == "roberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    elif model_name.startswith("meta-llama"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name.startswith("google/"):
        assert model_name in {"google/gemma-7b", "google/gemma-2b"}
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    elif model_name == "mistralai/Mistral-7B-v0.1":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Unknown model_name={model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

    return tokenizer


@functools.lru_cache()
def load_tokenizer_with_prefix_space(model_name="openai-community/gpt2"):
    return AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)


if __name__ == "__main__":
    model = "google/gemma-7b"
    model = "mistralai/Mistral-7B-v0.1"
    # pass
    tokenizer = _load_pretrained_tokenizer(model)
    model_obj = _load_pretrained_model(model, "cuda:0")

import functools

import datasets
import ipdb
import pandas as pd

import models


def sample_to_prompt(sample, **kwargs):
    if isinstance(sample["question"], list):
        return [sample_to_prompt({"question": _}, **kwargs) for _ in sample["question"]]
    return f"""Answer these questions:

*Question*: In Scotland a bothy/bothie is a?
*Answer*: House
*Question*: {sample['question']}
*Answer*:"""


@functools.lru_cache()
def preprocess_data(tokenizer, split="validation"):
    data = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split)
    id_mem = set()

    def remove_dups(batch):
        if batch["question_id"][0] in id_mem:
            return {_: [] for _ in batch.keys()}
        id_mem.add(batch["question_id"][0])
        return batch

    data = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    assert pd.Series([_["question_id"] for _ in data]).value_counts().max() == 1

    def process_data_to_model_inputs(example):
        example["id"] = example["question_id"]
        example["additional_answers"] = example["answer"]["aliases"]
        example["answer"] = example["answer"]["value"]
        example["prompt"] = sample_to_prompt({k: example[k] for k in ["question"]})
        inputs = tokenizer(example["prompt"], padding=False, truncation=False)
        example["input_ids"] = inputs["input_ids"]
        example["attention_mask"] = inputs.attention_mask
        return example

    data = data.map(
        process_data_to_model_inputs,
        load_from_cache_file=False,
        remove_columns=["search_results", "question_source", "entity_pages"],
    )
    data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        output_all_columns=True,
    )
    return data


def generate_config(input_ids, model_name, data_name):
    import dataeval.coqa_new

    return dataeval.coqa_new.generate_config(input_ids, model_name, data_name)


if __name__ == "__main__":
    dataset = preprocess_data(models.load_tokenizer())

import functools

import datasets
import ipdb
import numpy as np

import models


@functools.lru_cache()
def get_fs_samples_prompt():
    data = datasets.load_dataset("nq_open", split="train")
    indices = np.random.RandomState(42).choice(len(data), 5)
    ret = ""
    for i in indices:
        i = int(i)
        ret += (
            "\n*Question*: "
            + data[i]["question"]
            + "\n*Answer*: "
            + data[i]["answer"][0]
        )
    return ret


def sample_to_prompt(sample, **kwargs):
    if isinstance(sample["question"], list):
        return [sample_to_prompt({"question": _}, **kwargs) for _ in sample["question"]]
    return f"""Answer these questions:
{get_fs_samples_prompt()}
*Question*: {sample['question']}
*Answer*:"""


def preprocess_data(tokenizer):
    # For Natural Questions we use the test split used for open-domain question answering containing 3610 questions.
    data = datasets.load_dataset("nq_open", split="validation")
    id_map = {_["question"]: str(i) for i, _ in enumerate(data)}

    def process_instance(example):
        example["id"] = id_map[example["question"]]
        all_answers = example.pop("answer")
        example["additional_answers"] = all_answers[1:]
        example["answer"] = all_answers[0]
        example["prompt"] = sample_to_prompt({k: example[k] for k in ["question"]})
        inputs = tokenizer(example["prompt"], padding=False, truncation=False)
        example["input_ids"] = inputs["input_ids"]
        example["attention_mask"] = inputs.attention_mask
        return example

    data = data.map(process_instance, load_from_cache_file=False)
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

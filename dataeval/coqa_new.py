import functools
import json
import os

import datasets
import pandas as pd
from datasets import Dataset

import _settings
import models


def _save_dataset():
    save_path = f"{_settings.DATA_FOLDER}/coqa_dataset_cat_new"
    if not os.path.exists(save_path):
        # https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json
        with open(f"{_settings.DATA_FOLDER}/coqa-dev-v1.0.json", "r") as infile:
            data = json.load(infile)["data"]

        dataset = {}

        dataset["story"] = []
        dataset["question"] = []
        dataset["answer"] = []
        dataset["additional_answers"] = []
        dataset["id"] = []

        for sample_id, sample in enumerate(data):
            story = sample["story"]
            questions = sample["questions"]
            answers = sample["answers"]
            additional_answers = sample["additional_answers"]

            story = f"""Read the context and answer the questions below.

*Context*: {sample["story"]}
"""
            for question_index, question in enumerate(questions):
                dataset["story"].append(story)
                dataset["question"].append(question["input_text"])
                dataset["answer"].append(
                    {
                        "text": answers[question_index]["input_text"],
                        "answer_start": answers[question_index]["span_start"],
                    }
                )
                dataset["id"].append(sample["id"] + "_" + str(question_index))
                additional_answers_list = []

                for i in range(3):
                    additional_answers_list.append(
                        additional_answers[str(i)][question_index]["input_text"]
                    )

                dataset["additional_answers"].append(additional_answers_list)
                story = f"""{story}
*Question*: {question["input_text"]}
*Answer*: {answers[question_index]["input_text"]}"""

        dataset_df = pd.DataFrame.from_dict(dataset)

        dataset = Dataset.from_pandas(dataset_df)

        dataset.save_to_disk(save_path)
    return save_path


@functools.lru_cache(1)
def read_all_contexts(cat=False):
    dataset = datasets.load_from_disk(_save_dataset())
    return {_["id"]: _["story"] for _ in dataset}


def preprocess_data(tokenizer):
    dataset = datasets.load_from_disk(_save_dataset())

    def encode_coqa(example):
        example["answer"] = example["answer"]["text"]
        example["prompt"] = prompt = f"""{example["story"]}
*Question*: {example["question"]}
*Answer*:"""
        return tokenizer(prompt, truncation=False, padding=False)

    dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True
    )

    return dataset


def generate_config(input_ids, model_name, data_name):
    tokenizer = models.load_tokenizer(model_name)
    if tokenizer.__class__.__name__ == "LlamaTokenizer":
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ["\n", "\n\n"]]
    elif tokenizer.__class__.__name__ in {"GemmaTokenizer"}:
        eos_token_id = [
            tokenizer.encode(_, add_special_tokens=False)[0] for _ in ["\n", "\n\n"]
        ]
    else:
        raise NotImplementedError
    eos_token_id += [tokenizer.eos_token_id]
    bad_words_ids = models.get_tokens_as_list(
        ["*Question*:", "*Answer*:", "*", "\xa0"], model_name
    )
    if tokenizer.__class__.__name__ == "LlamaTokenizer":
        if "Mistral" in tokenizer.name_or_path:
            bad_words_ids.append([29000])
        elif "Llama-2" in tokenizer.name_or_path:
            bad_words_ids.append([30081])
    if "llama" in model_name and "llama2" not in model_name:
        max_new_tokens = max(64, 2048 - len(input_ids[0]))
    else:
        max_new_tokens = 256
    return dict(
        eos_token_id=eos_token_id,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":
    import models

    dataset = preprocess_data(models.load_tokenizer())

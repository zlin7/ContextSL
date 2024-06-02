import argparse
import glob
import json
import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
else:
    print(
        f"CUDA_VISIBLE_DEVICES already set to {os.environ['CUDA_VISIBLE_DEVICES']} ({__file__})"
    )
from importlib import reload

import ipdb
import pandas as pd
import torch
import tqdm
import transformers

import _settings
import models
import utils
from dataeval import coqa_new, nq_open_new, triviaqa_new

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--dataset", type=str, default="coqa_new")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--fraction_of_data_to_use", type=float, default=1.0)
parser.add_argument("--num_generations_per_prompt", type=int, default=20)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--decoding_method", type=str, default="greedy")
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--nprocess", type=int, default=None)
parser.add_argument("--debug", action="store_true")


args = parser.parse_args()


def get_dataset_module(data_name):
    return {
        "coqa_new": coqa_new,
        "nq_open_new": nq_open_new,
        "triviaqa_new": triviaqa_new,
    }[data_name]


def get_dataset_fn(data_name):
    return get_dataset_module(data_name).preprocess_data


def get_generation_config(input_ids, model_name, data_name):
    assert len(input_ids.shape) == 2
    tokenizer = models.load_tokenizer(model_name)
    max_length_of_generated_sequence = 256
    generation_config = get_dataset_module(data_name).generate_config(
        input_ids, model_name, data_name
    )
    generation_config.setdefault("max_new_tokens", max_length_of_generated_sequence)
    generation_config["early_stopping"] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config["pad_token_id"] = tokenizer.eos_token_id
    return generation_config


@torch.no_grad()
def get_generations(
    model_name: str,
    args,
    seed=10,
    old_sequences=None,
    max_num_gen_once=2,
    cache_dir=None,
):
    device = args.device

    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer)
    data_module = get_dataset_module(args.dataset)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(
            test_size=(1 - args.fraction_of_data_to_use), seed=seed
        )["train"]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
        if cache_dir is not None and os.path.exists(
            os.path.join(cache_dir, f"partial.pkl")
        ):
            old_sequences = pd.read_pickle(os.path.join(cache_dir, f"partial.pkl"))
    old_sequences = {_["id"]: _ for _ in old_sequences}

    sequences = []
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch["id"][0] in old_sequences:
            sequences.append(old_sequences[batch["id"][0]])
            continue

        input_ids = batch["input_ids"].to(device)
        generation_config = get_generation_config(input_ids, model_name, args.dataset)
        generation_config = transformers.GenerationConfig(**generation_config)
        if model_name == "llama-13b":
            input_ids = input_ids[
                :, input_ids.shape[1] + generation_config.max_new_tokens - 2048 :
            ]
            batch["attention_mask"] = batch["attention_mask"][:, -input_ids.shape[1] :]
        input_length = input_ids.shape[1]
        if args.decoding_method == "beam_search":
            raise NotImplementedError()
        elif args.decoding_method == "greedy":
            most_likely_generations = model.generate(
                input_ids,
                attention_mask=batch["attention_mask"].to(device),
                num_beams=1,
                do_sample=False,
                generation_config=generation_config,
            ).cpu()[0, input_length:]
        generations = []
        num_gens = args.num_generations_per_prompt
        while num_gens > 0:
            _ = model.generate(
                input_ids,
                attention_mask=batch["attention_mask"].to(device),
                num_beams=1,
                num_return_sequences=min(max_num_gen_once, num_gens),
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                generation_config=generation_config,
            )
            generations.append(_[:, input_length:].cpu())
            num_gens -= len(_)

        generations = torch.nested.nested_tensor(generations).to_padded_tensor(
            tokenizer.eos_token_id
        )
        generations = generations.reshape(-1, generations.shape[-1])[
            : args.num_generations_per_prompt
        ]
        generated_texts = [
            tokenizer.decode(_, skip_special_tokens=True) for _ in generations
        ]
        # remember the data
        curr_seq = dict(
            prompt=batch["input_ids"].cpu()[0],
            id=batch["id"][0],
            question=batch["question"][0],
            answer=batch["answer"][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids=most_likely_generations,
                generations_ids=generations,
            )
        )
        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(
                    curr_seq["most_likely_generation_ids"], skip_special_tokens=True
                ),
                generations=generated_texts,
            )
        )
        if args.dataset.startswith("coqa"):
            curr_seq["additional_answers"] = [x[0] for x in batch["additional_answers"]]

        if hasattr(data_module, "retrieve_choice_with_cot"):
            most_likely_generation_choice = data_module.retrieve_choice_with_cot(
                curr_seq["most_likely_generation"], batch, model, tokenizer
            )
            generations_choice = [
                data_module.retrieve_choice_with_cot(_, batch, model, tokenizer)
                for _ in curr_seq["generations"]
            ]
            curr_seq["final_choice"] = dict(
                most_likely_generation=most_likely_generation_choice,
                generations=generations_choice,
            )

            if "ref_chain_of_thought" in batch:
                curr_seq["ref_chain_of_thought"] = batch["ref_chain_of_thought"][0]
                ref_chain_of_thought_choice = data_module.retrieve_choice_with_cot(
                    curr_seq["ref_chain_of_thought"], batch, model, tokenizer
                )
                curr_seq["final_choice"]["ref_chain_of_thought"] = (
                    ref_chain_of_thought_choice
                )
            curr_seq = data_module.clean_choice(curr_seq, batch, model, tokenizer)
        if args.debug:
            print(tokenizer.decode(curr_seq["prompt"], skip_special_tokens=True))
            ipdb.set_trace()
        sequences.append(curr_seq)

        if cache_dir is not None and len(sequences) % 50 == 0:
            pd.to_pickle(sequences, os.path.join(cache_dir, f"partial.pkl"))
    return sequences


def main(overwrite=False, parallel: int = None):
    model_name_f = model_name = args.model
    if "/" in model_name:
        model_name_f = model_name.replace("/", "_")
    if args.temperature != 1.0:
        cache_dir = os.path.join(
            _settings.GENERATION_FOLDER,
            f"{model_name_f}_{args.dataset}_{args.seed}_{args.temperature}",
        )
    else:
        cache_dir = os.path.join(
            _settings.GENERATION_FOLDER, f"{model_name_f}_{args.dataset}_{args.seed}"
        )
    os.makedirs(cache_dir, exist_ok=True)
    old_results = glob.glob(os.path.join(cache_dir, "*.pkl"))
    old_results = [_ for _ in old_results if "partial" not in _]
    if len(old_results) > 0 and not overwrite:
        print(f"Found {len(old_results)} generations in {cache_dir}.")
        return
    run_id = len(old_results)

    with open(os.path.join(cache_dir, f"args{run_id}.json"), "w") as f:
        json.dump(args.__dict__, f)
    print(
        f"Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}..."
    )
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    print(f"Using GPU={args.device}")
    sequences = get_generations(
        model_name,
        args,
        seed=args.seed,
        cache_dir=cache_dir,
    )
    print(f"Writing {len(sequences)} generations to {cache_dir}...")
    pd.to_pickle(sequences, os.path.join(cache_dir, f"{run_id}.pkl"))
    return


if __name__ == "__main__":
    import time

    from huggingface_hub.utils import HfHubHTTPError, hf_raise_for_status

    while True:
        try:
            task_runner = main(parallel=args.nprocess)
            break
        except HfHubHTTPError:
            pass
        time.sleep(10)

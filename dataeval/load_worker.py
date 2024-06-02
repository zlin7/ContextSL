import functools
import os
from collections import defaultdict
from importlib import reload
from typing import Dict, List, Optional, Tuple, Union

import ipdb
import numpy as np
import pandas as pd
import torch
import tqdm

import _settings
import models
import models.nli as sc
import utils
import utils.nlg as unlg

IGNORE_INDEX = -100


def _clean_sample_new(sample, tokenizer, logger=None):
    import unicodedata

    # https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/clean_generated_strings.py
    def _clean_answer(old_text: str, old_token_ids, tokenizer):
        assert tokenizer.decode(old_token_ids, skip_special_tokens=True) == old_text
        assert not tokenizer.bos_token_id in old_token_ids
        assert not tokenizer.eos_token_id in old_token_ids

        # assert '\xa0' not in old_text
        cleaned_text = unicodedata.normalize("NFKD", old_text).strip("\n")
        strings_to_filter_on = [
            "\n",
            "*Context*",
            "*Question*",
            "*Answer*",
            "*Explanation*",
        ]
        for string in strings_to_filter_on:
            if string in cleaned_text:
                if len(cleaned_text.split(string)[0]):
                    cleaned_text = cleaned_text.split(string)[0]
        if len(cleaned_text) == 0:
            return dict(
                text_cleaned=cleaned_text,
                text=old_text,
                token=old_token_ids.cpu(),
                token_cleaned=torch.tensor([]).long().cpu(),
            )

        if tokenizer.__class__.__name__ == "GemmaTokenizer":
            if cleaned_text.startswith(":"):
                cleaned_text = cleaned_text[1:]
            if not cleaned_text.startswith(" "):
                cleaned_text = " " + cleaned_text
                logger.warning(f"Fix bad old text ||{old_text}||")
            assert cleaned_text[0] in {" "}
        elif tokenizer.__class__.__name__ == "LlamaTokenizer":
            if cleaned_text.startswith(" "):
                if old_text[0] != "\xa0":
                    logger.warning(f"Fix bad old text ||{old_text}||")
                cleaned_text = cleaned_text.lstrip()
        token_ids = tokenizer.encode(
            cleaned_text, return_tensors="pt", add_special_tokens=False
        )[0]
        token_ids = tokenizer.encode(
            tokenizer.decode(token_ids, skip_special_tokens=True),
            return_tensors="pt",
            add_special_tokens=False,
        )[0]
        assert (
            tokenizer.decode(token_ids) == cleaned_text
        ), f"{tokenizer.decode(token_ids)} != {cleaned_text}"
        # if tokenizer.decode(token_ids) != cleaned_text:
        #    logger.error(f"{tokenizer.decode(token_ids)} != {cleaned_text}")
        # ipdb.set_trace()
        return dict(
            text_cleaned=cleaned_text,
            token_cleaned=token_ids.cpu(),
            text=old_text,
            token=old_token_ids.cpu(),
        )

    ret = {
        k: sample[k]
        for k in ["prompt", "id", "question", "answer", "additional_answers"]
    }

    def _clean_tokens(g):
        if tokenizer.pad_token_id is not None:
            g = g[g.ne(tokenizer.pad_token_id)]
        g = g[g.ne(tokenizer.eos_token_id)]
        return g

    ret["generations"] = [
        _clean_answer(sample["generations"][i], _clean_tokens(_), tokenizer)
        for i, _ in enumerate(sample["generations_ids"])
    ]
    ret["most_likely_generation"] = _clean_answer(
        sample["most_likely_generation"],
        _clean_tokens(sample["most_likely_generation_ids"]),
        tokenizer,
    )
    ret["generations"] = {
        k: [v[k] for v in ret["generations"]] for k in ret["generations"][0].keys()
    }
    for _ in ["ref_chain_of_thought", "final_choice"]:
        if _ in sample:
            ret[_] = sample[_]
    return ret


def _get_model_eval_sample_prompt(
    row, text_key=None, ith=None, dataset: str = None, few_shots=None
):
    if text_key is None:
        pred = row["pred"]
    else:
        pred = row["generations"][text_key][ith]
    if few_shots is None:
        dataset = dataset or row["dataset"]
        few_shots = {
            "triviaqa": [
                {
                    "question": "In Scotland a bothy/bothie is a?",
                    "reference": "House",
                    "answer": "House",
                },
                {
                    "question": "Where in England was Dame Judi Dench born?",
                    "reference": "York",
                    "answer": "London",
                },
            ],
            "nq_open": [
                {
                    "question": "who makes up the state council in russia",
                    "reference": "governors and presidents",
                    "answer": "governors and presidents",
                },
                {
                    "question": "when does real time with bill maher come back",
                    "reference": "November 9, 2018",
                    "answer": "September 8, 2000",
                },
            ],
            "coqa": [
                {
                    "question": "When was the Vat formally opened?",
                    "reference": "It was formally established in 1475",
                    "answer": "In 1475",
                },
                {
                    "question": "what is the library for?",
                    "reference": "research",
                    "answer": "tourism",
                },
            ],
        }
        for _ in ["triviaqa", "nq_open", "coqa"]:
            few_shots[f"{_}_new"] = few_shots[_]
        few_shots = few_shots[dataset]
    prompt = f"""Rate the level of consistency between the answer to the question and the reference answer, from 0 to 100.
Question: {few_shots[0]['question']}
Reference: {few_shots[0]['reference']}
Answer: {few_shots[0]['answer']}
Rating: 100.

Question: {few_shots[1]['question']}
Reference: {few_shots[1]['reference']}
Answer: {few_shots[1]['answer']}
Rating: 0.

Question: {row['question']}
Reference: {row['answer']}
Answer: {pred.strip()}
Rating:"""
    return prompt


def _get_model_eval_sample(
    row, text_key=None, ith=None, dataset: str = None, few_shots=None
):
    prompt = _get_model_eval_sample_prompt(row, text_key, ith, dataset, few_shots)
    assert row["model_name"] == "meta-llama/Llama-2-70b-hf"
    return models.llama2_completion(
        prompt, model_name="meta-llama/Llama-2-70b-hf"
    ).strip()


def _get_api_eval_sample(
    row, text_key=None, ith=None, dataset: str = None, few_shots=None
):
    from openai.error import InvalidRequestError

    prompt = _get_model_eval_sample_prompt(row, text_key, ith, dataset, few_shots)
    model = row["model_name"]
    if model == "gpt-3.5-turbo-0125":
        try:
            return models.openai_query(prompt, model=model, attemptd_id=0, max_tries=50)
        except InvalidRequestError:
            print(f"Prompt too long: {len(prompt)}")
            return models.openai_query(
                prompt[:10000], model=model, attemptd_id=0, max_tries=50
            )
    else:
        raise ValueError(f"Unknown model: {model}")


def _get_model_eval_general(
    samples,
    clean: bool,
    ith: int,
    dataset: str,
    model: str,
    logger=None,
):
    model = {
        "llama2": "meta-llama/Llama-2-70b-hf",
        "gpt": "gpt-3.5-turbo-0125",
    }[model]
    text_key = "text_cleaned" if clean else "text"
    df = pd.DataFrame(
        {
            key: [sample[key] for sample in samples]
            for key in ["id", "answer", "question"]
        }
    )
    df["ith"] = ith
    df["text_key"] = text_key
    df["dataset"] = dataset
    if ith is None:
        df["pred"] = [sample["most_likely_generation"][text_key] for sample in samples]
    else:
        df["pred"] = [sample["generations"][text_key][ith] for sample in samples]
    df["model_name"] = model
    ret = {}

    if model in {"gpt-3.5-turbo-0125"}:
        func = _get_api_eval_sample
    elif model == "meta-llama/Llama-2-70b-hf":
        func = _get_model_eval_sample
    else:
        raise ValueError(f"Unknown model: {model}")
    for i in tqdm.tqdm(np.random.permutation(len(df))):
        ret[df.iloc[i]["id"]] = func(df.iloc[i])
    ret = pd.Series(ret).reindex(df["id"])
    return ret.values.tolist()


# =======================loglikelihood=======================
def _compute_token_nll(model_output, prompt_len, generation):
    # log probabilities of the target words
    # Just in case the loss is not NLL for the model
    assert len(generation.shape) == 1
    _logits = model_output["logits"][0, prompt_len - 1 : -1]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    assert generation[prompt_len:].ne(IGNORE_INDEX).all()
    loss = criterion(_logits, generation[prompt_len:])
    return loss


def _compute_token_entropy(model_output, prompt_len):
    # only the geenrated words
    # NOTE: Can we include the last word here?
    _logits = model_output["logits"][0, prompt_len - 1 : -1]
    _logp = torch.nn.functional.log_softmax(_logits, dim=-1)
    _p = torch.exp(_logp)
    _token_entropy = -torch.where(_p > 0, _p * _logp, 0).sum(1)  # avoid -inf
    # higher -> more uncertain
    return _token_entropy


def _compute_token_mean(embedding, prompt_len):
    # only the geenrated words
    # NOTE: Can we include the last word here? If so, replace -1 with None
    _embedding = embedding[0, prompt_len - 1 : -1]
    return _embedding.mean(0)


def _create_output_prompt(model, tokenizer, prompt):
    prompt = prompt.to(model.device)
    assert 1 == len(prompt.shape) and prompt.ne(tokenizer.pad_token_id).all()
    model_output = model(
        prompt.unsqueeze(0), output_hidden_states=True, labels=prompt.unsqueeze(0)
    )
    token_nll = _compute_token_nll(model_output, 1, prompt)
    token_entropy = _compute_token_entropy(model_output, 1)
    sequence_embedding = _compute_token_mean(model_output["hidden_states"][-1], 1)
    return dict(
        neg_log_likelihood=token_nll.sum().item(),
        length=len(prompt),
        token_nll=token_nll.cpu(),
        token_entropy=token_entropy.cpu(),
        sequence_embedding=sequence_embedding.cpu(),
    )


@torch.no_grad()
def _create_output_from_generation(model, tokenizer, generation, prompt):
    prompt = prompt.to(model.device)
    if len(generation) == 0:
        generation = torch.tensor([], dtype=prompt.dtype, device=model.device)
    else:
        assert generation.dtype == prompt.dtype == torch.long
    generation = torch.concat([prompt, generation.to(model.device)])
    prompt_len = len(prompt)
    assert len(generation.shape) == 1 == len(prompt.shape)
    generation = generation[generation.ne(tokenizer.pad_token_id)]
    generation_only = generation.clone()[prompt_len - 1 :]  # with one token prefix
    generation = generation.clone()

    model_output = model(generation.unsqueeze(0), output_hidden_states=True)
    unconditioned_model_output = model(
        generation_only.unsqueeze(0),
        output_hidden_states=True,
        labels=generation_only.unsqueeze(0),
    )

    token_nll = _compute_token_nll(model_output, prompt_len, generation)
    unconditioned_token_nll = _compute_token_nll(
        unconditioned_model_output, 1, generation_only
    )
    token_entropy = _compute_token_entropy(model_output, prompt_len)
    unconditioned_token_entropy = _compute_token_entropy(unconditioned_model_output, 1)

    # embedding
    sequence_embedding = _compute_token_mean(
        model_output["hidden_states"][-1], prompt_len
    )
    unconditioned_sequence_embedding = _compute_token_mean(
        unconditioned_model_output["hidden_states"][-1], 1
    )
    return dict(
        neg_log_likelihood=token_nll.sum().item(),
        unconditioned_neg_log_likelihood=unconditioned_token_nll.sum().item(),
        length=len(generation) - prompt_len,
        #
        token_nll=token_nll.cpu(),  # .numpy(),
        unconditioned_token_nll=unconditioned_token_nll.cpu(),  # .numpy(),
        token_entropy=token_entropy.cpu(),  # .numpy(),
        unconditioned_token_entropy=unconditioned_token_entropy.cpu(),  # .numpy(),
        # embeddings
        sequence_embedding=sequence_embedding.cpu(),  # .numpy(),
        unconditioned_sequence_embedding=unconditioned_sequence_embedding.cpu(),  # .numpy(),
    )


@torch.no_grad()
def _get_loglikelihoods(
    samples, model, tokenizer, clean: bool, logger=None, old_res=None
):
    token_key = "token_cleaned" if clean else "token"
    ret = []
    for i, sample in tqdm.tqdm(enumerate(samples), total=len(samples)):
        curr_summ = {"id": sample["id"]}

        prompt = sample["prompt"].to(model.device)
        assert prompt.ne(tokenizer.pad_token_id).all() and len(prompt.shape) == 1
        curr_summ["prompt"] = _create_output_prompt(model, tokenizer, prompt)

        if old_res is not None:
            assert sample["id"] == old_res[i]["id"]
            curr_summ["generations"] = old_res[i]["generations"]
        else:
            sampled_summ = [
                _create_output_from_generation(model, tokenizer, _, prompt)
                for _ in sample["generations"][token_key]
            ]
            curr_summ["generations"] = {
                k: [_[k] for _ in sampled_summ] for k in sampled_summ[0].keys()
            }
            for _ in ["sequence_embedding", "unconditioned_sequence_embedding"]:
                curr_summ["generations"][_] = torch.stack(curr_summ["generations"][_])
        curr_summ["most_likely_generation"] = _create_output_from_generation(
            model, tokenizer, sample["most_likely_generation"][token_key], prompt
        )
        if "ref_chain_of_thought" in sample:
            curr_summ["ref_chain_of_thought"] = _create_output_from_generation(
                model,
                tokenizer,
                tokenizer.encode(
                    sample["ref_chain_of_thought"],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).squeeze(0),
                prompt,
            )
        ret.append(curr_summ)
    return ret


# ======================================================== Attention based loglikelihoods


@torch.no_grad()
def _get_attn_loglikelihoods_single_batched(row):
    layer_heads = row.get("layer_heads", None)
    device = utils.gpuid_to_device(row["device"])
    model, tokenizer = models.load_model_and_tokenizer(row["model_name"], device)
    errs = defaultdict(list)

    tdf = [
        {
            "id": "most_likely_generation",
            "tokens": row["most_likely_generation"],
            "token_nll": row["most_likely_generation|token_nll"],
        }
    ]
    for ith_gen in range(len(row["generations"])):
        tdf.append(
            {
                "id": ith_gen,
                "tokens": row["generations"][ith_gen],
                "token_nll": row["generations|token_nll"][ith_gen],
            }
        )
    tdf = pd.DataFrame(tdf)
    tdf["gen_key"] = tdf["tokens"].apply(lambda x: tuple(x.tolist()))
    token_nll_map = {}
    for i, _curr in tdf.iterrows():
        if _curr["gen_key"] not in token_nll_map:
            token_nll_map[_curr["gen_key"]] = _curr["token_nll"]
        if not torch.allclose(
            token_nll_map[_curr["gen_key"]], _curr["token_nll"], atol=1e-2, rtol=1e-2
        ):
            errs[_curr["id"]].append(f"Token NLLs do not match for {_curr['gen_key']}")

    unique_df = tdf.drop_duplicates("gen_key").set_index("gen_key")
    unique_df["ith"] = range(len(unique_df))
    unique_df["results"] = unlg._compute_attn_weighted_sum_batched(
        unique_df["token_nll"].tolist(),
        [row["prompt"]] * len(unique_df),
        unique_df["tokens"].tolist(),
        model,
        tokenizer,
        has_context=row["dataset"].startswith("coqa"),
        layer_heads=layer_heads,
        next_prompt=row.get("next_prompt", False),
    )
    for i, _curr in unique_df.iterrows():
        errs[_curr["id"]].extend(_curr["results"][1])
    mapping = unique_df["ith"]
    mapping = {_["id"]: mapping[_["gen_key"]] for _ in tdf.to_dict(orient="records")}
    errs = {k: v for k, v in errs.items() if len(v) > 0}
    return dict(
        id=row["id"],
        mapping=mapping,
        attn_loglikelihoods={
            i: _[0] for i, _ in enumerate(unique_df["results"].tolist())
        },
        errs=errs,
    )


# Baselines below=============================================================================


@torch.no_grad()
def _get_self_eval_sample(row, text_key, dataset, model, tokenizer, logsm=False):
    anss = [
        _.lstrip()
        for _ in row["generations"][text_key]
        + [row["most_likely_generation"][text_key]]
    ]
    unique_answers = set(anss)
    few_shots = "\n".join(list(unique_answers)[:10])
    story = ""
    if dataset == "coqa":
        raise ValueError("coqa is not supported")
    elif dataset == "coqa_new":
        import dataeval.coqa_new

        story = dataeval.coqa_new.read_all_contexts()[row["id"]] + "\n"
    A_tok = tokenizer.encode("(A")[-1]
    B_tok = tokenizer.encode("(B")[-1]

    ret = {}
    for _ans in unique_answers:
        prompt = f"""{story}Question: {row['question']}
Here are some brainstormed ideas: {few_shots}
Possible Answer: {_ans}
Is the possible answer:
(A) True
(B) False
The possible answer is: ("""
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(model.device)
        res = model(input_ids, output_hidden_states=True)
        if logsm:
            logits = torch.nn.functional.log_softmax(res["logits"][0][-1], 0)
        else:
            logits = res["logits"][0][-1]
        # check if A and B are high likelihood
        ret[_ans] = logits[[A_tok, B_tok]].detach().cpu()
        _sorted_logits = logits.sort()
        if ret[_ans].min() < _sorted_logits.values[-30].detach().cpu():
            print(
                f"Warning: A and B are not the most likely: {tokenizer.convert_ids_to_tokens(_sorted_logits.indices[-30:])}"
            )
    return dict(
        id=row["id"],
        logits=pd.DataFrame(
            torch.stack([ret[_] for _ in anss]).cpu().numpy(),
            columns=["True", "False"],
            index=list(range(len(anss) - 1)) + ["most_likely_generation"],
        ),
    )


@torch.no_grad()
def _get_self_eval(
    samples, model, tokenizer, clean: bool, dataset: str, logger=None, logsm=False
):
    text_key = "text_cleaned" if clean else "text"
    ret = []
    for _ in tqdm.tqdm(samples):
        ret.append(
            _get_self_eval_sample(_, text_key, dataset, model, tokenizer, logsm=logsm)
        )
        continue
    return ret


def _compute_token_relevance_single(sample):
    device = utils.gpuid_to_device(sample["device"])
    tokenizer = models.load_tokenizer(sample["model_name"])
    sc_model = sc.ClassifyWrapper(sample["judge_model"], device=device)

    _mem = {}  # mem[s] is the result for generation s

    def _compute_one_gen(s_tokens, question=sample["question"]):
        s_tokens = s_tokens.tolist()
        key = tuple(s_tokens)
        if key not in _mem:
            sen_2 = [
                f"{question} {tokenizer.decode(s_tokens[:i] + s_tokens[i+1:])}"
                for i in range(len(s_tokens))
            ]
            sen_1 = [f"{question} {tokenizer.decode(s_tokens)}"] * len(sen_2)
            if len(sen_1) == 0:
                _mem[key] = torch.tensor([])
            else:
                _mem[key] = sc_model._batch_pred(sen_1, sen_2).cpu()[:, 0]
        return _mem[key]

    ret = dict(
        generations=[_compute_one_gen(_) for _ in sample["generations"]],
        most_likely_generation=_compute_one_gen(sample["most_likely_generation"]),
        id=sample["id"],
    )
    # check length
    if "lll_sample" in sample:
        lll_sample = sample["lll_sample"]
        assert all(
            [
                len(_[0]) == len(_[1])
                for _ in zip(lll_sample["generations"]["token_nll"], ret["generations"])
            ]
        )
        assert len(lll_sample["most_likely_generation"]["token_nll"]) == len(
            ret["most_likely_generation"]
        )
    return ret


if __name__ == "__main__":
    pass

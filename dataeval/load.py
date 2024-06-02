import functools
import getpass
import os
import re
from importlib import reload
from typing import Dict, List, Optional, Tuple, Union

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import simple_disk_queue as sdq
import tqdm

import dataeval.load_worker as lw

reload(lw)
import models
import models.nli as sc
import utils

_GLOBAL_LOCAL_CACHE_FLAG = False


def _get_model_name(path: str):
    base_fnames = os.path.basename(path).split("_")
    if base_fnames[-1] == "generations.pkl":
        return base_fnames[0]
    path = os.path.basename(os.path.dirname(path))
    path = path.replace("google_gemma", "google/gemma")
    path = path.replace("mistralai_Mistral", "mistralai/Mistral")
    return path.split("_")[0]


def _get_dataset_name(path: str):
    for _ in ["triviaqa_new", "nq_open_new", "coqa_new"]:
        if _ in path:
            return _
    raise ValueError(f"Cannot find dataset name in {path}")


def _partition_sequence(data, ith, npartitions, seed):
    # filter df - dividing df into partition[1] chunks and only taking partition[0]th chunk
    chosen_ids = np.random.RandomState(seed).permutation(len(data))
    chosen_ids = [_ for _ in chosen_ids if _ % npartitions == ith]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[chosen_ids]
    elif isinstance(data, list):
        data = [data[_] for _ in chosen_ids]
    return data


def _read_diskqueue_wrapper(
    _worker,
    key,
    debug=False,
    readonly=False,
    checkonly=False,
    queue: sdq.DiskQueue = None,
    num_partitions: int = 8,
    stacklevel=2,  # save_list=True,
    **kwargs,
):
    if readonly or checkonly:
        return ptd.manual_cache(
            key,
            local=_GLOBAL_LOCAL_CACHE_FLAG,
            checkonly=checkonly,
            stacklevel=stacklevel,
        )
    res = (
        None
        if debug
        else ptd.manual_cache(
            key, local=_GLOBAL_LOCAL_CACHE_FLAG, stacklevel=stacklevel
        )
    )
    if res is None:
        if debug:
            assert "device" in kwargs
            return _worker(partition=(0, 1000), cache=0, **kwargs)
        count = 0
        for i in range(num_partitions):
            try:
                _worker(partition=(i, num_partitions), device=0, cache=3, **kwargs)
                count += 1
            except Exception as e:
                queue.add_task(
                    _worker, partition=(i, num_partitions), device=0, **kwargs
                )
        if count < num_partitions:
            return queue.id
        if (count == num_partitions) or (queue is None):
            kwargs.pop("device", None)
            res = {}
            for i in range(num_partitions):
                _tres = _worker(partition=(i, num_partitions), device=0, **kwargs)
                if isinstance(_tres, list):
                    _tres = {_["id"]: _ for _ in _tres}
                res.update(_tres)
            ptd.manual_cache(
                key,
                res,
                write=not debug,
                local=_GLOBAL_LOCAL_CACHE_FLAG,
                stacklevel=stacklevel,
            )
    return res


def get_key_from_generated_strings_path_new(path, clean):
    run_id = os.path.basename(path).replace(".pkl", "")
    specs = os.path.basename(os.path.dirname(path))
    return f"{specs}_{run_id}" + ("_cleaned" if clean else "")


@functools.lru_cache(maxsize=4)
def read_cleaned_outputs(path, readonly=False):
    # Re-organize the result and include the "cleaned" answers.
    # This is the same as the semantic entropy paper.
    # The post-processing is a bit ugly, but somewhat unavoidable because
    # differnt tokens could lead to the same output character (like \n),
    # so simply specifying the tokens in generation config is not enough.
    key = get_key_from_generated_strings_path_new(path, False)
    dataset = _get_dataset_name(path)
    cleaned_sequences = ptd.manual_cache(key, local=_GLOBAL_LOCAL_CACHE_FLAG)
    if readonly:
        return cleaned_sequences
    if cleaned_sequences is None:
        sequences = utils.cached_read_pickle(path)
        tokenizer = models.load_tokenizer(_get_model_name(path))
        logger = utils.get_logger(
            f"read_cleaned_outputs#{key}",
            os.path.join(ptd.get_caller_cache_path(), "logs", f"{key}.log"),
            stream_handler=True,
        )

        cleaned_sequences = [
            lw._clean_sample_new(sample, tokenizer, logger=logger)
            for sample in tqdm.tqdm(sequences)
        ]
        ptd.manual_cache(
            key, obj=cleaned_sequences, write=True, local=_GLOBAL_LOCAL_CACHE_FLAG
        )
    return cleaned_sequences


@functools.lru_cache(maxsize=4)
def read_model_eval_general(
    path: str,
    clean=True,
    debug=False,
    ith=0,
    readonly=False,
    checkonly=False,
    model="llama2",
):
    assert model in {"llama2", "gpt"}
    key = get_key_from_generated_strings_path_new(path, clean=clean)
    key += f"_{ith}_{model}"
    if checkonly:
        return ptd.manual_cache(
            key, local=_GLOBAL_LOCAL_CACHE_FLAG, checkonly=checkonly
        )
    evals = None if debug else ptd.manual_cache(key, local=_GLOBAL_LOCAL_CACHE_FLAG)
    if readonly and evals is None:
        return evals
    if evals is None:
        cleaned_sequences = read_cleaned_outputs(path)[: 5 if debug else None]
        dataset = _get_dataset_name(path)
        evals = lw._get_model_eval_general(
            cleaned_sequences,
            clean,
            ith,
            dataset=dataset,
            model=model,
        )
        evals = {_["id"]: _eval for _, _eval in zip(cleaned_sequences, evals)}
        ptd.manual_cache(key, evals, write=not debug, local=_GLOBAL_LOCAL_CACHE_FLAG)
    if model == "llama2":

        def _parse(v):
            if len(v) == 0 or v == ".":
                return np.NaN
            if all([len(_) == 0 for _ in v.split(".")]):
                return np.NaN
            return v.split(".")[0].split()[0]

        evals = {k: _parse(v) for k, v in evals.items()}
    else:
        evals = {
            k: v if len(v) == 0 else v.split(".")[0].split()[0]
            for k, v in evals.items()
        }

    ret = {}
    for k, val in evals.items():
        try:
            val = int(val)
            assert 0 <= val <= 100
            ret[k] = val
        except:
            ret[k] = np.NaN

    if pd.Series(ret).count() < len(ret):
        print(
            f"Warning: {path.replace(getpass.getuser(), 'USERNAME')} has {len(ret) - pd.Series(ret).count()} NaNs"
        )
    return pd.Series(ret) / 100.0


@functools.lru_cache(maxsize=4)
def read_loglikelihoods_and_more(
    path: str,
    device=None,
    clean=True,
    debug=False,
    readonly=True,
    checkonly=False,
):
    device = utils.gpuid_to_device(device)
    key = get_key_from_generated_strings_path_new(path, clean=clean)
    if checkonly or readonly:
        return ptd.manual_cache(
            key, local=_GLOBAL_LOCAL_CACHE_FLAG, checkonly=checkonly
        )

    likelihoods = (
        None if debug else ptd.manual_cache(key, local=_GLOBAL_LOCAL_CACHE_FLAG)
    )
    if likelihoods is None:
        cleaned_sequences = read_cleaned_outputs(path)
        if debug:
            cleaned_sequences = cleaned_sequences[:5]
        logger = utils.get_logger(
            f"read_loglikelihoods_and_more#{key}",
            os.path.join(ptd.get_caller_cache_path(), f"{key}.log"),
            stream_handler=True,
        )
        model, tokenizer = models.load_model_and_tokenizer(
            _get_model_name(path), device
        )
        likelihoods = lw._get_loglikelihoods(
            cleaned_sequences,
            model,
            tokenizer,
            clean=clean,
            logger=logger,
        )
        ptd.manual_cache(
            key, likelihoods, write=not debug, local=_GLOBAL_LOCAL_CACHE_FLAG
        )
    return likelihoods


def read_self_eval(
    path: str, device=None, clean=True, debug=False, readonly=False, checkonly=False
):
    # used in the P(true) baseline
    device = utils.gpuid_to_device(device)
    key = get_key_from_generated_strings_path_new(path, clean=clean)
    if checkonly or readonly:
        return ptd.manual_cache(
            key, local=_GLOBAL_LOCAL_CACHE_FLAG, checkonly=checkonly
        )
    results = None if debug else ptd.manual_cache(key, local=_GLOBAL_LOCAL_CACHE_FLAG)
    if results is None:
        cleaned_sequences = read_cleaned_outputs(path)[: 5 if debug else None]
        model, tokenizer = models.load_model_and_tokenizer(
            _get_model_name(path), device
        )
        dataset = _get_dataset_name(path)
        results = lw._get_self_eval(
            cleaned_sequences,
            model,
            tokenizer,
            clean=clean,
            dataset=dataset,
        )
        ptd.manual_cache(key, results, write=not debug, local=_GLOBAL_LOCAL_CACHE_FLAG)
    return results


@ptd.persistf(skip_kwargs=["device"], switch_kwarg="cache", lock_granularity="call")
def _get_attn_loglikelihoods_diskqueue(
    path: str,
    partition: Tuple[int, int],
    device: Optional[int] = 0,
    clean=True,
    seed: int = 7,
    all_heads=False,
    *,
    next_prompt=False,
):
    dataset = _get_dataset_name(path)
    model_name = _get_model_name(path)
    ll_samples = read_loglikelihoods_and_more(
        path, device=None, clean=clean, debug=False, readonly=True
    )
    samples = read_cleaned_outputs(path)

    token_key = "token_cleaned" if clean else "token"
    df = pd.DataFrame({key: [sample[key] for sample in samples] for key in ["id"]})
    df["generations"] = [sample["generations"][token_key] for sample in samples]
    df["generations|token_nll"] = [
        sample["generations"]["token_nll"] for sample in ll_samples
    ]
    df["most_likely_generation"] = [
        sample["most_likely_generation"][token_key] for sample in samples
    ]
    df["most_likely_generation|token_nll"] = [
        sample["most_likely_generation"]["token_nll"] for sample in ll_samples
    ]
    df["prompt"] = [sample["prompt"] for sample in samples]
    df["model_name"] = model_name
    df["dataset"] = dataset
    assert not isinstance(device, tuple)
    df["device"] = device
    df["layer_heads"] = "all" if all_heads else None
    if next_prompt:
        df["next_prompt"] = True

    chosen_ids = np.random.RandomState(seed).permutation(len(df))
    chosen_ids = [_ for _ in chosen_ids if _ % partition[1] == partition[0]]
    df = df.iloc[chosen_ids]
    ret = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"attn_lll_{partition}"):
        ret.append(lw._get_attn_loglikelihoods_single_batched(row))
    return ret


@ptd.persistf(switch_kwarg="cache", skip_kwargs=["device"])
def _read_token_relevance_diskqueue(
    path: str,
    partition: Tuple[int, int],
    device=None,
    judge_model: str = "cross-encoder/stsb-roberta-large",
    clean=True,
    seed: int = 7,
):
    ll_samples = read_loglikelihoods_and_more(
        path, None, clean, debug=False, readonly=True
    )
    samples = read_cleaned_outputs(path)
    model_name = _get_model_name(path)

    token_key = "token_cleaned" if clean else "token"
    df = pd.DataFrame({key: [sample[key] for sample in samples] for key in ["id"]})
    df["generations"] = [sample["generations"][token_key] for sample in samples]
    df["question"] = [sample["question"] for sample in samples]
    df["most_likely_generation"] = [
        sample["most_likely_generation"][token_key] for sample in samples
    ]
    df["judge_model"] = judge_model
    df["model_name"] = model_name
    if ll_samples is not None:
        assert all([_[0]["id"] == _[1]["id"] for _ in zip(ll_samples, samples)])
        df["lll_sample"] = ll_samples
    assert not isinstance(device, tuple)
    df["device"] = device
    ret = []

    # filter df - dividing df into partition[1] chunks and only taking partition[0]th chunk
    chosen_ids = np.random.RandomState(seed).permutation(len(df))
    chosen_ids = [_ for _ in chosen_ids if _ % partition[1] == partition[0]]
    df = df.iloc[chosen_ids]
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        ret.append(lw._compute_token_relevance_single(row))
    return ret


@ptd.persistf(switch_kwarg="cache", skip_kwargs=["device"])
def _read_semantic_similarities(
    path: str,
    partition: Tuple[int, int],
    clean=True,
    device=0,
    judge_model: str = "microsoft/deberta-large-mnli",
    seed: int = 7,
):
    samples = read_cleaned_outputs(path)
    samples = _partition_sequence(samples, partition[0], partition[1], seed)
    sc_model = sc.ClassifyWrapper(judge_model, device=device)
    text_key = "text_cleaned" if clean else "text"
    semantic_sims = {}
    for _ in tqdm.tqdm(samples, desc="computing similarities"):
        anss = [_["most_likely_generation"][text_key]] + _["generations"][text_key]
        if "ref_chain_of_thought" in _:
            anss = [_["ref_chain_of_thought"]] + anss
        _tres = sc_model.create_sim_mat_batched(_["question"], anss)
        _tres["id"] = _["id"]
        semantic_sims[_["id"]] = _tres
    return semantic_sims


@functools.lru_cache(maxsize=4)
def read_semantic_similarities(
    path: str,
    clean=True,
    judge_model: str = "microsoft/deberta-large-mnli",
    queue: sdq.DiskQueue = None,
    **kwargs,
) -> Dict:
    import dataeval.load as dl

    key = get_key_from_generated_strings_path_new(path, clean=False)
    key += f"_model={judge_model.replace('/', '#')}"
    if clean:
        key += "_cleaned"
    semantic_sims = _read_diskqueue_wrapper(
        dl._read_semantic_similarities,
        key,
        queue=queue,
        path=path,
        clean=clean,
        judge_model=judge_model,
        **kwargs,
    )
    if isinstance(semantic_sims, dict):
        mlg_loc = 0
        for _, v in semantic_sims.items():
            v["mapping"] = {
                "generations": v["mapping"][mlg_loc + 1 :],
                "most_likely_generation": v["mapping"][mlg_loc],
            }
        assert len(v["mapping"]["generations"]) == 20
    return semantic_sims


@functools.lru_cache(maxsize=4)
def read_attn_loglikelihoods_all(
    path: str, clean=True, queue: sdq.DiskQueue = None, **kwargs
):
    if not read_loglikelihoods_and_more(path, clean=clean, debug=False, checkonly=True):
        return
    key = get_key_from_generated_strings_path_new(path, clean=clean)
    import dataeval.load as dld

    return _read_diskqueue_wrapper(
        dld._get_attn_loglikelihoods_diskqueue,
        key,
        queue=queue,
        path=path,
        clean=clean,
        all_heads=True,
        **kwargs,
    )


@functools.lru_cache(maxsize=4)
def read_attn_loglikelihoods_all_next_token(
    path: str, clean=True, queue: sdq.DiskQueue = None, **kwargs
):
    if not read_loglikelihoods_and_more(path, clean=clean, debug=False, checkonly=True):
        return
    key = get_key_from_generated_strings_path_new(path, clean=clean)
    import dataeval.load as dld

    return _read_diskqueue_wrapper(
        dld._get_attn_loglikelihoods_diskqueue,
        key,
        queue=queue,
        path=path,
        clean=clean,
        all_heads=True,
        next_prompt=True,
        **kwargs,
    )


@functools.lru_cache(maxsize=4)
def read_token_relevance(
    path: str,
    clean=True,
    judge_model: str = "cross-encoder/stsb-roberta-large",
    queue: sdq.DiskQueue = None,
    **kwargs,
):
    key = get_key_from_generated_strings_path_new(path, clean=clean)
    key += f"_model={judge_model.replace('/', '#')}"
    import dataeval.load as dld

    return _read_diskqueue_wrapper(
        dld._read_token_relevance_diskqueue,
        key,
        queue=queue,
        path=path,
        clean=clean,
        judge_model=judge_model,
        **kwargs,
    )


if __name__ == "__main__":
    pass

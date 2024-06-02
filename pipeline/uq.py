"""Main script for UQ computation. No hyperparameter selection, just pure computation."""

import functools
import itertools
import os
from collections import defaultdict
from importlib import reload
from typing import Dict, List, Optional, Tuple

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import torch
import tqdm
from scipy.special import softmax

import _settings
import dataeval as dload
import pipeline.clustering as pc

CONTRADICT, NEUTRAL, AGREE = 0, 1, 2


def _clean_tensor_id(data):
    def _change_sample(sample):
        if isinstance(sample["id"], torch.Tensor):
            sample["id"] = sample["id"].cpu().item()
        return sample

    if isinstance(data, list):
        return [_change_sample(_) for _ in data]
    assert isinstance(data, dict)
    if isinstance(next(iter(data.keys())), torch.Tensor):
        return {k.cpu().item(): _change_sample(v) for k, v in data.items()}
    return data


def _precompute_attn_nll_helper(data):
    assert data is not None
    data = _clean_tensor_id(data)

    def convert_func(sample):
        results = {}
        for k, v in sample["attn_loglikelihoods"].items():
            w = v["token_nll"]
            v = v.drop(columns=["token_nll"])
            v = v / v.sum(0)
            results[k] = (v.T * w).sum(1)
            # sample has key: id, mapping, attn_loglikelihoods, errs
        df = pd.DataFrame(
            {_id: results[_uid] for _id, _uid in sample["mapping"].items()}
        )
        df = df.reset_index().rename(columns={"level_0": "layer", "level_1": "head"})
        df["id"] = sample["id"]
        errs = [f"{_[0]}: {_[1]}" for _ in sample["errs"].items() if len(_[1]) > 0]
        if len(errs):
            print("\n".join(errs))
        return df

    data = {k: convert_func(v) for k, v in tqdm.tqdm(data.items(), "reading attnnll")}
    return data


@functools.lru_cache(10)
def _precompute_token_rel_nll(path, clean):
    key = dload.get_key_from_generated_strings_path_new(path, clean=clean)
    if not ptd.manual_cache(key, checkonly=True):
        data = dload.read_token_relevance(path, clean=clean, readonly=True)
        assert data is not None
        ll_samples = dload.read_loglikelihoods_and_more(
            path, clean=clean, readonly=True
        )
        ll_samples = {_["id"]: _ for _ in ll_samples}
        data = _clean_tensor_id(data)
        ll_samples = _clean_tensor_id(ll_samples)

        def _sar(w, v):
            w = 1 - torch.sigmoid(w)
            w = w / w.sum()
            return (v * w).sum().cpu().item()

        def convert_func(idx):
            _w = data[idx]
            ll_sample = ll_samples[idx]
            gens = [
                _sar(_wi, _g1)
                for _wi, _g1 in zip(
                    _w["generations"], ll_sample["generations"]["token_nll"]
                )
            ]
            mlg = _sar(
                _w["most_likely_generation"],
                ll_sample["most_likely_generation"]["token_nll"],
            )
            return pd.Series(
                gens + [mlg], index=list(range(len(gens))) + ["most_likely_generation"]
            )

        data = {
            k: convert_func(k)
            for k in tqdm.tqdm(data.keys(), "reading token relevance")
        }
        data = pd.DataFrame(data).T
        ptd.manual_cache(key, data, write=True, local=False)
    return ptd.manual_cache(key, local=False)


@functools.lru_cache(10)
def _precompute_attn_nll(path, clean):
    key = dload.get_key_from_generated_strings_path_new(path, clean=clean)
    if not ptd.manual_cache(key, checkonly=True):
        data = dload.read_attn_loglikelihoods_all(path, clean=clean, readonly=True)
        ptd.manual_cache(
            key, _precompute_attn_nll_helper(data), write=True, local=False
        )
    return ptd.manual_cache(key, local=False)


@functools.lru_cache(10)
def _precompute_attn_nll_next_token(path, clean):
    key = dload.get_key_from_generated_strings_path_new(path, clean=clean)
    if not ptd.manual_cache(key, checkonly=True):
        data = dload.read_attn_loglikelihoods_all_next_token(
            path, clean=clean, readonly=True
        )
        ptd.manual_cache(
            key, _precompute_attn_nll_helper(data), write=True, local=False
        )
    return ptd.manual_cache(key, local=False)


def _clean_path(path, check=True):
    base_dir = os.path.normpath(_settings.GENERATION_FOLDER)
    path = os.path.normpath(path)
    assert (not check) or path.startswith(base_dir)
    return path[len(base_dir) :]


def recover_sim_mat(sim):
    sim_mat = sim["sim_mat"].clone()
    sim_mat[torch.arange(sim_mat.shape[0]), torch.arange(sim_mat.shape[0]), :] = (
        torch.tensor([-torch.inf, -torch.inf, 100])
    )
    mapping = [sim["mapping"]["most_likely_generation"]] + sim["mapping"]["generations"]
    # a len(ans) x len(ans) x 3 tensor
    ret = torch.zeros((len(mapping), len(mapping), 3))
    for i, ans_i in enumerate(mapping):
        for j, ans_j in enumerate(mapping):
            ret[i, j] = sim_mat[mapping[i], mapping[j]].clone().detach()
    return None, ret


def _create_semantic_sets(sample):
    # https://github.com/lorenzkuhn/semantic_uncertainty
    generated_texts = sample["mapping"]
    sim_mat = sample["sim_mat"].argmax(axis=-1)
    # unique_ans is also a list of integers.
    unique_generated_texts = sorted(list(set(generated_texts)))
    semantic_set_ids = {
        ans: i for i, ans in enumerate(unique_generated_texts)
    }  # one id for each exact-match answer
    for i, ans_i in enumerate(unique_generated_texts):
        for j, ans_j in enumerate(unique_generated_texts[i + 1 :], i + 1):
            if min(sim_mat[ans_i, ans_j], sim_mat[ans_j, ans_i]) > CONTRADICT:
                semantic_set_ids[ans_j] = semantic_set_ids[ans_i]

    list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
    # map according to the order of appearance
    _map = defaultdict(int)
    ret = []
    for i, ans in enumerate(list_of_semantic_set_ids):
        if ans not in _map:
            _map[ans] = len(_map)
        ret.append(_map[ans])
    return ret


# whitebox methods
def _logmeanexp(x, dim, ignore_negative_inf=False):
    if ignore_negative_inf:
        cnt = (x > -torch.inf).sum(dim)
    else:
        cnt = torch.tensor(x.shape[dim])
    return torch.logsumexp(x, dim=dim) - torch.log(cnt)


def _hard_semantic_entropies(neg_log_likelihoods, semantic_set_ids=None, **kwargs):
    num_samples, num_gens = neg_log_likelihoods.shape
    if semantic_set_ids is None:
        semantic_set_ids = torch.tensor(list(range(num_gens))).long()

    log_likelihoods = -neg_log_likelihoods
    # initilaize to -inf for all possible semantic ids
    max_num_semantic_ids = semantic_set_ids.max().item() + 1 + 1
    aggregated_likelihoods = torch.log(torch.zeros((num_samples, max_num_semantic_ids)))
    for semantic_set_id in torch.unique(semantic_set_ids):
        temp = torch.where(
            semantic_set_ids == semantic_set_id, log_likelihoods, -torch.inf
        )
        aggregated_likelihoods[:, semantic_set_id] = torch.logsumexp(temp, 1)
    # if torch.any(torch.isnan(_logmeanexp(aggregated_likelihoods, dim=1, ignore_negative_inf=True))):
    #    ipdb.set_trace()
    return -_logmeanexp(aggregated_likelihoods, dim=1, ignore_negative_inf=True)


class UQ_computer:
    def __init__(self, path, clean=True, split=None, cal_size=None, seed=None) -> None:
        assert isinstance(path, str), f"Path must be a string, got {path}"
        self.path = path
        self.key = (_clean_path(path), clean)
        self.generations = _clean_tensor_id(dload.read_cleaned_outputs(path))

        self.keep_indices = None
        if split is not None:
            assert (
                split in ["val", "test"] and cal_size is not None and seed is not None
            )
            self.key = (
                _clean_path(path),
                clean,
                split,
                cal_size,
                seed,
            )
            self.keep_indices = np.random.RandomState(seed).choice(
                len(self.generations), cal_size, replace=False
            )
            if split == "test":
                self.keep_indices = set(np.arange(len(self.generations))) - set(
                    self.keep_indices
                )
            self.generations = [self.generations[_] for _ in self.keep_indices]

        self.ids = [_["id"] for _ in self.generations]

    @functools.cached_property
    def similarities(self):
        sims = dload.read_semantic_similarities(
            self.path, clean=self.key[1], debug=False
        )

        sims = _clean_tensor_id(sims)
        sims = [sims[_] for _ in self.ids]
        return sims

    @functools.cached_property
    def likelihoods(self):
        assert self.path is not None, "likelihoods are not available for black-box data"
        print("load likelihoods")
        likelihoods = dload.read_loglikelihoods_and_more(
            self.path, clean=self.key[1], debug=False
        )
        # ipdb.set_trace()
        if likelihoods is not None:
            likelihoods = _clean_tensor_id(likelihoods)
            likelihoods = {_["id"]: _ for _ in likelihoods}
            likelihoods = [likelihoods[_] for _ in self.ids]
            likelihoods = self.batchify(likelihoods)
        return likelihoods

    @functools.cached_property
    def self_eval(self):
        assert (
            self.path is not None
        ), "self evaluatinn (P(true)) is not available for black-box data"
        print("load self eval")
        self_eval = dload.read_self_eval(self.path, None, self.key[1])
        self_eval = {_["id"]: _ for _ in self_eval}
        self_eval = [self_eval[_] for _ in self.ids]
        self_eval = np.stack(
            [softmax(_["logits"].values, 1)[:, 0] for _ in self_eval]  # p(true)
        )
        return self_eval[:, :-1], self_eval[:, -1]

    @functools.lru_cache(2)
    def attn_likelihoods(self, next_token=False):
        if next_token:
            data = _precompute_attn_nll_next_token(self.path, self.key[1])
        else:
            data = _precompute_attn_nll(self.path, self.key[1])
        data = [data[_] for _ in self.ids]
        df = pd.concat(data, axis=0, ignore_index=True).set_index(
            ["layer", "head", "id"]
        )
        # make df float32
        df = df.astype("float32")
        return dict(
            most_likely_generation=df.pop("most_likely_generation")
            .unstack([0, 1])
            .reindex(self.ids)
            .sort_index(axis=1),
            generations={
                k: v.droplevel([0, 1]).reindex(self.ids).copy()
                for k, v in df.sort_index(axis=1).groupby(level=[0, 1])
            },
        )

    @classmethod
    def batchify(cls, likelihoods):
        result_dict = defaultdict(list)
        to_stack = set()
        for sample in likelihoods:
            result_dict["id"].append(sample["id"])
            for pref, sub_dict in sample.items():
                if pref == "id":
                    continue
                for key, val in sub_dict.items():
                    if isinstance(val, list) and (
                        isinstance(val[0], int) or isinstance(val[0], float)
                    ):
                        val = torch.tensor(val)
                        to_stack.add(pref + "|" + key)
                    result_dict[pref + "|" + key].append(val)
        result_dict = dict(result_dict)
        for key, val in result_dict.items():
            if key in to_stack:
                result_dict[key] = torch.stack(val)
            else:
                if isinstance(val, list) and (
                    isinstance(val[0], int) or isinstance(val[0], float)
                ):
                    val = torch.tensor(val)
                result_dict[key] = val
        return result_dict

    # ========================================Derived properties========================================
    @functools.cached_property
    def _nli_logit_matrix(self):
        return [recover_sim_mat(_)[1] for _ in self.similarities]

    @functools.lru_cache()
    def _get_sim_general(self, num_gens: int, affinity_mode: str):
        return [_[: num_gens + 1, : num_gens + 1] for _ in self._nli_logit_matrix]

    @functools.lru_cache(10)
    def _get_semantic_ids(self, num_gens):
        # We must filter sims first before passing to _create_gal_semantic_ids
        sims = [
            {
                "mapping": _["mapping"]["generations"][:num_gens],
                "sim_mat": _["sim_mat"],
            }
            for _ in self.similarities
        ]
        return [_create_semantic_sets(_) for _ in sims]

    @functools.lru_cache()
    def _get_spectral_projected(
        self,
        num_gens: int,
        eigv_threshold: float,
        affinity_mode: str,
        temperature: float,
    ):
        clusterer = pc.SpetralClusteringFromLogits(
            affinity_mode=affinity_mode,
            eigv_threshold=eigv_threshold,
            cluster=False,
            temperature=temperature,
        )
        sim_mats = self._get_sim_general(num_gens, affinity_mode)
        return [
            clusterer.proj(_)
            for _ in tqdm.tqdm(sim_mats, desc="projecting", disable=True)
        ]

    def get_length(self, num_gens: int):
        text_key = "text_cleaned" if self.key[1] else "text"
        lengths = [
            [
                len(set(_.split()))
                for _ in [sample["most_likely_generation"][text_key]]
                + sample["generations"][text_key][:num_gens]
            ]
            for sample in self.generations
        ]
        lengths = np.asarray(lengths)
        return lengths[:, 1:].mean(1), lengths[:, 1:], lengths[:, 0]

    def get_degreeuq(self, num_gens: int, affinity_mode: str, temperature: float):
        sim_mats = self._get_sim_general(num_gens, affinity_mode)
        Ws = [
            pc.get_affinity_mat(_, affinity_mode, temperature, symmetric=False)
            for _ in sim_mats
        ]
        gen = np.asarray([np.sum(1 - _[1:, 1:], axis=1) for _ in Ws])
        return gen.mean(1), gen, np.stack([1 - _[0][1:] for _ in Ws]).sum(1)

    def get_selfprob(self, num_gens: int):
        neg_ci = 1 - self.self_eval[0][:, :num_gens]
        return neg_ci.mean(1), neg_ci, 1 - self.self_eval[1]

    def get_nll(self, num_gens: int, normalize: bool):
        # higher=more negative = less confident
        nlls = self.likelihoods["generations|neg_log_likelihood"][:, :num_gens]
        mlg_nll = self.likelihoods["most_likely_generation|neg_log_likelihood"]
        if normalize:
            nlls = nlls / self.likelihoods["generations|length"][:, :num_gens]
            mlg_nll = mlg_nll / self.likelihoods["most_likely_generation|length"]
        return _hard_semantic_entropies(nlls), nlls, mlg_nll

    def get_attn_nll(
        self,
        num_gens: int,
        layer_heads: Optional[Tuple[Tuple]] = None,
        average=True,
        next_token=False,
    ):
        attn_likelihoods = self.attn_likelihoods(next_token=next_token)
        if layer_heads is None:
            layer_heads = attn_likelihoods["generations"].keys()
        elif not isinstance(layer_heads[0], tuple):
            # A single layer head
            assert len(layer_heads) == 2
            layer_heads = [layer_heads]
        nlls = [
            torch.tensor(
                attn_likelihoods["generations"][layer_head].values[:, :num_gens]
            )
            for layer_head in layer_heads
        ]
        mlg_nll = torch.tensor(
            attn_likelihoods["most_likely_generation"]
            .reindex(columns=layer_heads)
            .values
        )
        if average:
            nlls = sum(nlls) / len(nlls)
            mlg_nll = mlg_nll.mean(1)
            return _hard_semantic_entropies(nlls), nlls, mlg_nll
        else:
            ret = {}
            for i, layer_head in enumerate(layer_heads):
                ret[str({"layer_heads": layer_head})] = dict(
                    u=pd.Series(_hard_semantic_entropies(nlls[i]), self.ids),
                    neg_ic=pd.DataFrame(nlls[i], index=self.ids),
                    neg_mlgc=pd.Series(mlg_nll[:, i], self.ids),
                )
            return ret

    def get_tokrel_nll(self, num_gens: int):
        res = _precompute_token_rel_nll(self.path, self.key[1]).reindex(self.ids)
        nlls = torch.tensor(res.reindex(columns=list(range(num_gens))).values).float()
        mlg_nll = torch.tensor(res["most_likely_generation"].values).float()
        return _hard_semantic_entropies(nlls), nlls, mlg_nll

    def get_semantic_entropy(
        self,
        num_gens: int,
        normalize: bool = None,
        layer_heads: Optional[Tuple[Tuple]] = None,
    ):
        if self.likelihoods is None:
            return None
        semantic_set_ids = self._get_semantic_ids(num_gens)
        if layer_heads is not None:
            nlls = [
                torch.tensor(
                    self.attn_likelihoods()["generations"][layer_head].values[
                        :, :num_gens
                    ]
                )
                for layer_head in layer_heads
            ]
            nlls = sum(nlls) / len(nlls)
        else:
            nlls = self.likelihoods["generations|neg_log_likelihood"][:, :num_gens]
            if normalize:
                nlls = nlls / self.likelihoods["generations|length"][:, :num_gens].clip(
                    1
                )
        res = _hard_semantic_entropies(nlls, torch.tensor(semantic_set_ids))
        return res, torch.stack([res] * num_gens).T, res

    def get_uq(self, name, num_gens=5, cache=None, **kwargs):
        if cache is None:
            cache = ptd.NOCACHE if name in {"debug"} else ptd.CACHE
        if self.path is None:
            cache = ptd.NOCACHE
        if name.endswith("sar") or name.startswith("attnnll"):
            cache = ptd.NOCACHE  # computation from cache is very fast
        u, neg_ic, neg_mlgc = _compute_uq_cached_with_mlg(
            self,
            self.key,
            name,
            num_gens=num_gens,
            metric_kwargs=kwargs,
            cache=cache,
        )
        assert len(u) == len(self.ids)
        assert neg_mlgc is None or len(neg_mlgc) == len(self.ids)
        # use overall for individual if not provided
        if neg_ic is None:
            neg_ic = np.tile(u, (num_gens, 1)).T
        assert neg_ic.shape[1] == num_gens and neg_ic.shape[0] == len(self.ids)
        return dict(
            u=pd.Series(np.asanyarray(u), self.ids),
            neg_ic=pd.DataFrame(np.asarray(neg_ic), index=self.ids),
            neg_mlgc=pd.Series(np.asanyarray(neg_mlgc), self.ids)
            if neg_mlgc is not None
            else None,
        )


@ptd.persistf(
    expand_dict_kwargs=["metric_kwargs"],
    skip_kwargs=["self"],
    lock_granularity="call",
    switch_kwarg="cache",
    groupby=["uq_name"],
    local=False,
)
def _compute_uq_cached_with_mlg(
    self: UQ_computer, key, uq_name, num_gens=5, metric_kwargs=None, **kwargs
) -> Tuple[torch.Tensor]:
    """Compute uncertainty and confidence measures
    :return:
        Uncertainty or None: (N,)
        Negative Confidence for generations or None: (N, m)
        Negative Confidence for most_likely_generation or None: (N,)
    """
    if metric_kwargs is None:
        metric_kwargs = {}
    if uq_name in {"sar"}:
        return self.get_tokrel_nll(num_gens)
    if uq_name.startswith("semanticEntropy"):
        return self.get_semantic_entropy(
            num_gens,
            normalize=uq_name.split("|")[1] == "norm" if "|" in uq_name else None,
            **metric_kwargs,
        )
    if uq_name.startswith("attnnll_nexttoken") or uq_name.startswith("attnnll"):
        if uq_name.startswith("attnnll_nexttoken"):
            metric_kwargs["next_token"] = True
            uq_name = uq_name.replace("attnnll_nexttoken", "attnnll")
        if uq_name == "attnnll|all":
            assert (
                "layer_heads" not in metric_kwargs
            ), "layer_heads is not allowed for attnnll|all"
        else:
            assert uq_name == "attnnll", uq_name
        return self.get_attn_nll(num_gens, **metric_kwargs)
    if uq_name.startswith("nll"):
        return self.get_nll(num_gens, normalize=uq_name.split("|")[1] == "norm")
    if uq_name.startswith("degree"):
        return self.get_degreeuq(
            num_gens,
            affinity_mode=uq_name.split("|")[1],
            temperature=metric_kwargs["temperature"],
        )
    if uq_name == "self_prob":
        return self.get_selfprob(num_gens)
    raise ValueError(f"Unknown metric {uq_name}")


if __name__ == "__main__":
    from _settings import DATASETS, GEN_PATHS, MODELS

    for temp in [0.5, 1.0]:
        for data, model in itertools.product(DATASETS, MODELS):
            path = GEN_PATHS[temp][data][model]
            try:
                _precompute_attn_nll(path, clean=True)
                _precompute_attn_nll_next_token(path, clean=True)
                _precompute_token_rel_nll(path, clean=True)
            except AssertionError as err:
                pass

    pass

"""Main script for incl hyperparameter selection and summary"""

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
import simple_disk_queue as sdq
import tqdm

import _settings
import dataeval as dload
import pipeline.eval_uq as eval_uq
import pipeline.uq as uq

_clean_path = uq._clean_path


class UQ_summ(uq.UQ_computer):
    MAX_NUM_GENS = 5
    _uq_measures = [
        "degree|agreement_w",
        "self_prob",
        "nll|unnorm",
        "nll|norm",
        "attnnll|all",
        "attnnll",
        "attnnll_nexttoken|all",
        "attnnll_nexttoken",
        "sar",
        "semanticEntropy|norm",
        "semanticEntropy|unnorm",
        "semanticEntropyFROMattnnll@10",
    ] + [
        f"attnnll{_}@{k}"
        for _ in ["", "_nexttoken"]
        for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
    ]

    tunable_hyperparams = {
        "degree|agreement_w": ["temperature"],
        "attnnll": ["layer_heads"],
        "attnnll|unnorm": ["layer_heads"],
        "attnnll_nexttoken": ["layer_heads"],
    }

    default_params = {"eigv_threshold": 0.9, "temperature": 3.0}
    whitebox_uqs = [
        "semanticEntropy|unnorm",
        "semanticEntropy|norm",
        "self_prob",
        "nll|unnorm",
        "nll|norm",
    ]

    def __init__(
        self,
        path,
        clean=True,
        split=None,
        cal_size: int = 1000,
        seed=None,
        transfer_dataset=None,
    ) -> None:
        super().__init__(path, clean, split, cal_size, seed)

        self.transfer_dataset = transfer_dataset

    @functools.cached_property
    def tune_cal_obj(self):
        if self.transfer_dataset is not None:
            dataset = dload._get_dataset_name(self.path)
            path = self.path.replace(dataset, self.transfer_dataset)
            return UQ_summ(path, self.key[1], "val", self.key[3], self.key[4])
        assert self.key[2] == "test"
        return UQ_summ(self.path, self.key[1], "val", self.key[3], self.key[4])

    @functools.lru_cache(10)
    def _get_param_ranges(self, name):
        if name == "eigv_threshold":
            return 0.9, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if name == "temperature":
            return 3.0, [0.1, 0.25, 0.5, 1, 3, 5, 7]
        if name == "layer_heads":
            all_layer_heads = list(self.attn_likelihoods()["generations"].keys())
            return None, all_layer_heads
        raise ValueError(f"Unknown param {name}")

    @functools.lru_cache()
    def _default_params(self, name):
        return {
            k: self._get_param_ranges(k)[0]
            for k in self.tunable_hyperparams.get(name, [])
        }

    @functools.cached_property
    def uq_measures(self):
        uq_measures = self._uq_measures
        return uq_measures

    # NOTE: Don't change this method
    def _param_to_perfs(
        self,
        uq_name: str = "attnnll",
        num_gens=5,
        setting: str = "ea~u",
        acc_name: str = "llama2|acc",
        curve="auarc",  # tune the hyperparams using this curve
        cache=ptd.CACHE,
    ):
        return _compute_performance_by_param(
            self, self.key, setting, uq_name, acc_name, num_gens, curve, cache=cache
        )

    @functools.lru_cache(100)
    def get_tuned_uq(
        self,
        name,
        num_gens,
        setting: str = "mlga~neg_mlgc",
        acc_name: str = "llama2|acc",
        curve="auarc",
        cache=ptd.CACHE,
    ):
        if len(self.key) == 2:
            return self.get_uq(
                name, num_gens, cache=cache, **self._default_params(name)
            )

        if (
            "FROM" in name
        ):  # sematic entropy using attnnll. The tuning is done on attnnll only
            to_name, from_name = name.split("FROM")
            from_name, topk = from_name.split("@")
            topk = int(topk)
            layer_heads = (
                self.tune_cal_obj._param_to_perfs(
                    uq_name=from_name,
                    setting=setting,
                    acc_name=acc_name,
                    num_gens=num_gens,
                    curve=curve,
                    cache=cache,
                )
                .sort_values(ascending=False)
                .iloc[:topk]
                .index
            )
            layer_heads = tuple(sorted([eval(_)["layer_heads"] for _ in layer_heads]))
            return self.get_uq(to_name, num_gens, cache=cache, layer_heads=layer_heads)

        if "@" in name:
            # instead of the best param, use the top N best params, and average the results
            name, topk = name.split("@")
            topk = int(topk)
        else:
            topk = 1
        if len(self.tunable_hyperparams.get(name, [])) == 0:
            return self.get_uq(name, num_gens, cache=cache)
        if acc_name.endswith("|score") and curve == "auroc":
            acc_name = acc_name.replace("|score", "|acc")
        perfs = (
            self.tune_cal_obj._param_to_perfs(
                uq_name=name,
                setting=setting,
                acc_name=acc_name,
                num_gens=num_gens,
                curve=curve,
                cache=cache,
            )
            .sort_values(ascending=False)
            .iloc[:topk]
        )
        ret = [self.get_uq(name, num_gens, cache=cache, **eval(_)) for _ in perfs.index]
        return {
            _key: sum([_[_key] for _ in ret]) / len(ret)
            for _key in ["u", "neg_ic", "neg_mlgc"]
        }

    @functools.cached_property
    def _mask_acc(self):
        token_key = "token_cleaned" if self.key[1] else "token"
        mask = {}
        for gen in self.generations:
            mask[gen["id"]] = pd.Series([len(_) for _ in gen["generations"][token_key]])
            mask[gen["id"]]["most_likely_generation"] = len(
                gen["most_likely_generation"][token_key]
            )
        return pd.DataFrame(mask).T

    @functools.lru_cache(10)
    def get_acc(self, acc_name="llama2|acc"):
        # returns the expected accuracy (over all generations) as well as individual accuracy
        if acc_name in {"moe|acc"}:
            llama2_acc = self.get_acc("llama2|acc")
            gpt_acc = self.get_acc("gpt|acc")
            ret = {k: v.copy() for k, v in llama2_acc.items()}
            ret["ia"][gpt_acc["ia"] != ret["ia"]] = np.nan
            ret["mlga"][gpt_acc["mlga"] != ret["mlga"]] = np.nan
            return dict(
                ea=ret["ia"].mean(1),
                ia=ret["ia"],
                mlga=ret["mlga"],
            )
        name, suffix = acc_name.split("|")
        assert name in {"gpt", "llama2"}, f"Unknown type {acc_name}"
        score_df = {}
        for ith in [None] + list(
            range(
                min(self.MAX_NUM_GENS, len(self.generations[0]["generations"]["text"]))
            )
        ):
            try:
                score_df["most_likely_generation" if ith is None else ith] = (
                    dload.read_model_eval_general(
                        self.path,
                        clean=self.key[1],
                        debug=False,
                        readonly=True,
                        ith=ith,
                        model=name,
                    )
                )
            except Exception as err:
                print(err)
                break
        score_df = pd.DataFrame(score_df).reindex(self.ids)

        indiv_acc = score_df.reindex(self.ids)
        if suffix == "acc":
            thresold = {
                "llama2": 0.2,
                "gpt": 0.6,
            }[name]
            indiv_acc = (indiv_acc >= thresold).astype(float)
        mask = self._mask_acc.reindex(indiv_acc.index, columns=indiv_acc.columns)
        indiv_acc[mask == 0] = np.nan
        mlg_acc = indiv_acc.pop("most_likely_generation")
        return dict(
            ea=indiv_acc.mean(1),
            ia=indiv_acc,
            mlga=mlg_acc,
        )

    def summ(
        self,
        setting: str,
        uq_names: Optional[List[str]] = None,
        curve: str = "auarc",
        acc_name: str = "llama2|acc",
        num_gens=5,
        uq_kwargs: dict = None,
        get_eval_obj=False,
    ):
        if uq_names is None:
            uq_names = self.uq_measures
        if isinstance(uq_names, str):
            uq_names = [uq_names]

        def _get_uq(name):
            if uq_kwargs is not None and name in uq_kwargs:
                return self.get_uq(name, num_gens, **uq_kwargs[name])
            return self.get_tuned_uq(
                name, num_gens, setting, curve=curve, acc_name=acc_name
            )

        summarizer = eval_uq.Summarizer(
            {_: _get_uq(_) for _ in uq_names},
            self.get_acc(acc_name),
            lengths=self.get_length(num_gens),
        )
        if get_eval_obj:
            return summarizer
        return summarizer.list_uq_perfs(
            y=setting.split("~")[0],
            x=setting.split("~")[1],
            metric=curve,
            keep_all_rows=True,
        )


@ptd.persistf(groupby=["acc_name"], switch_kwarg="cache", lock_granularity="call")
def cached_summ(
    path,
    clean,
    seed,
    setting,
    curve,
    num_gens,
    acc_name: str = "gpt",
    split: str = "test",
    cal_size: int = 1000,
    **kwargs,
):
    assert not acc_name.endswith(
        "|acc"
    ), f"acc_name should not end with |acc, got {acc_name}"
    transfer_dataset = kwargs.get("transfer_dataset", None)
    old_path = path
    path = f"{os.path.normpath(_settings.GENERATION_FOLDER)}{path}"
    assert _clean_path(path) == old_path
    obj = UQ_summ(path, clean, split, cal_size, seed, transfer_dataset=transfer_dataset)
    if acc_name.endswith("_score"):
        acc_name = acc_name.replace("_score", "|score")
    else:
        acc_name = f"{acc_name}|acc"
    return obj.summ(setting, curve=curve, acc_name=acc_name, num_gens=num_gens)


@ptd.persistf(
    skip_kwargs=["self"],
    switch_kwarg="cache",
    groupby=["uq_name"],
    local=False,
    lock_granularity="call",
)
def _compute_performance_by_param(
    self: UQ_summ,
    key,
    setting: str,  # ea~u, ia~neg_ic, mlga~neg_mlgc, ia~u,
    uq_name: str = "attnnll",
    acc_name: str = "llama2|acc",
    num_gens=5,
    curve="auarc",  # tune the hyperparams using this curve
):
    assert self.key == key
    tunable_params = self.tunable_hyperparams.get(uq_name, [])
    uqs = {}
    kwargs = {k: self._get_param_ranges(k)[1] for k in tunable_params}
    for _vals in itertools.product(*[kwargs[_] for _ in tunable_params]):
        _kwargs = dict(zip(tunable_params, _vals))
        uqs[str(_kwargs)] = self.get_uq(uq_name, num_gens=num_gens, **_kwargs)
    summ_obj = eval_uq.Summarizer(uqs, self.get_acc(acc_name))
    y, x = setting.split("~")
    return summ_obj.list_uq_perfs(y=y, x=x, metric=curve)


def _cache_all(
    path,
    clean,
    nseeds=10,
    split: str = "test",
    cal_size: int = 1000,
    num_gens: Tuple[int] = (3, 5),
    acc_name="gpt",
    cache=ptd.CACHE,
    queue: sdq.DiskQueue = None,
):
    print(f"Processing {path}")

    def run(func, *args, **kwargs):
        if queue is None:
            return func(*args, **kwargs)
        kwargs2 = {k: v for k, v in kwargs.items()}
        kwargs2["cache"] = ptd.READONLY
        try:
            func(*args, **kwargs2)
        except KeyboardInterrupt as e:
            raise e
        except:
            queue.add_task(func, *args, **kwargs)

    path = _clean_path(path)
    pbar = tqdm.tqdm(total=nseeds * len(num_gens) * 4 * 3, desc=path)
    for transfer_dataset in [None, "coqa_new"]:
        if transfer_dataset == dload._get_dataset_name(path):
            transfer_dataset = None
        for _num_gens in num_gens:
            for seed in range(nseeds):
                _kwargs = dict(
                    path=path,
                    clean=clean,
                    seed=seed,
                    num_gens=_num_gens,
                    split=split,
                    cal_size=cal_size,
                    acc_name=acc_name,
                    transfer_dataset=transfer_dataset,
                    cache=cache,
                )
                for setting in ["mlga~neg_mlgc", "ia~neg_ic"]:
                    run(cached_summ, setting=setting, curve="auarc", **_kwargs)
                    run(cached_summ, setting=setting, curve="auroc", **_kwargs)
                    pbar.update(1)
                run(cached_summ, setting="ia~neg_ic", curve="pickmax", **_kwargs)
                pbar.update(1)


if __name__ == "__main__":
    pass

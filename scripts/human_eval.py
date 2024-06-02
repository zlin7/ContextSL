from importlib import reload

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd

import _settings
import dataeval as dload
import dataeval.coqa_new as coqa_new
import models
import pipeline.summ as summ


@ptd.persistf(switch_kwarg="cache")
def sample(path, clean=True, n=80, seed=42):
    o = summ.UQ_summ(
        path,
        clean=clean,
        split="test",
        cal_size=1000,
        seed=(7 + seed) % 10,
    )
    acc_scores = {k: o.get_acc(f"{k}|score") for k in ["llama2", "gpt"]}
    text_key = "text_cleaned" if clean else "text"
    ret = []
    rs = np.random.RandomState(seed)
    mlg_locs = rs.choice(len(o.generations), n // 2, replace=False)
    ig_locs = rs.choice(len(o.generations), n - n // 2, replace=False)

    stories = None
    if dload._get_dataset_name(path) == "coqa_new":
        stories = coqa_new.read_all_contexts()

    tokeninzer = models.load_tokenizer(dload._get_model_name(path))

    for key, locs in [("most_likely_generation", mlg_locs), ("generations", ig_locs)]:
        for loc in locs:
            curr = o.generations[loc]
            response = curr[key][text_key]
            ret.append(
                {
                    "id": o.ids[loc],
                    "question": curr["question"],
                    "ref_answer": curr["answer"],
                }
            )
            if stories is not None:
                ret[-1]["story"] = tokeninzer.decode(
                    curr["prompt"], add_special_tokens=False
                )
            if key == "generations":
                response = response[0]
                for k, v in acc_scores.items():
                    ret[-1][f"{k}_score"] = v["ia"][0][curr["id"]]
            else:
                for k, v in acc_scores.items():
                    ret[-1][f"{k}_score"] = v["mlga"][curr["id"]]
            ret[-1]["response"] = response
    ret = pd.DataFrame(ret)
    ret["dataset"] = dload._get_dataset_name(path)
    ret["model"] = dload._get_model_name(path)
    return ret.reindex(
        columns=[
            "dataset",
            "model",
            "id",
            "story",
            "question",
            "ref_answer",
            "response",
            "llama2_score",
            "gpt_score",
        ]
    )


def sample_all(seed=0):
    dfs = []
    for dataset in ["coqa_new", "triviaqa_new", "nq_open_new"]:
        for model in ["llama2-13b", "gemma-7b", "mistral-7b"]:
            path = _settings.GEN_PATHS[0.5][dataset][model]

            dfs.append(sample(path, seed=seed, cache=1 if dataset == "coqa_new" else 1))
            seed += 1
    dfs = pd.concat(dfs, ignore_index=True)
    dfs = dfs.reindex(
        columns=[
            "dataset",
            "model",
            "id",
            "story",
            "question",
            "ref_answer",
            "response",
            "llama2_score",
            "gpt_score",
            "human_eval",
        ]
    )
    return dfs


if __name__ == "__main__":
    import _settings

    sample_all().to_excel("human_eval.xlsx", index=True)

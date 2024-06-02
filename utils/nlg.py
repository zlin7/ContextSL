import csv
import functools
import os
from collections import defaultdict
from importlib import reload
from typing import List

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import torch
import tqdm


def locate_answer(target, full, prefix):
    # find the sequence (prefix + target) in full
    # return only the last occurence of target
    if not isinstance(target, list):
        target = target.cpu().tolist()
    if not isinstance(full, list):
        full = full.cpu().tolist()
    to_find = prefix + target
    # print(to_find)
    last = None
    for i in range(len(full) - len(to_find), -1, -1):
        if full[i : i + len(to_find)] == to_find:
            last = i + len(prefix)
            break
    assert (
        last is not None
    ), f"Couldn't find the old generation tokens {target} \n\nin the new tokens {full}"
    return last


def _create_prompt_qa(optional_context, question):
    elicit_prompt = f"""Read the following question with optional context and decide if the \
answer correctly answer the question. Focus on the answer, and reply Y or N.


Context: Luxor International Airport is a airport near Luxor in Egypt (EG). It is 353km away from the nearest seaport (Duba). The offical IATA for this airport is LXR.
Question: Luxor international airport is in which country?
Answer: It is in the United States, and its IATA is LXR.
 Decision: N. (The airport is in Egypt, not the United States.)


Context: Harry is a good witcher.
Question: How old is Harry?
Answer: Harry practices witchcraft.
 Decision: N. (The answer does not mention Harry's age.)


Question: What is the capital of Kenya?
Answer: Nairobi is the capital of Kenya.
 Decision: Y.


Question: Who has won the most Premier League titles since 2015?
Answer: Manchester City have win the most Premier League title after 2015.
 Decision: Y. (Grammar errors are ignored.)


{optional_context}Question: {question}
Answer:"""
    return elicit_prompt


class ModelOutputWrapperBatched:
    @torch.no_grad()
    def __init__(
        self, input_ids: List[List[int]], model, tokenizer, output_attentions=True
    ) -> None:
        input_ids_padded = (
            torch.nested.nested_tensor(input_ids)
            .to_padded_tensor(tokenizer.pad_token_id)
            .to(model.device)
        )
        self.model_output = model(
            input_ids_padded,
            output_attentions=output_attentions,
            return_dict=True,
            output_hidden_states=True,
        )
        if output_attentions:
            self.model_output["attentions"] = [
                _ for _ in self.model_output["attentions"]
            ]
        self.model = model
        self.tokenizer = tokenizer
        self.input_ids = input_ids

    def num_heads(self, layer: int = -1):
        return self.model_output["attentions"][layer].shape[1]

    @property
    def num_layers(self):
        return len(self.model_output["attentions"])

    @functools.cached_property
    def tokens(self):
        return [self.tokenizer.convert_ids_to_tokens(_) for _ in self.input_ids]

    def vis_attn(self, idx, layer: int = -1, head: int = 0, token=None):
        _len = len(self.input_ids[idx])
        if token is None:
            token = _len - 1
        attn = (
            self.model_output["attentions"][layer][idx, head, token, :_len]
            .cpu()
            .numpy()
        )
        attn_ser = pd.Series(attn, index=self.tokens[idx])
        return attn_ser

    @functools.cached_property
    def _layer_heads_mapping(self):
        ret = {}
        i = 0
        for layer in range(self.num_layers):
            for head in range(self.num_heads(layer)):
                ret[(layer, head)] = i
                i += 1
        return ret


class _AttnGenHelperBatched(ModelOutputWrapperBatched):
    @torch.no_grad()
    def __init__(
        self,
        prompts: List[List[int]],
        generations: List[List[int]],
        model,
        tokenizer,
        output_attentions=True,
    ) -> None:
        super().__init__(prompts, model, tokenizer, output_attentions=output_attentions)
        self.generations = generations
        # Need to be located by the subclass
        self.generation_st = None

    def _fast_attn_weighted_sum(self, idx, val, layer_heads="all", debug=False):
        assert layer_heads is not None, "layer_heads must be specified (can use 'all')"
        # for idx, input_ids in enumerate(self.input_ids):
        input_ids = self.input_ids[idx]
        _len = len(input_ids)
        _st = self.generation_st[idx]
        _ed = _st + len(self.generations[idx])
        weights = torch.concat(
            [_[idx, :, _len - 1, _st:_ed] for _ in self.model_output["attentions"]]
        ).float()
        if layer_heads == "all":
            layer_heads = [
                (layer, head)
                for layer in range(self.num_layers)
                for head in range(self.num_heads(layer))
            ]
        else:
            weights = weights[[self._layer_heads_mapping[_] for _ in layer_heads]]

        weights = weights / weights.sum(1, keepdim=True)
        weights = pd.DataFrame(
            weights.cpu().numpy(), index=layer_heads, columns=self.tokens[idx][_st:_ed]
        ).T
        # make weights' columns two level multi-index with [layer, head]
        weights.columns = pd.MultiIndex.from_tuples(
            weights.columns, names=["layer", "head"]
        )
        weights["token_nll"] = val
        return weights, self.errs

    def vis_ans_attn(self, idx, *, layer: int = -1, head: int = 0):
        _len = len(self.input_ids[idx])
        attn_ser = self.vis_attn(idx, layer, head, token=_len - 1)
        generation_st = self.generation_st[idx]
        assert abs(attn_ser.sum() - 1) < 1e-3
        attn_ser = attn_ser[generation_st : generation_st + len(self.generations[idx])]
        return attn_ser

    def attn_by_head(self, idx, *, layer: int = -1, normalize=False, head: int = None):
        if layer is None:
            ret = []
            for layer in range(self.num_layers):
                ret.append(
                    self.attn_by_head(idx, layer=layer, normalize=normalize, head=head)
                )
            return pd.concat(ret, axis=1)
        ret = {}
        for h in range(self.num_heads(layer)):
            if head is not None and h != head:
                continue
            attn = self.vis_ans_attn(idx, layer=layer, head=h)
            if normalize:
                attn = attn / attn.sum()
            ret[(layer, h)] = attn
        return pd.DataFrame(ret)

    def summ_attn_by_head(
        self,
        idx,
        groups,
        *,
        layer: int = -1,
        normalize=False,
        head: int = None,
        debug=False,
    ):
        attn = self.attn_by_head(idx, layer=layer, normalize=normalize, head=head)

        summ = {}
        for concept, (st, ed) in groups.items():
            if self.debug or debug:
                print(concept, attn.index[st : ed + 1])
            summ[concept] = attn.iloc[st : ed + 1].sum(axis=0)
        return pd.DataFrame(summ)

    def compute_attn_weighted_sum(self, idx, val, layer_heads="all", debug=False):
        assert len(val) == len(self.generations[idx])
        return self._fast_attn_weighted_sum(
            idx, val, layer_heads=layer_heads, debug=debug
        )


class AttnGenerationBatched(_AttnGenHelperBatched):
    def __init__(
        self,
        prompts,
        generations,
        model,
        tokenizer,
        debug=False,
        has_context=False,
        self_prob_only=False,
    ) -> None:
        self.tokenizer = tokenizer
        assert tokenizer.__class__.__name__ in {
            "GPT2Tokenizer",
            "LlamaTokenizer",
            "GemmaTokenizer",
        }, f"Unknown tokenizer {tokenizer}"
        generations = [_[_.ne(tokenizer.pad_token_id)].tolist() for _ in generations]
        elicit_prompts = [
            self._get_input_ids(_[0], _[1], has_context)
            for _ in zip(prompts, generations)
        ]
        super().__init__(
            elicit_prompts,
            generations,
            model,
            tokenizer,
            output_attentions=not self_prob_only,
        )
        self.errs = []
        if self_prob_only:
            return
        self._check_top_token()

        self.generation_st = [
            locate_answer(_[1], _[0], prefix=tokenizer.encode("\nAnswer:")[2:])
            for _ in zip(self.input_ids, generations)
        ]
        self.debug = debug

    def _split_prompt_new(self, prompt_str, has_context):
        assert "*Question*" in prompt_str
        assert prompt_str.endswith(
            "nswer*:"
        ), f"Prompt should end with 'nswer*:', but is {prompt_str}"
        prompt_str = prompt_str.split("*Question*:")
        if len(prompt_str) == 1 or (not has_context):
            optional_context = ""
        else:
            prompt_str[0] = prompt_str[0].split("*Context*:")[-1].lstrip()
            if prompt_str[0].endswith("\n\n"):
                prompt_str[0] = prompt_str[0][:-1]
            optional_context = (
                "Context: " + "Q:".join(prompt_str[:-1]).strip() + "\n"
            )  # remove \n as well
            optional_context = optional_context.replace("*Answer*:", "A:")
        question = prompt_str[-1].split("*Answer*:")[0].strip()
        return dict(optional_context=optional_context, question=question)

    def _get_input_ids(self, prompt, generation, has_context):
        tokenizer = self.tokenizer

        assert (
            prompt[0] == tokenizer.bos_token_id
        ), f"Prompt should start with <BOS>, but is {prompt[:3]}"
        prompt_str = tokenizer.decode(prompt, skip_special_tokens=True)
        assert "*Question*" in prompt_str
        _res = self._split_prompt_new(prompt_str, has_context)
        elicit_prompt = tokenizer.encode(_create_prompt_qa(**_res), return_tensors=None)

        _suffix_prompt = " Decision:"
        if not tokenizer.decode(generation).endswith("\n"):
            _suffix_prompt = "\n" + _suffix_prompt
        return (
            elicit_prompt
            + generation
            + tokenizer.encode(
                _suffix_prompt, return_tensors=None, add_special_tokens=False
            )
        )

    def _check_top_token(self):
        ret = []
        tokenizer = self.tokenizer
        # Sanity check
        if "GPT2Tokenizer" in str(tokenizer):
            Y_token, N_token = 854, 234
        elif "LlamaTokenizer" in str(tokenizer):
            Y_token = tokenizer.encode("Y", add_special_tokens=False)[0]
            N_token = tokenizer.encode("N", add_special_tokens=False)[0]
        elif tokenizer.__class__.__name__ in {"GemmaTokenizer"}:
            Y_token = tokenizer.encode("▁Y", add_special_tokens=False)[0]  # 890
            N_token = tokenizer.encode("▁N", add_special_tokens=False)[0]  # 646

        for idx in range(self.model_output["logits"].shape[0]):
            _len = len(self.input_ids[idx])
            _top_tokens = (
                self.model_output["logits"][idx, _len - 1].argsort()[-30:].tolist()
            )
            if not all([_ in _top_tokens for _ in {Y_token, N_token}]):
                print(f"Y/N are not in the top 30 tokens: {_top_tokens}")
                self.errs.append(f"Y/N are not in the top 30 tokens for idx {idx}")
            _logsoftmaxed_logits = torch.nn.functional.log_softmax(
                self.model_output["logits"][idx, _len - 1], dim=0
            )
            ret.append(_logsoftmaxed_logits[[Y_token, N_token]].cpu())
        return torch.stack(ret)


class AttnGenerationBatchedNextToken(_AttnGenHelperBatched):
    def __init__(
        self,
        prompts,
        generations,
        model,
        tokenizer,
        debug=False,
        has_context=False,
        self_prob_only=False,
    ) -> None:
        self.tokenizer = tokenizer
        assert tokenizer.__class__.__name__ in {
            "GPT2Tokenizer",
            "LlamaTokenizer",
            "GemmaTokenizer",
        }, f"Unknown tokenizer {tokenizer}"
        generations = [_[_.ne(tokenizer.pad_token_id)] for _ in generations]
        elicit_prompts = [
            self._get_input_ids(_[0], _[1], has_context)
            for _ in zip(prompts, generations)
        ]
        super().__init__(
            elicit_prompts,
            generations,
            model,
            tokenizer,
            output_attentions=not self_prob_only,
        )
        self.errs = []
        self.generation_st = [len(_) for _ in prompts]
        self.debug = debug

    def _split_prompt_new(self, prompt_str, has_context):
        assert "*Question*" in prompt_str
        assert prompt_str.endswith(
            "nswer*:"
        ), f"Prompt should end with 'nswer*:', but is {prompt_str}"
        prompt_str = prompt_str.split("*Question*:")
        if len(prompt_str) == 1 or (not has_context):
            optional_context = ""
        else:
            prompt_str[0] = prompt_str[0].split("*Context*:")[-1].lstrip()
            if prompt_str[0].endswith("\n\n"):
                prompt_str[0] = prompt_str[0][:-1]
            optional_context = (
                "Context: " + "Q:".join(prompt_str[:-1]).strip() + "\n"
            )  # remove \n as well
            optional_context = optional_context.replace("*Answer*:", "A:")
        question = prompt_str[-1].split("*Answer*:")[0].strip()
        return dict(optional_context=optional_context, question=question)

    def _get_input_ids(self, prompt, generation, has_context):
        return torch.concat([prompt, generation])


def _compute_attn_weighted_sum_batched(
    vals,
    prompts,
    generations,
    model,
    tokenizer,
    debug=False,
    has_context=False,
    layer_heads="all",
    max_batch_size=3,
    next_prompt=False,
):
    """Compute the attention-weighted sum of the values in val, given the prompt and generation.
    val is a list of values, one for each token in the generation.
    """
    if has_context:
        max_batch_size = 1
    assert all([len(_[0]) == len(_[1]) for _ in zip(vals, generations)])
    assert len(vals) == len(prompts) == len(generations)
    for i, generation in enumerate(generations):
        if len(generation) == 0:
            generations[i] = torch.tensor(
                [], dtype=prompts[i].dtype, device=prompts[i].device
            )
        else:
            assert generation.dtype == prompts[i].dtype == torch.long
    ret = []
    if next_prompt:
        for st in range(0, len(prompts), max_batch_size):
            obj = AttnGenerationBatchedNextToken(
                prompts[st : st + max_batch_size],
                generations[st : st + max_batch_size],
                model,
                tokenizer,
                debug=False,
                has_context=has_context,
            )
            for idx, val in enumerate(vals[st : st + max_batch_size]):
                ret.append(
                    obj.compute_attn_weighted_sum(
                        idx, val, layer_heads=layer_heads, debug=debug
                    )
                )
        return ret

    for st in range(0, len(prompts), max_batch_size):
        obj = AttnGenerationBatched(
            prompts[st : st + max_batch_size],
            generations[st : st + max_batch_size],
            model,
            tokenizer,
            debug=False,
            has_context=has_context,
        )
        for idx, val in enumerate(vals[st : st + max_batch_size]):
            ret.append(
                obj.compute_attn_weighted_sum(
                    idx, val, layer_heads=layer_heads, debug=debug
                )
            )
    return ret

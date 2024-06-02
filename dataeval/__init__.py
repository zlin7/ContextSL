from .load import (
    _get_dataset_name,
    _get_model_name,
    get_key_from_generated_strings_path_new,
    read_attn_loglikelihoods_all,
    read_attn_loglikelihoods_all_next_token,
    read_cleaned_outputs,
    read_loglikelihoods_and_more,
    read_model_eval_general,
    read_self_eval,
    read_semantic_similarities,
    read_token_relevance,
)

"""
After you set the generation output path in _settings.py, for each `path`, the dependency is as follows:

1. read_cleaned_outputs
2. (all depend on 1)
    read_loglikelihoods_and_more;
    read_self_eval;
    read_semantic_similarities;
    read_token_relevance;
    read_model_eval_general (you can use model='gpt' or 'llama2'.)
3.  (both depend on read_loglikelihoods_and_more)
    read_attn_loglikelihoods_all
    read_attn_loglikelihoods_all_next_token


After all functions in 1/2/3 are computed, we can compute the performance of different baselines.
"""

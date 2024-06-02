import json
import os
import time

import ipdb
import openai
import persist_to_disk as ptd
from openai.error import (
    APIError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
)

TOTAL_TOKEN = 0

with open(os.path.join(os.path.dirname(__file__), "..", "keys.json"), "r") as f:
    openai.api_key = json.load(f)["openai"]["apiKey"]


@ptd.persistf(groupby=["model"], hashsize=10000, lock_granularity="call")
def _openai_query_cached(
    prompt="Hello World", model="ada", attempt_id=0, temperature=0.5
):
    return openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )


def retry_openai_query(
    prompt="Hello World", model="ada", attempt_id=0, max_tries=5, temperature=0.5
):
    for i in range(max_tries):
        try:
            return _openai_query_cached(
                prompt, model, attempt_id, temperature=temperature
            )
        except (RateLimitError, APIError, ServiceUnavailableError) as e:
            print(e)
            time.sleep(1)
            if i == max_tries - 1:
                raise e
        except InvalidRequestError as e:
            err_str = str(e)
            print(err_str)
            assert (
                "We've encountered an issue with repetitive patterns in your prompt"
                in err_str
            )
            return None
        except Exception as err:
            raise err


def _token_to_price(model, tokens):
    return (
        tokens
        // 1000
        * {
            "gpt-3.5-turbo-0125": 0.0015,
            "gpt-3.5-turbo-0613": 0.002,
            "gpt-3.5-turbo-0301": 0.002,
        }[model]
    )


def openai_query(
    prompt, model, attemptd_id, max_tries=50, verbose=False, temperature=1
):
    assert model == "gpt-3.5-turbo-0125"
    global TOTAL_TOKEN
    completion = retry_openai_query(
        prompt, model, attemptd_id, max_tries=max_tries, temperature=temperature
    )
    if completion is None:
        return ""
    txt_ans = completion.choices[0].message.content
    prev_milestone = _token_to_price(model, TOTAL_TOKEN) // 0.1
    TOTAL_TOKEN += completion["usage"]["total_tokens"]

    if (_token_to_price(model, TOTAL_TOKEN) // 0.1) > prev_milestone:
        if verbose:
            print(
                f"Total Cost > $ {(_token_to_price(model, TOTAL_TOKEN) // 0.1) * 0.1:.1f}"
            )
    return txt_ans

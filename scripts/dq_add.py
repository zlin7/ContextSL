import itertools

import simple_disk_queue as sdq

from _settings import DATASETS, GEN_PATHS, MODELS


def add_tasks(func, queue: sdq.DiskQueue, queue_based=False):
    kwargs = {"clean": True}
    for temp, data, model in itertools.product([0.5, 1.0], DATASETS, MODELS):
        path = GEN_PATHS[temp][data][model]
        if not os.path.exists(path):
            continue
        if func(path, checkonly=True, **kwargs):
            continue
        if queue_based:
            func(path, queue=queue, **kwargs)
        else:
            queue.add_task(func, path, **kwargs)


def add_tasks_eval(func, queue: sdq.DiskQueue, **kwargs):
    for temp, data, model in itertools.product([0.5, 1.0], DATASETS, MODELS):
        path = GEN_PATHS[temp][data][model]
        if not os.path.exists(path):
            continue
        kwargs.setdefault("clean", True)

        for ith in [None, 0, 1, 2, 3, 4]:
            if not func(path, ith=ith, checkonly=True, **kwargs):
                queue.add_task(func, path, ith=ith, readonly=False, **kwargs)
    return queue.id


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    import dataeval.load as dload

    queue = sdq.DiskQueue("qAll_1", overwrite=True, verbose=False)
    queue2 = sdq.DiskQueue("qAll_2", overwrite=True, verbose=False)
    add_tasks(dload.read_loglikelihoods_and_more, queue)  # 9 tasks per temp
    add_tasks(dload.read_self_eval, queue)
    add_tasks(dload.read_token_relevance, queue, queue_based=True)  # 72 tasks per temp
    add_tasks(dload.read_semantic_similarities, queue, queue_based=True)

    # queue 2 must run after queue
    add_tasks(dload.read_attn_loglikelihoods_all_next_token, queue2, queue_based=True)
    add_tasks(dload.read_attn_loglikelihoods_all, queue2, queue_based=True)

    # evaluations
    queue_e = sdq.DiskQueue("qMult", overwrite=True, verbose=False)
    queue_e2 = sdq.DiskQueue("qAPI", overwrite=True, verbose=False)
    add_tasks_eval(dload.read_model_eval_general, queue_e, model="llama2")
    add_tasks_eval(dload.read_model_eval_general, queue_e2, model="gpt")

    print(f"""Lengths of queues
{queue.id}: {len(queue)}
{queue2.id}: {len(queue2)}
{queue_e.id}: {len(queue_e)}
{queue_e2.id}: {len(queue_e2)}
    """)

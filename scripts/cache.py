if __name__ == "__main__":
    import argparse
    import itertools
    import os
    from importlib import reload

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-np", "--npartitions", type=int, default=1, help="number of partitions"
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="model")
    parser.add_argument("-a", "--acc", type=str, default=None, help="acc measure")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="dataset")
    parser.add_argument("-t", "--temp", type=float, default=None, help="temperature")
    args = parser.parse_args()
    print(args)

    acc_names = ["llama2", "gpt", "moe"]
    datasets = ["triviaqa_new", "nq_open_new", "coqa_new"]
    models = (
        ["llama2-13b", "mistral-7b", "gemma-7b"] if args.model is None else [args.model]
    )
    temps = [0.5, 1.0] if args.temp is None else [args.temp]
    if args.acc is not None:
        acc_names = [args.acc]
    if args.dataset is not None:
        datasets = [args.dataset]

    import utils

    o = utils.TaskPartitioner(seed=4)
    import pipeline.summ as summ
    from _settings import GEN_PATHS as paths

    for temp, data, model, acc_name in itertools.product(
        temps, datasets, models, acc_names
    ):
        o.add_task(
            summ._cache_all,
            paths[temp][data][model],
            clean=True,
            acc_name=acc_name,
            cache=1,
        )
    o.run_multi_process(args.npartitions)

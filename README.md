Code for "Contextualized Sequence Likelihood: Enhanced Confidence Scores for Natural Language Generation" ([arxiv](https://arxiv.org/abs/2406.01806))

# Replicate Our Experiments

Packages you might need:

[simple-disk-queue](https://pypi.org/project/simple-disk-queue/): Used to store and run tasks.

[persist_to_disk](https://pypi.org/project/persist-to-disk/): Used to cache experiment results (i.e. those `@ptd.persistf` decorators and `ptd.manual_cache` calls).

## Set the Paths
First, set the corresponding paths of "Step 1" in `_settings.py`.

## Generate the responses
Use the `llama2-13b`, `gemma-7b` or `mistralai/Mistral-7B-v0.1` for model, and `coqa_new`, `triviaqa_new` and `nq_open_new` for the dataset  below.
```
python -m pipeline.generate --model llama2-13b --dataset coqa_new
```

Update `GEN_PATHS` in `_settings.py` for next steps.
(You could find the exact generatoins we used in our paper [here](https://drive.google.com/drive/folders/1_ziVhzPDMJicevCQq2mjUgmp4cRASLEA?usp=drive_link) in "output".)

## Caching/Computing Results


First, add all tasks to a queue on disk, by running
```
python -m scripts.dq_add
```

Then, run the actual computation via the following (in sequence). You could specify the device to use via `-d [device_numbers]`
```
python -m scripts.dq_work -q qAll_1 -d 1
python -m scripts.dq_work -q qAll_2 -d 1
python -m scripts.dq_work -q qMult -d 0,1,2 # This runs a 70B model so might require more GPUs
python -m scripts.dq_work -q qAPI # This queue has only GPT API calls, so no GPU is needed
```

### Downloading the Cache

The previous computation could be skipped by downloading our cache from [link](https://drive.google.com/drive/folders/1_ziVhzPDMJicevCQq2mjUgmp4cRASLEA?usp=drive_link) in "persist_to_disk".
Run `python -m test` so that `persist_to_disk` package will automatically create a cache folder that looks like `/path/persist_to_disk/cache/ContextSL-1/test`.
Put all contents in "persist_to_disk cache" under `/path/persist_to_disk/cache/ContextSL-1`.
Once you download the chace, run `python -m scripts.dq_add` to confirm that all queues are empty.


### Optional But Recommended
After all queues finished, you can optionally run the following to cache down some summarization.
```
python -m pipeline.uq
python -m scripts.cache
```

## Run the Notebooks
Now, you can run `notebook/demo.ipynb` (or other notebooks)

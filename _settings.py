import getpass
import os
import sys

# Step 1: Set the following paths
__USERNAME = getpass.getuser()

_BASE_DIR = f"/shared/{__USERNAME}/"

DATA_FOLDER = os.path.join(_BASE_DIR, "NLG")
GENERATION_FOLDER = os.path.join(DATA_FOLDER, "output")
os.makedirs(GENERATION_FOLDER, exist_ok=True)


# Step 2: After running pipeline/generate.py, update GEN_PATHS such that
# GEN_PATHS[temperature][dataset'][model] is the path to the generated output
# for the given temperature, dataset, and model.
GEN_PATHS = {
    "coqa_new": {
        "llama2-13b": f"{GENERATION_FOLDER}/llama2-13b_coqa_new_10/0.pkl",
        "gemma-7b": f"{GENERATION_FOLDER}/gemma-7b_coqa_new_10/0.pkl",
        "mistral-7b": f"{GENERATION_FOLDER}/mistralai_Mistral-7B-v0.1_coqa_new_10/0.pkl",
    },
    "triviaqa_new": {
        "llama2-13b": f"{GENERATION_FOLDER}/llama2-13b_triviaqa_new_10/0.pkl",
        "gemma-7b": f"{GENERATION_FOLDER}/gemma-7b_triviaqa_new_10/0.pkl",
        "mistral-7b": f"{GENERATION_FOLDER}/mistralai_Mistral-7B-v0.1_triviaqa_new_10/0.pkl",
    },
    "nq_open_new": {
        "llama2-13b": f"{GENERATION_FOLDER}/llama2-13b_nq_open_new_10/0.pkl",
        "gemma-7b": f"{GENERATION_FOLDER}/gemma-7b_nq_open_new_10/0.pkl",
        "mistral-7b": f"{GENERATION_FOLDER}/mistralai_Mistral-7B-v0.1_nq_open_new_10/0.pkl",
    },
}

GEN_PATHS = {
    1.0: GEN_PATHS,
    0.5: {
        dataset: {
            model: path.replace("_10/", "_10_0.5/") for model, path in paths.items()
        }
        for dataset, paths in GEN_PATHS.items()
    },
}

# No need to change the following
DATASETS = ["coqa_new", "triviaqa_new", "nq_open_new"]
MODELS = ["llama2-13b", "gemma-7b", "mistral-7b"]

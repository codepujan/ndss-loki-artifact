#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_experiment.py

Preload all datasets and model weights for further experiments.
caching everything under a single HF_CACHE directory.
"""

import os
import torch
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel

# ─── 1. CONFIGURE CACHE ───────────────────────────────────────────────────────

# Change this path to wherever you’d like all HF artifacts to live
HF_CACHE = os.path.expanduser("./HF_CACHE")

# Make sure the cache dir exists
os.makedirs(HF_CACHE, exist_ok=True)

# Tell all HF libraries to use this folder
os.environ["HF_HOME"]           = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("▶️  Loading dataset ‘ppaudel/loki-test-folds’ …")
fold_data = load_dataset("ppaudel/loki-test-folds")
folds = list(fold_data.keys())



BASE_MODEL = "distilbert-base-uncased"
print(f"▶️  Preloading tokenizer & base model ({BASE_MODEL}) …")
tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
_ = DistilBertModel.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)



for fold in folds:
    fold_id = fold[-1]
    # teacher
    repo_teacher = f"ppaudel/loki-model-teacher-fold-{fold_id}"
    print(f"▶️  Downloading teacher weights for fold {fold_id} …")
    tf = hf_hub_download(
        repo_id=repo_teacher,
        filename="pytorch_model.bin",
        repo_type="model",
        cache_dir=HF_CACHE
    )

    # student
    repo_student = f"ppaudel/loki-model-student-fold-{fold_id}"
    print(f"▶️  Downloading student weights for fold {fold_id} …")
    sf = hf_hub_download(
        repo_id=repo_student,
        filename="pytorch_model.bin",
        repo_type="model",
        cache_dir=HF_CACHE
    )


print("🎉  Setup complete. Everything is downloaded, cached, and ready to go!")

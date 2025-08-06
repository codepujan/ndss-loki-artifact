#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_experiment.py

Download datasets and model weights from Zenodo, unzip, and load locally.
"""
import urllib.request
import sys
import os
import torch
import zipfile
from datasets import load_from_disk
from transformers import DistilBertTokenizer, DistilBertModel
# ─── 1. CONFIGURE CACHE ───────────────────────────────────────────────────────

HF_CACHE = os.path.expanduser("./HF_CACHE")
ZENODO_CACHE = os.path.join(HF_CACHE, "zenodo_data")
os.makedirs(ZENODO_CACHE, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── 2. DEFINE ZENODO ARTIFACTS ───────────────────────────────────────────────

ZENODO_ARTIFACTS = {
    "dataset": {
        "url": "https://zenodo.org/records/16755716/files/loki-test-folds.zip", 
        "target_dir": os.path.join(ZENODO_CACHE, "loki-test-folds")
    },
    "teacher_models": {
        f"fold{i}": {
            "url": f"https://zenodo.org/record/16755716/files/teacher-fold-{i}.zip", 
            "target_dir": os.path.join(ZENODO_CACHE, f"loki-model-teacher-fold-{i}")
        } for i in range(1,6)
    },
    "student_models": {
        f"fold{i}": {
            "url": f"https://zenodo.org/record/16755716/files/student-fold-{i}.zip", 
            "target_dir": os.path.join(ZENODO_CACHE, f"loki-model-student-fold-{i}")
        } for i in range(1,6)
    }
}

# ─── 3. DOWNLOAD & UNZIP FUNCTION ─────────────────────────────────────────────

def download_and_unzip(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = dest_dir + ".zip"
    
    if not os.path.exists(zip_path):
        print(f"Downloading {url} …")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print(f"Found existing ZIP: {zip_path}")

    print(f"Unzipping {zip_path} into {dest_dir} …")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

# ─── 4. DOWNLOAD & LOAD DATASET ───────────────────────────────────────────────

dataset_info = ZENODO_ARTIFACTS["dataset"]
download_and_unzip(dataset_info["url"], dataset_info["target_dir"])

print("▶️  Loading dataset from disk …")
fold_data = load_from_disk(dataset_info["target_dir"])
folds = list(fold_data.keys())

# ─── 5. LOAD BASE MODEL & TOKENIZER ───────────────────────────────────────────

BASE_MODEL = "distilbert-base-uncased"
print(f"▶️  Loading tokenizer & base model ({BASE_MODEL}) …")
tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
_ = DistilBertModel.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)

# ─── 6. DOWNLOAD MODELS FOR EACH FOLD ─────────────────────────────────────────
print(ZENODO_ARTIFACTS)
for fold_id in range(1,6):
    fold = "fold"+str(fold_id)

    # Teacher
    teacher_info = ZENODO_ARTIFACTS["teacher_models"][fold]
    download_and_unzip(teacher_info["url"], teacher_info["target_dir"])
    teacher_model_path = os.path.join(teacher_info["target_dir"], "pytorch_model.bin")
    print(f"Teacher model for fold {fold_id} ready at {teacher_model_path}")

    # Student
    student_info = ZENODO_ARTIFACTS["student_models"][fold]
    download_and_unzip(student_info["url"], student_info["target_dir"])
    student_model_path = os.path.join(student_info["target_dir"], "pytorch_model.bin")
    print(f"Student model for fold {fold_id} ready at {student_model_path}")
# --- Also Download additional metadata files -------------
print("Downloading remaining metadata files")
KEYWORD_CATEGORIES_URL = "https://zenodo.org/records/16755716/files/keywords_categories.json" 
urllib.request.urlretrieve(KEYWORD_CATEGORIES_URL, "keywords_categories.json")

NER_OUTPUT_URL = "https://zenodo.org/records/16755716/files/query_ner_output.json"
urllib.request.urlretrieve(NER_OUTPUT_URL,"query_ner_output.json")

RAW_KEYWORDS_URL = "https://zenodo.org/records/16755716/files/raw_data_keywords.csv"
urllib.request.urlretrieve(RAW_KEYWORDS_URL,"raw_data_keywords.csv")
# ─── 8. DONE ──────────────────────────────────────────────────────────────────
print("Setup complete. Everything downloaded and unzipped from Zenodo.")

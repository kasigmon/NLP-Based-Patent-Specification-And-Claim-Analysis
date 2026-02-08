## server_initialize.py
##
## SERVER INITIALIZATION CODE
##
## Pre-downloads a lot of heavy models, intended to be run before users
## ever touch anything.  This avoids download/loading when users provide queries.
##
## Author: Kirk A. Sigmon
## Last Updated: Feb. 7, 2026

## ----------------------------------------------------------
## IMPORTS AND CONFIG
## ----------------------------------------------------------

from sentence_transformers import SentenceTransformer, CrossEncoder
import os

from config import SPACY_CACHE_DIR, HF_CACHE_DIR

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)

from pathlib import Path
import spacy
from spacy.cli import download
from spacy.util import get_package_path

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer, CrossEncoder

## ----------------------------------------------------------
## CACHE SPACY
## ----------------------------------------------------------

for name in ["en_core_web_sm", "en_core_web_trf"]:
    download(name)

    target = SPACY_CACHE_DIR / name

    if not target.exists():
        print(f"Saving {name} to cache at {target}")
        nlp = spacy.load(name)
        nlp.to_disk(target)

## ----------------------------------------------------------
## CACHE HUGGINGFACE CONTENT
## ----------------------------------------------------------

snapshot_download(repo_id="BAAI/bge-m3", cache_dir=str(HF_CACHE_DIR))
snapshot_download(repo_id="BAAI/bge-reranker-large", cache_dir=str(HF_CACHE_DIR))

# Optional: instantiate to validate everything loads from cache
SentenceTransformer("BAAI/bge-m3", cache_folder=str(HF_CACHE_DIR))
CrossEncoder("BAAI/bge-reranker-large")

print("Model caching done. Cached under:", HF_CACHE_DIR)
print("HF_HOME =", os.environ["HF_HOME"])
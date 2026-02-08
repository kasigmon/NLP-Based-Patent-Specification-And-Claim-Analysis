## config.py
##
## APPLICATION CONFIGURATION SCRIPT
##
## Initializes various configuration settings for app
##
## Author: Kirk A. Sigmon
## Last Updated: Feb. 7, 2026

import os
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

USPTO_API_KEY = os.environ.get("USPTO_API_KEY", "")
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", str(ROOT / "temp" / "uspto_pfw_downloads"))

if not USPTO_API_KEY:
    raise RuntimeError("Missing USPTO_API_KEY. Put it in .env at project root.")

POPPLER_BIN = ROOT / "libraries" / "poppler-25.12.0" / "Library" / "bin"
TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

MODELS_DIR = ROOT / "temp" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SPACY_CACHE_DIR = MODELS_DIR / "spacy"
HF_CACHE_DIR = MODELS_DIR / "hf"
SPACY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
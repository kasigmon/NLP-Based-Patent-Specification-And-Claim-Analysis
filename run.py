## run.py
##
## MASTER SCRIPT TO INITIALIZE SYSTEM
##
## Runs Flask system, ultimately providing an HTML-based interface
## for the system.
##
## Author: Kirk A. Sigmon
## Last Updated: Feb. 7, 2026

# Ensure HF caches are saved to folder
# This is a bit belt-and-suspenders, but avoids issues
import os
from config import HF_CACHE_DIR
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))

# Load Flask app function
from webapp.flask_app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
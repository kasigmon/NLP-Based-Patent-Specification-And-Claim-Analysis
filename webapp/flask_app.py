## flask_app.py
##
## FLASK APP FOR SUPPORT QUERIES
##
## Uses Flask to perform support queries, makes public-friendly
##
## Author: Kirk A. Sigmon
## Last Updated: Feb. 7, 2026

## ----------------------------------------------------------
## IMPORTS
## ----------------------------------------------------------

# Flask and File-level Functionalities
import os
from flask import Flask, request, render_template, abort
import time

# Retrieval of Github-safe .env file that contains API key, etc.
from dotenv import load_dotenv
load_dotenv()

# Our other scripts for the Flask script to use
import core.uspto_file_retrieval as uspto
import core.ocr as ocr_mod
import core.support_search as ss

# Get our configurations, as well
from config import USPTO_API_KEY, DOWNLOAD_DIR, POPPLER_BIN, HF_CACHE_DIR

# Set OS environments such that we always load from our caches
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)

## ----------------------------------------------------------
## FLASK APP FUNCTIONS
## ----------------------------------------------------------

# Function to time each step
def timed(stage, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0)

# Create an app function
def create_app():
    app = Flask(__name__, template_folder="html_templates")

    # Get our USPTO API key (get your own!)
    API_KEY = os.environ["USPTO_API_KEY"]

    # Simple in-memory cache so that repeated queries on same patent/app don’t redo OCR/embeddings
    # Keyed by application number (resolved)
    INDEX_CACHE = {}

    # Initialize our index.html when given a generic GET command to root
    @app.get("/")
    def home():
        return render_template("index.html")

    # Define how we handle a POST function to /run
    @app.post("/run")
    def run():
        # Start counting time
        timings = {}
        t0 = time.perf_counter()

        # Grab patent and/or app information
        patent_no = (request.form.get("patent_number_to_search") or "").strip() or None
        app_no_in = (request.form.get("app_number_to_search") or "").strip() or None
        query = (request.form.get("query") or "").strip()

        # Throw complaints if we don't get a query or patent
        if not query:
            abort(400, "Missing query.")
        if not (patent_no or app_no_in):
            abort(400, "Provide patent_number_to_search or app_number_to_search.")

        # Resolve to application number (initializes a stable cache key to handle multiple queries)
        app_no, timings["resolve_app_no"] = timed(
            "resolve_app_no",
            uspto.resolve_application_number,
            patent_number=patent_no,
            application_number=app_no_in,
            API_KEY=API_KEY,
        )

        idx = INDEX_CACHE.get(app_no)

        # If not cached, do the heavy steps (download -> extract/OCR -> init models -> build index)
        if idx is None:
            pdf_path, timings["download_pdf"] = timed(
                "download_pdf",
                uspto.download_specification,
                patent_number=patent_no,
                application_number=app_no_in,
                API_KEY=API_KEY,
            )

            full_text, timings["text_extract_or_ocr"] = timed(
                "text_extract_or_ocr",
                ocr_mod.specification_to_text,
                pdf_path,
                poppler_path=POPPLER_BIN,
            )

            _, timings["init_models"] = timed(
                "init_models",
                ss.init_models,
            )

            (bm25, sent_emb, sentences, meta, para_to_sentence_idxs), timings["build_index"] = timed(
                "build_index",
                ss.build_support_index,
                full_text,
            )

            idx = {
                "bm25": bm25,
                "sent_emb": sent_emb,
                "sentences": sentences,
                "meta": meta,
                "para_to_sentence_idxs": para_to_sentence_idxs,
            }
            INDEX_CACHE[app_no] = idx

        # Perform a support search on the appropriate indices
        hits, timings["search"] = timed(
            "search",
            ss.support_search_using_query,
            query,
            bm25=idx["bm25"],
            sent_emb=idx["sent_emb"],
            sentences=idx["sentences"],
            meta=idx["meta"],
            bm25_k=400,
            dense_k=400,
            fused_top_n=250,
            rerank_top_n=30,
        )

        # Collect the HTML indicating our results
        support_html, timings["render"] = timed(
            "render",
            ss.render_sentence_support_html,
            hits,
            query,
            sentences=idx["sentences"],
            para_to_sentence_idxs=idx["para_to_sentence_idxs"],
            extract_query_terms_fn=ss.extract_query_terms,
            highlight_html_fn=ss.highlight_html,
            underline_html_fn=ss.underline_html,
            nlp=ss._nlp if ss._nlp is not None else (ss.init_models() or ss._nlp),
            stopwords=ss.DEFAULT_STOPWORDS,
            min_score=0.20,
            strong_score=0.40,
            max_results=10,
        )

        # Finalize time counting
        elapsed_s = time.perf_counter() - t0
        timings["total"] = elapsed_s

        # Return the HTML via results.html
        return render_template(
            "results.html",
            query=query,
            app_no=app_no,
            patent_no=patent_no,
            hits=hits,
            support_html=support_html,
            elapsed_s=elapsed_s,
            timings=timings,
        )

    return app

# If we're the main, go ahead and create the app
if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8000, debug=True)

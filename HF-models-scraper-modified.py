#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HF-models-scraper-resume.py
- Scrapes HF model catalog page-by-page
- Saves each page JSON
- Saves per-model detail JSON
- Retries transient errors
- Resumes from checkpoint (next_url + page_index)
- Skips already-downloaded model files
"""

import os
import re
import json
import time
import requests
from typing import Optional, Tuple

# =========================
# CONFIG
# =========================
BASE_DIR = "/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/HuggingFaceStudy"
PAGES_DIR = os.path.join(BASE_DIR, "modelsPerPages")
INFO_DIR = os.path.join(BASE_DIR, "modelsInfo")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint.json")
ERROR_LOG = os.path.join(BASE_DIR, "errors.log")

START_URL = "https://huggingface.co/api/models/"

# polite pacing
REQUEST_SLEEP_SEC = 0.1
MAX_RETRIES = 6
TIMEOUT_SEC = 60

# legacy throttle (optional)
SLEEP_EVERY_N_REQS = 1000
SLEEP_SECONDS = 1800

# =========================
# AUTH
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Run: export HF_TOKEN='hf_xxx'")

HEADERS = {"authorization": f"Bearer {HF_TOKEN}"}

# =========================
# UTILS
# =========================
def ensure_dirs():
    os.makedirs(PAGES_DIR, exist_ok=True)
    os.makedirs(INFO_DIR, exist_ok=True)

def safe_model_filename(model_id: str) -> str:
    # keep your separator style
    return model_id.replace("/", "£sep£") + ".json"

def model_file_path(model_id: str) -> str:
    return os.path.join(INFO_DIR, safe_model_filename(model_id))

def log_error(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line.strip())
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(line)

def save_checkpoint(obj: dict):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)

def load_checkpoint() -> dict:
    if not os.path.exists(CHECKPOINT_FILE):
        return {"next_url": START_URL, "page_index": 0, "num_reqs": 0}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_next_link(link_header: Optional[str]) -> Optional[str]:
    """
    Robustly parse RFC5988 Link header and return rel="next" URL if present.
    """
    if not link_header:
        return None
    # Example chunk: <https://huggingface.co/api/models?...>; rel="next"
    parts = [p.strip() for p in link_header.split(",")]
    for p in parts:
        m = re.match(r'<([^>]+)>\s*;\s*rel="([^"]+)"', p)
        if m and m.group(2) == "next":
            return m.group(1)
    return None

def request_with_retry(url: str, headers: dict, max_retries: int = MAX_RETRIES, timeout: int = TIMEOUT_SEC):
    """
    Retries on transient errors and rate limits.
    """
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)

            # Rate limited
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after) + 2
                else:
                    wait = min(2 ** attempt, 120)
                log_error(f"429 rate limit for {url} -> sleep {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue

            # Transient server errors
            if r.status_code in (500, 502, 503, 504):
                wait = min(2 ** attempt, 60)
                log_error(f"{r.status_code} for {url} -> retry in {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue

            return r

        except requests.RequestException as e:
            wait = min(2 ** attempt, 60)
            log_error(f"Request exception for {url}: {e} -> retry in {wait}s (attempt {attempt})")
            time.sleep(wait)

    return None

def json_from_response_or_none(r: requests.Response):
    ctype = r.headers.get("Content-Type", "")
    if "application/json" not in ctype.lower():
        return None
    try:
        return r.json()
    except ValueError:
        return None

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()
    ckpt = load_checkpoint()

    next_url = ckpt.get("next_url", START_URL)
    page_index = int(ckpt.get("page_index", 0))
    num_reqs = int(ckpt.get("num_reqs", 0))

    print(f"Resume from checkpoint:")
    print(f"  next_url  : {next_url}")
    print(f"  page_index: {page_index}")
    print(f"  num_reqs  : {num_reqs}")

    while next_url:
        # 1) Fetch page
        page_resp = request_with_retry(next_url, HEADERS)
        num_reqs += 1

        if page_resp is None:
            log_error(f"Failed page request after retries: {next_url}")
            break

        if page_resp.status_code != 200:
            log_error(f"Page HTTP {page_resp.status_code}: {next_url}")
            break

        page_data = json_from_response_or_none(page_resp)
        if page_data is None:
            log_error(f"Non-JSON page response: {next_url}")
            break

        # Save page snapshot
        page_path = os.path.join(PAGES_DIR, f"page{page_index}.json")
        with open(page_path, "w", encoding="utf-8") as f:
            json.dump(page_data, f, indent=2)

        print(f"\n[Page {page_index}] models: {len(page_data)} | url: {next_url}")

        # 2) Fetch each model detail (skip if already exists)
        for entry in page_data:
            model_id = entry.get("id")
            if not model_id:
                continue

            out_path = model_file_path(model_id)

            if os.path.exists(out_path):
                # already done in previous run
                continue

            print(model_id)

            model_url = f"https://huggingface.co/api/models/{model_id}"
            r = request_with_retry(model_url, HEADERS)
            num_reqs += 1

            if r is None:
                log_error(f"Detail request failed after retries: {model_id}")
                continue

            if r.status_code != 200:
                log_error(f"Detail HTTP {r.status_code}: {model_id}")
                continue

            obj = json_from_response_or_none(r)
            if obj is None:
                # avoid crash on HTML/empty body
                log_error(f"Detail non-JSON response: {model_id}")
                continue

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)

            # optional throttle
            if num_reqs % SLEEP_EVERY_N_REQS == 0:
                print(f"Hit {num_reqs} requests -> sleep {SLEEP_SECONDS}s")
                time.sleep(SLEEP_SECONDS)

            time.sleep(REQUEST_SLEEP_SEC)

        # 3) Determine next page + checkpoint
        link_header = page_resp.headers.get("Link")
        next_link = parse_next_link(link_header)

        # Save checkpoint AFTER finishing this page
        ckpt = {
            "next_url": next_link,              # where to continue next time
            "page_index": page_index + 1,       # next page index
            "num_reqs": num_reqs,
            "last_saved_page_file": page_path,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_checkpoint(ckpt)

        page_index += 1
        next_url = next_link

    print("\nDone.")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print(f"Pages dir  : {PAGES_DIR}")
    print(f"Info dir   : {INFO_DIR}")
    print(f"Errors log : {ERROR_LOG}")


if __name__ == "__main__":
    main()
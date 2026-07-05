#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HF-models-scraper-resume-fast.py
Optimizations over original:
  - Parallel model detail fetching via ThreadPoolExecutor
  - Persistent requests.Session (TCP connection reuse)
  - Configurable worker count & reduced per-request sleep
  - Checkpoint saved after every page (unchanged, still safe)
  - Skips already-downloaded model files (unchanged)
"""

import os
import re
import json
import time
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
BASE_DIR = "/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/HuggingFaceStudy"
PAGES_DIR = os.path.join(BASE_DIR, "modelsPerPages")
INFO_DIR  = os.path.join(BASE_DIR, "modelsInfo")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint.json")
ERROR_LOG       = os.path.join(BASE_DIR, "errors.log")

START_URL = "https://huggingface.co/api/models/"

# ---- Tunable knobs ----
MAX_WORKERS       = 5      # parallel threads for model-detail fetches
                           # Raise to 8 if no 429s, lower to 3 if they persist
REQUEST_SLEEP_SEC = 0.3    # sleep between individual requests inside a thread
MAX_RETRIES  = 10
TIMEOUT_SEC  = 60

# Legacy big-sleep throttle (optional safety valve)
SLEEP_EVERY_N_REQS = 2000
SLEEP_SECONDS      = 1800

# =========================
# AUTH
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Run: export HF_TOKEN='hf_xxx'")

HEADERS = {"authorization": f"Bearer {HF_TOKEN}"}

# =========================
# SHARED SESSION
# =========================
# One session per thread via thread-local storage
import threading
_thread_local = threading.local()

def get_session() -> requests.Session:
    """Return (or create) a per-thread requests.Session."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0,   # we handle retries manually
        )
        s.mount("https://", adapter)
        s.mount("http://",  adapter)
        _thread_local.session = s
    return _thread_local.session

# =========================
# UTILS
# =========================
def ensure_dirs():
    os.makedirs(PAGES_DIR, exist_ok=True)
    os.makedirs(INFO_DIR,  exist_ok=True)

def safe_model_filename(model_id: str) -> str:
    return model_id.replace("/", "£sep£") + ".json"

def model_file_path(model_id: str) -> str:
    return os.path.join(INFO_DIR, safe_model_filename(model_id))

_log_lock = threading.Lock()

def log_error(msg: str):
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line.strip())
    with _log_lock:
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
    if not link_header:
        return None
    for p in [p.strip() for p in link_header.split(",")]:
        m = re.match(r'<([^>]+)>\s*;\s*rel="([^"]+)"', p)
        if m and m.group(2) == "next":
            return m.group(1)
    return None

def request_with_retry(url: str, session: Optional[requests.Session] = None,
                       max_retries: int = MAX_RETRIES, timeout: int = TIMEOUT_SEC):
    """Retries on transient errors and rate limits using a shared session."""
    if session is None:
        session = get_session()

    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, timeout=timeout)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = int(retry_after) + 2 if (retry_after and retry_after.isdigit()) \
                       else min(2 ** attempt, 120)
                log_error(f"429 rate limit for {url} -> sleep {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue

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
# WORKER: fetch one model
# =========================
def fetch_model(model_id: str) -> int:
    """
    Fetch and save a single model's detail JSON.
    Returns 1 if a network request was made, 0 if skipped (already exists).
    """
    out_path = model_file_path(model_id)
    if os.path.exists(out_path):
        return 0   # already downloaded

    session   = get_session()
    model_url = f"https://huggingface.co/api/models/{model_id}"
    r         = request_with_retry(model_url, session=session)

    if r is None:
        log_error(f"Detail request failed after retries: {model_id}")
        return 1

    if r.status_code != 200:
        log_error(f"Detail HTTP {r.status_code}: {model_id}")
        return 1

    obj = json_from_response_or_none(r)
    if obj is None:
        log_error(f"Detail non-JSON response: {model_id}")
        return 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    if REQUEST_SLEEP_SEC > 0:
        time.sleep(REQUEST_SLEEP_SEC)

    return 1   # made one request

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()
    ckpt = load_checkpoint()

    next_url   = ckpt.get("next_url",   START_URL)
    page_index = int(ckpt.get("page_index", 0))
    num_reqs   = int(ckpt.get("num_reqs",   0))

    print(f"Resume from checkpoint:")
    print(f"  next_url  : {next_url}")
    print(f"  page_index: {page_index}")
    print(f"  num_reqs  : {num_reqs}")
    print(f"  workers   : {MAX_WORKERS}")

    # Use a single session for page-level requests (main thread)
    page_session = requests.Session()
    page_session.headers.update(HEADERS)

    while next_url:
        # ---- 1. Fetch page list ----
        page_resp = request_with_retry(next_url, session=page_session)
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

        page_path = os.path.join(PAGES_DIR, f"page{page_index}.json")
        with open(page_path, "w", encoding="utf-8") as f:
            json.dump(page_data, f, indent=2)

        model_ids = [e.get("id") for e in page_data if e.get("id")]
        print(f"\n[Page {page_index}] models: {len(model_ids)} | url: {next_url}")

        # ---- 2. Parallel model detail fetches ----
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(fetch_model, mid): mid for mid in model_ids}
            for fut in as_completed(futures):
                mid = futures[fut]
                try:
                    reqs_made = fut.result()
                    num_reqs += reqs_made
                    print(f"  ✓ {mid}")
                except Exception as exc:
                    log_error(f"Unexpected error for {mid}: {exc}")

                # Legacy big-sleep safety valve
                if num_reqs % SLEEP_EVERY_N_REQS == 0:
                    print(f"Hit {num_reqs} requests -> sleep {SLEEP_SECONDS}s")
                    time.sleep(SLEEP_SECONDS)

        # ---- 3. Checkpoint + advance ----
        link_header = page_resp.headers.get("Link")
        next_link   = parse_next_link(link_header)

        save_checkpoint({
            "next_url":           next_link,
            "page_index":         page_index + 1,
            "num_reqs":           num_reqs,
            "last_saved_page_file": page_path,
            "updated_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        page_index += 1
        next_url   = next_link

    print("\nDone.")
    print(f"Checkpoint : {CHECKPOINT_FILE}")
    print(f"Pages dir  : {PAGES_DIR}")
    print(f"Info dir   : {INFO_DIR}")
    print(f"Errors log : {ERROR_LOG}")


if __name__ == "__main__":
    main()
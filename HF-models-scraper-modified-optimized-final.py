#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HF-models-scraper-resume-fast.py
Optimizations:
  - Multi-token round-robin (doubles/triples quota with 2-3 tokens)
  - Per-token 429 cooldown: backs off only the exhausted token, not all
  - Parallel model detail fetching via ThreadPoolExecutor
  - Persistent requests.Session per thread
  - Checkpoint saved after every page
  - Skips already-downloaded model files

Setup:
  export HF_TOKEN_1='hf_yourtoken'
  export HF_TOKEN_2='hf_labmatestoken'
  export HF_TOKEN_3='hf_thirdtoken'
  export HF_TOKEN_4='hf_fourthtoken'
  export HF_TOKEN_5='hf_fifthtoken'
"""

import os
import re
import json
import time
import threading
import itertools
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
BASE_DIR        = "/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/HuggingFaceStudy"
PAGES_DIR       = os.path.join(BASE_DIR, "modelsPerPages")
INFO_DIR        = os.path.join(BASE_DIR, "modelsInfo")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint.json")
ERROR_LOG       = os.path.join(BASE_DIR, "errors.log")

START_URL = "https://huggingface.co/api/models/"

# ---- Tunable knobs (adjusted for 5 tokens) ----
MAX_WORKERS        = 10     # 2 per token for 5 tokens
REQUEST_SLEEP_SEC  = 0.15   # faster with 5 tokens spreading the load
MAX_RETRIES        = 15     # enough to survive a full 300s rate-limit window
TIMEOUT_SEC        = 60
PAGE_SLEEP_SEC     = 0.5    # minimal pause; 5 tokens absorb the load

# Legacy big-sleep safety valve (relaxed for 5 tokens)
SLEEP_EVERY_N_REQS = 8000   # 5 tokens = 5x quota headroom
SLEEP_SECONDS      = 300    # 5 min cooldown

# =========================
# MULTI-TOKEN SETUP
# =========================
def load_tokens() -> list:
    tokens = []
    for i in range(1, 20):
        t = os.environ.get(f"HF_TOKEN_{i}", "").strip()
        if t:
            tokens.append(t)
    # fallback to plain HF_TOKEN
    if not tokens:
        t = os.environ.get("HF_TOKEN", "").strip()
        if t:
            tokens.append(t)
    return tokens

TOKENS = load_tokens()
if not TOKENS:
    raise RuntimeError(
        "No HF tokens found. Set HF_TOKEN_1, HF_TOKEN_2, ... or HF_TOKEN.\n"
        "  export HF_TOKEN_1='hf_yourtoken'\n"
        "  export HF_TOKEN_2='hf_labmatestoken'"
    )

print(f"Loaded {len(TOKENS)} token(s).")

# Per-token cooldown tracking
_token_lock     = threading.Lock()
_token_cooldown = {t: 0.0 for t in TOKENS}

def get_next_token() -> str:
    """
    Return the next available token (not in cooldown).
    If all tokens are cooling down, sleep until the earliest one recovers.
    """
    while True:
        with _token_lock:
            now = time.time()
            available = [t for t, until in _token_cooldown.items() if now >= until]
            if available:
                return min(available, key=lambda t: _token_cooldown[t])
            soonest = min(_token_cooldown.values())
            wait = max(0.0, soonest - now) + 1.0
        time.sleep(wait)

def mark_token_cooldown(token: str, seconds: float):
    with _token_lock:
        _token_cooldown[token] = time.time() + seconds

# =========================
# THREAD-LOCAL SESSION
# =========================
_thread_local = threading.local()

def get_session(token: str) -> requests.Session:
    if not hasattr(_thread_local, "session") or getattr(_thread_local, "token", None) != token:
        s = requests.Session()
        s.headers.update({"authorization": f"Bearer {token}"})
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1, pool_maxsize=1, max_retries=0
        )
        s.mount("https://", adapter)
        s.mount("http://",  adapter)
        _thread_local.session = s
        _thread_local.token   = token
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

def request_with_retry(url: str, max_retries: int = MAX_RETRIES, timeout: int = TIMEOUT_SEC):
    """Fetch URL with retry + per-token cooldown. Rotates to a fresh token on 429."""
    for attempt in range(1, max_retries + 1):
        token   = get_next_token()
        session = get_session(token)
        try:
            r = session.get(url, timeout=timeout)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = int(retry_after) + 2 if (retry_after and retry_after.isdigit()) \
                       else min(2 ** attempt, 300)
                log_error(f"429 token=...{token[-6:]} for {url} -> cooldown {wait}s (attempt {attempt})")
                mark_token_cooldown(token, wait)
                continue  # get_next_token() will wait if all tokens exhausted

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
    out_path = model_file_path(model_id)
    if os.path.exists(out_path):
        return 0

    model_url = f"https://huggingface.co/api/models/{model_id}"
    r = request_with_retry(model_url)

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

    return 1

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
    print(f"  tokens    : {len(TOKENS)}")
    print(f"  workers   : {MAX_WORKERS}")

    while next_url:
        # ---- 1. Fetch page list ----
        page_resp = request_with_retry(next_url)
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

        # Breathe between pages
        time.sleep(PAGE_SLEEP_SEC)

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

                if num_reqs % SLEEP_EVERY_N_REQS == 0 and num_reqs > 0:
                    print(f"Hit {num_reqs} requests -> sleep {SLEEP_SECONDS}s")
                    time.sleep(SLEEP_SECONDS)

        # ---- 3. Checkpoint + advance ----
        link_header = page_resp.headers.get("Link")
        next_link   = parse_next_link(link_header)

        save_checkpoint({
            "next_url":             next_link,
            "page_index":           page_index + 1,
            "num_reqs":             num_reqs,
            "last_saved_page_file": page_path,
            "updated_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
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
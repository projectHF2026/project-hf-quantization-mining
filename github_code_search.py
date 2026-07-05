#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
github_code_search.py
=====================
Search GitHub Code Search API for references to quantized HF models.

Features:
  - Multi-token round-robin (4 GitHub tokens)
  - Per-token 429/403 cooldown with automatic rotation
  - Parallel workers via ThreadPoolExecutor
  - Checkpoint/resume after every batch
  - Flags models exceeding 1,000 results (API cap)
  - Stores model → repo + file path mapping
  - All languages searched (no language filter)

Setup:
  export GH_TOKEN_1='ghp_...'
  export GH_TOKEN_2='ghp_...'
  export GH_TOKEN_3='ghp_...'
  export GH_TOKEN_4='ghp_...'

Usage:
  python3 github_code_search.py \
    --models-file /path/to/quantized_models_high_confidence_ids.txt \
    --output-dir /path/to/output \
    --workers 4
"""

import os
import re
import json
import time
import threading
import argparse
import sys
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ============================================================================
# CONFIG
# ============================================================================
MAX_RETRIES       = 10
TIMEOUT_SEC       = 30
# GitHub Code Search has a 10 requests/min limit per token
# So we pace at ~6.5 seconds per request per token to stay under
REQUEST_SLEEP_SEC = 6.5
CHECKPOINT_EVERY  = 100    # save checkpoint every N models processed
RESULTS_PER_PAGE  = 100    # max for GitHub Code Search API

# ============================================================================
# MULTI-TOKEN SETUP
# ============================================================================
def load_github_tokens() -> list:
    tokens = []
    for i in range(1, 20):
        t = os.environ.get(f"GH_TOKEN_{i}", "").strip()
        if t:
            tokens.append(t)
    # Fallback to single GITHUB_TOKEN
    if not tokens:
        t = os.environ.get("GITHUB_TOKEN", "").strip()
        if t:
            tokens.append(t)
    return tokens

TOKENS = load_github_tokens()
if not TOKENS:
    raise RuntimeError(
        "No GitHub tokens found. Set GH_TOKEN_1, GH_TOKEN_2, ... or GITHUB_TOKEN.\n"
        "  export GH_TOKEN_1='ghp_yourtoken'\n"
        "  export GH_TOKEN_2='ghp_labmatestoken'"
    )

print(f"Loaded {len(TOKENS)} GitHub token(s).")

# Per-token cooldown tracking
_token_lock     = threading.Lock()
_token_cooldown = {t: 0.0 for t in TOKENS}
_token_usage    = {t: 0 for t in TOKENS}  # track usage per token

def get_next_token() -> str:
    """Return next available token not in cooldown."""
    while True:
        with _token_lock:
            now = time.time()
            available = [t for t, until in _token_cooldown.items() if now >= until]
            if available:
                # Pick the least-used available token
                chosen = min(available, key=lambda t: _token_usage[t])
                _token_usage[chosen] += 1
                return chosen
            soonest = min(_token_cooldown.values())
            wait = max(0.0, soonest - now) + 1.0
        time.sleep(wait)

def mark_token_cooldown(token: str, seconds: float):
    with _token_lock:
        _token_cooldown[token] = time.time() + seconds

# ============================================================================
# THREAD-LOCAL SESSION
# ============================================================================
_thread_local = threading.local()

def get_session(token: str) -> requests.Session:
    if not hasattr(_thread_local, "session") or getattr(_thread_local, "_token", None) != token:
        s = requests.Session()
        s.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        _thread_local.session = s
        _thread_local._token = token
    return _thread_local.session


# How often to check /rate_limit per token (every N requests)
_PROACTIVE_CHECK_EVERY = 8
_token_check_counter = {t: 0 for t in TOKENS}

def check_rate_limit_proactive(token: str):
    """
    Query /rate_limit and sleep if code_search bucket is exhausted.
    Only checks every _PROACTIVE_CHECK_EVERY requests per token to avoid
    doubling total API calls. Relies on reactive 403 handling in between.
    """
    with _token_lock:
        _token_check_counter[token] += 1
        if _token_check_counter[token] % _PROACTIVE_CHECK_EVERY != 0:
            return True  # skip check this time

    session = get_session(token)
    try:
        r = session.get("https://api.github.com/rate_limit", timeout=10)
        if r.status_code == 200:
            data = r.json()
            code_search = data.get("resources", {}).get("code_search", {})
            if not code_search:
                code_search = data.get("resources", {}).get("search", {})
            remaining = code_search.get("remaining", 10)
            reset_at = code_search.get("reset", 0)

            if remaining <= 1:
                wait = max(reset_at - int(time.time()), 0) + 2
                if wait > 0:
                    log_msg(f"Proactive rate limit: token=...{token[-6:]} "
                            f"remaining={remaining} -> sleep {wait}s")
                    mark_token_cooldown(token, wait)
                    return False
    except Exception:
        pass
    return True

# ============================================================================
# UTILS
# ============================================================================
_log_lock = threading.Lock()

def log_msg(msg: str, error: bool = False):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line)
        if error:
            with open(os.path.join(OUTPUT_DIR, "errors.log"), "a") as f:
                f.write(line + "\n")

# Global output dir (set in main)
OUTPUT_DIR = "."

def save_checkpoint(data: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

# ============================================================================
# GITHUB CODE SEARCH
# ============================================================================

def search_model_in_github(model_id: str) -> dict:
    """
    Search GitHub Code Search API for a model ID.
    Returns dict with repos, file paths, and whether results were capped.
    """
    # Use exact string match for model ID
    query = f'"{model_id}"'
    url = "https://api.github.com/search/code"

    all_items = []
    total_count = 0
    capped = False
    page = 1

    while True:
        params = {
            "q": query,
            "per_page": RESULTS_PER_PAGE,
            "page": page,
        }

        response = _request_with_retry(url, params)

        if response is None:
            log_msg(f"FAILED after retries: {model_id}", error=True)
            break

        if response.status_code == 422:
            # Validation error (e.g., query too long)
            log_msg(f"422 Validation error for: {model_id}", error=True)
            break

        if response.status_code != 200:
            log_msg(f"HTTP {response.status_code} for: {model_id}", error=True)
            break

        try:
            data = response.json()
        except ValueError:
            log_msg(f"Non-JSON response for: {model_id}", error=True)
            break

        total_count = data.get("total_count", 0)

        if total_count == 0:
            break

        items = data.get("items", [])
        for item in items:
            repo_full_name = item.get("repository", {}).get("full_name", "")
            file_path = item.get("path", "")
            html_url = item.get("html_url", "")
            if repo_full_name:
                all_items.append({
                    "repo": repo_full_name,
                    "file_path": file_path,
                    "html_url": html_url,
                })

        # Check if we've hit the 1,000 result cap
        if total_count > 1000:
            capped = True

        # Pagination: GitHub Code Search returns max 1000 results (10 pages × 100)
        if len(items) < RESULTS_PER_PAGE or page >= 10:
            break

        page += 1
        time.sleep(REQUEST_SLEEP_SEC)

    # Deduplicate by repo + file_path
    seen = set()
    unique_items = []
    for item in all_items:
        key = (item["repo"], item["file_path"])
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    return {
        "model_id": model_id,
        "total_count": total_count,
        "capped": capped,
        "matches": unique_items,
        "unique_repos": list({item["repo"] for item in unique_items}),
        "num_repos": len({item["repo"] for item in unique_items}),
        "num_files": len(unique_items),
    }


def _request_with_retry(url: str, params: dict) -> Optional[requests.Response]:
    """Make a request with retry + token rotation + proactive rate limit check."""
    for attempt in range(1, MAX_RETRIES + 1):
        token = get_next_token()

        # Proactive rate limit check — if exhausted, cooldown and get new token
        if not check_rate_limit_proactive(token):
            continue

        session = get_session(token)

        try:
            r = session.get(url, params=params, timeout=TIMEOUT_SEC)

            # Rate limit (reactive fallback)
            if r.status_code == 403 or r.status_code == 429:
                remaining = r.headers.get("X-RateLimit-Remaining", "?")
                reset_at = r.headers.get("X-RateLimit-Reset", "")

                if reset_at and reset_at.isdigit():
                    wait = max(int(reset_at) - int(time.time()), 0) + 2
                else:
                    retry_after = r.headers.get("Retry-After", "")
                    wait = int(retry_after) + 2 if retry_after.isdigit() else min(2 ** attempt, 120)

                log_msg(f"Rate limit (reactive) token=...{token[-6:]} remaining={remaining} "
                        f"-> cooldown {wait}s (attempt {attempt})")
                mark_token_cooldown(token, wait)
                continue

            # Server errors
            if r.status_code in (500, 502, 503, 504):
                wait = min(2 ** attempt, 60)
                log_msg(f"{r.status_code} -> retry in {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue

            return r

        except requests.RequestException as e:
            wait = min(2 ** attempt, 60)
            log_msg(f"Request error: {e} -> retry in {wait}s (attempt {attempt})", error=True)
            time.sleep(wait)

    return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_model(model_id: str) -> dict:
    """Search for one model and return results."""
    return search_model_in_github(model_id)


def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Search GitHub for quantized HF model references"
    )
    parser.add_argument(
        "--models-file",
        required=True,
        help="File with model IDs (one per line)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, recommended: 1 per token)"
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model IDs
    with open(args.models_file) as f:
        all_models = [line.strip() for line in f if line.strip()]

    print(f"GitHub Code Search for Quantized Models")
    print(f"{'='*60}")
    print(f"Models file: {args.models_file}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Total models: {len(all_models):,}")
    print(f"Tokens: {len(TOKENS)}")
    print(f"Workers: {args.workers}")

    # Load processed models from append-only log (lightweight resume)
    processed_log_path = os.path.join(OUTPUT_DIR, "processed_models.txt")
    processed_set = set()
    if os.path.exists(processed_log_path):
        with open(processed_log_path) as f:
            processed_set = {line.strip() for line in f if line.strip()}
    print(f"Already processed: {len(processed_set):,}")

    # Load counters from stats checkpoint
    stats_ckpt_path = os.path.join(OUTPUT_DIR, "search_counters.json")
    stats_ckpt = load_checkpoint(stats_ckpt_path)
    total_with_results = stats_ckpt.get("total_with_results", 0)
    total_capped = stats_ckpt.get("total_capped", 0)
    total_repos_found = stats_ckpt.get("total_repos_found", 0)

    # Filter out already processed
    remaining = [m for m in all_models if m not in processed_set]
    print(f"Remaining: {len(remaining):,}")
    print()

    # Output file paths
    results_path = os.path.join(OUTPUT_DIR, "search_results.jsonl")
    capped_path = os.path.join(OUTPUT_DIR, "capped_models.txt")

    # Counters
    total_processed = len(processed_set)

    # Thread-safe lock for file writes
    _write_lock = threading.Lock()

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit in chunks to allow periodic stats saves
        chunk_size = CHECKPOINT_EVERY
        for chunk_start in range(0, len(remaining), chunk_size):
            chunk = remaining[chunk_start:chunk_start + chunk_size]

            futures = {executor.submit(process_model, mid): mid for mid in chunk}

            for future in as_completed(futures):
                mid = futures[future]
                try:
                    result = future.result()

                    with _write_lock:
                        # Append result to JSONL
                        with open(results_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

                        # Append to processed log (lightweight, append-only)
                        with open(processed_log_path, "a") as f:
                            f.write(mid + "\n")

                        # Track capped models
                        if result["capped"]:
                            with open(capped_path, "a") as f:
                                f.write(f"{mid}\t{result['total_count']}\n")
                            total_capped += 1

                        if result["num_repos"] > 0:
                            total_with_results += 1
                            total_repos_found += result["num_repos"]

                    processed_set.add(mid)
                    total_processed += 1

                    # Progress
                    if result["num_repos"] > 0:
                        log_msg(f"  ✓ {mid} -> {result['num_repos']} repos, "
                                f"{result['num_files']} files"
                                f"{' [CAPPED]' if result['capped'] else ''}")
                    else:
                        # Print progress periodically for zero-result models
                        if total_processed % 50 == 0:
                            elapsed = time.time() - t0
                            initial_processed = len(all_models) - len(remaining)
                            processed_this_run = total_processed - initial_processed
                            rate = processed_this_run / elapsed if elapsed > 0 else 0
                            remaining_count = len(all_models) - total_processed
                            eta_h = remaining_count / rate / 3600 if rate > 0 else 0
                            log_msg(f"  Progress: {total_processed:,}/{len(all_models):,} | "
                                    f"With results: {total_with_results:,} | "
                                    f"Capped: {total_capped} | "
                                    f"Rate: {rate:.1f}/s | "
                                    f"ETA: {eta_h:.1f}h")

                except Exception as e:
                    log_msg(f"  ✗ Error for {mid}: {e}", error=True)
                    with _write_lock:
                        with open(processed_log_path, "a") as f:
                            f.write(mid + "\n")
                    processed_set.add(mid)
                    total_processed += 1

            # Save lightweight counters after each chunk
            save_checkpoint({
                "total_with_results": total_with_results,
                "total_capped": total_capped,
                "total_repos_found": total_repos_found,
                "total_processed": total_processed,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, stats_ckpt_path)

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total models searched: {total_processed:,}")
    print(f"Models with GitHub results: {total_with_results:,}")
    print(f"Models capped (>1000 results): {total_capped}")
    print(f"Total unique repos found: {total_repos_found:,}")
    print(f"Time: {elapsed/3600:.1f}h")
    print(f"\nOutput files:")
    print(f"  Results:    {results_path}")
    print(f"  Capped:     {capped_path}")
    print(f"  Processed:  {processed_log_path}")
    print(f"  Counters:   {stats_ckpt_path}")

    # Generate summary files
    generate_summary(results_path, OUTPUT_DIR)


def generate_summary(results_path: str, output_dir: str):
    """Generate summary files from the JSONL results."""
    if not os.path.exists(results_path):
        return

    model_to_repos = defaultdict(set)
    repo_to_models = defaultdict(set)
    all_matches = []
    capped_models = []

    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            mid = r["model_id"]
            if r["capped"]:
                capped_models.append({"model_id": mid, "total_count": r["total_count"]})
            for match in r["matches"]:
                repo = match["repo"]
                model_to_repos[mid].add(repo)
                repo_to_models[repo].add(mid)
                all_matches.append({
                    "model_id": mid,
                    "repo": repo,
                    "file_path": match["file_path"],
                })

    # Model → repos mapping
    m2r_path = os.path.join(output_dir, "model_to_repos.json")
    with open(m2r_path, "w") as f:
        json.dump({k: sorted(v) for k, v in model_to_repos.items()}, f, indent=2)

    # Repo → models mapping
    r2m_path = os.path.join(output_dir, "repo_to_models.json")
    with open(r2m_path, "w") as f:
        json.dump({k: sorted(v) for k, v in repo_to_models.items()}, f, indent=2)

    # All matches CSV
    import csv
    csv_path = os.path.join(output_dir, "all_matches.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "repo", "file_path"])
        for m in all_matches:
            w.writerow([m["model_id"], m["repo"], m["file_path"]])

    # Unique repos list
    repos_path = os.path.join(output_dir, "unique_repos.txt")
    with open(repos_path, "w") as f:
        for repo in sorted(repo_to_models.keys()):
            f.write(repo + "\n")

    # Stats
    stats = {
        "total_models_with_results": len(model_to_repos),
        "total_unique_repos": len(repo_to_models),
        "total_matches": len(all_matches),
        "capped_models_count": len(capped_models),
        "capped_models": capped_models,
        "top_models_by_repos": sorted(
            [(k, len(v)) for k, v in model_to_repos.items()],
            key=lambda x: -x[1]
        )[:50],
        "top_repos_by_models": sorted(
            [(k, len(v)) for k, v in repo_to_models.items()],
            key=lambda x: -x[1]
        )[:50],
    }
    stats_path = os.path.join(output_dir, "search_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSummary generated:")
    print(f"  Models with results: {len(model_to_repos):,}")
    print(f"  Unique repos: {len(repo_to_models):,}")
    print(f"  Total matches: {len(all_matches):,}")
    print(f"  Capped models: {len(capped_models)}")
    print(f"  Files: {m2r_path}")
    print(f"         {r2m_path}")
    print(f"         {csv_path}")
    print(f"         {repos_path}")
    print(f"         {stats_path}")


if __name__ == "__main__":
    main()
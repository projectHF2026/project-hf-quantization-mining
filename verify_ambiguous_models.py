#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_ambiguous_models.py
==========================
For models that are legitimately quantized but don't have quantization
in their name (e.g., "google/gemma-2b" which has GGUF files on HF),
re-search GitHub with quantization context to verify if repos are
actually using the quantized version.

For each ambiguous model, we search:
  "model_id" "gguf"
  "model_id" "gptq"
  "model_id" "awq"
  "model_id" "quantiz"
  "model_id" "load_in_4bit"
  "model_id" "load_in_8bit"
  "model_id" "bitsandbytes"
  "model_id" "4bit"
  "model_id" "8bit"

If a repo appears in contextualized results, it's confirmed as
quantized usage. Repos only in the original (plain) search are
unverified.

Setup:
  export GH_TOKEN_1='ghp_...'
  ...

Usage:
  python3 verify_ambiguous_models.py \
    --ambiguous-models /tmp/no_quant_in_name.txt \
    --original-results /path/to/search_results.jsonl \
    --output-dir /path/to/output \
    --workers 9
"""

import os
import json
import time
import threading
import argparse
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ============================================================================
# CONFIG
# ============================================================================
MAX_RETRIES       = 8
TIMEOUT_SEC       = 30
REQUEST_SLEEP_SEC = 6.5  # GitHub Code Search: 10 req/min per token
RESULTS_PER_PAGE  = 100

# Context keywords to append to model ID searches
# Ordered strongest → weakest for early termination efficiency
QUANT_CONTEXT_KEYWORDS = [
    # Strongest: explicit quantized loading APIs
    "load_in_4bit",
    "load_in_8bit",
    "BitsAndBytesConfig",
    "bitsandbytes",
    "from_quantized",
    "AutoGPTQForCausalLM",
    "AutoAWQForCausalLM",
    # Strong: format-specific signals
    "gguf",
    ".gguf",
    "ggml",
    "llama.cpp",
    "llama_cpp",
    "ctransformers",
    # Medium: method names
    "gptq",
    "awq",
    "quantization_config",
    "quantiz",           # matches quantize, quantized, quantization
    # Weaker: bit-width indicators (noisier)
    "int4",
    "int8",
    "nf4",
    "fp8",
    "4bit",
    "8bit",
]

# ============================================================================
# TOKEN SETUP
# ============================================================================
def load_github_tokens() -> list:
    tokens = []
    for i in range(1, 20):
        t = os.environ.get(f"GH_TOKEN_{i}", "").strip()
        if t:
            tokens.append(t)
    if not tokens:
        t = os.environ.get("GITHUB_TOKEN", "").strip()
        if t:
            tokens.append(t)
    return tokens

TOKENS = load_github_tokens()
if not TOKENS:
    raise RuntimeError("No GitHub tokens found.")

print(f"Loaded {len(TOKENS)} GitHub token(s).")

_token_lock = threading.Lock()
_token_cooldown = {t: 0.0 for t in TOKENS}
_token_usage = {t: 0 for t in TOKENS}

# Thread-local sessions for connection pooling (thread-safe)
_thread_local = threading.local()

def _get_session(token: str) -> requests.Session:
    """Get or create a thread-local session for the given token."""
    if not hasattr(_thread_local, "sessions"):
        _thread_local.sessions = {}
    if token not in _thread_local.sessions:
        s = requests.Session()
        s.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        _thread_local.sessions[token] = s
    return _thread_local.sessions[token]

_PROACTIVE_CHECK_EVERY = 40
_token_check_counter = {t: 0 for t in TOKENS}

# Per-token pacing: enforce minimum 6.5s between requests on the same token
_MIN_TOKEN_INTERVAL = 6.5
_token_last_request = {t: 0.0 for t in TOKENS}

def get_next_token() -> str:
    while True:
        with _token_lock:
            now = time.time()
            # Available = not in cooldown AND enough time since last request
            available = []
            for t, until in _token_cooldown.items():
                if now >= until and (now - _token_last_request[t]) >= _MIN_TOKEN_INTERVAL:
                    available.append(t)

            if available:
                chosen = min(available, key=lambda t: _token_usage[t])
                _token_usage[chosen] += 1
                _token_last_request[chosen] = now
                return chosen

            # No token available — figure out how long to wait
            # Wait for either cooldown to expire or pacing interval to pass
            soonest_cooldown = min(_token_cooldown.values())
            soonest_pacing = min(_token_last_request[t] + _MIN_TOKEN_INTERVAL for t in TOKENS)
            soonest = min(soonest_cooldown, soonest_pacing)
            wait = max(0.0, soonest - now) + 0.1
        time.sleep(wait)

def mark_token_cooldown(token: str, seconds: float):
    with _token_lock:
        _token_cooldown[token] = time.time() + seconds

def check_rate_limit_proactive(token: str) -> bool:
    with _token_lock:
        _token_check_counter[token] += 1
        if _token_check_counter[token] % _PROACTIVE_CHECK_EVERY != 0:
            return True
    try:
        r = _get_session(token).get("https://api.github.com/rate_limit", timeout=10)
        if r.status_code == 200:
            data = r.json()
            cs = data.get("resources", {}).get("code_search", {})
            if not cs:
                cs = data.get("resources", {}).get("search", {})
            remaining = cs.get("remaining", 10)
            reset_at = cs.get("reset", 0)
            if remaining <= 1:
                wait = max(reset_at - int(time.time()), 0) + 2
                if wait > 0:
                    mark_token_cooldown(token, wait)
                    return False
    except Exception:
        pass
    return True

# ============================================================================
# SEARCH
# ============================================================================
_log_lock = threading.Lock()

def log_msg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _log_lock:
        print(f"[{ts}] {msg}")


def search_code(query: str, max_pages: int = 2) -> list:
    """Search GitHub Code Search and return list of (repo, file_path) tuples."""
    url = "https://api.github.com/search/code"
    results = []
    page = 1

    while True:
        token = get_next_token()
        if not check_rate_limit_proactive(token):
            continue

        session = _get_session(token)
        params = {"q": query, "per_page": RESULTS_PER_PAGE, "page": page}

        r = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = session.get(url, params=params, timeout=TIMEOUT_SEC)

                if r.status_code == 422:
                    # Query too long or invalid — return what we have
                    log_msg(f"  422 for query (len={len(query)}): {query[:80]}...")
                    return results

                if r.status_code in (403, 429):
                    reset_at = r.headers.get("X-RateLimit-Reset", "")
                    if reset_at and reset_at.isdigit():
                        wait = max(int(reset_at) - int(time.time()), 0) + 2
                    else:
                        wait = min(2 ** attempt, 120)
                    mark_token_cooldown(token, wait)
                    token = get_next_token()
                    session = _get_session(token)
                    continue

                if r.status_code in (500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 60))
                    continue

                break
            except requests.RequestException:
                time.sleep(min(2 ** attempt, 60))
        else:
            break

        # Burst-smoothing: tiny sleep after every successful request
        time.sleep(0.2)

        if r is None or r.status_code != 200:
            break

        try:
            data = r.json()
        except ValueError:
            break

        total_count = data.get("total_count", 0)
        if total_count == 0:
            break

        items = data.get("items", [])
        for item in items:
            repo = item.get("repository", {}).get("full_name", "")
            fpath = item.get("path", "")
            if repo:
                results.append((repo, fpath))

        if len(items) < RESULTS_PER_PAGE or page >= max_pages:
            break

        page += 1
        time.sleep(0.5)  # small page sleep; per-token pacing handles rate limits

    return results


def verify_model(model_id: str, original_repos: set) -> dict:
    """
    For an ambiguous model, search with quantization context keywords one at a time.
    GitHub Code Search doesn't reliably support OR with quoted strings,
    so we use individual queries with strong→weak ordering and early termination.
    """
    verified_repos = set()
    verified_files = defaultdict(set)

    # Strong keywords get more pages; weak keywords get fewer
    # Split into tiers for page limits
    strong_keywords = set(QUANT_CONTEXT_KEYWORDS[:8])   # APIs, explicit loaders

    for keyword in QUANT_CONTEXT_KEYWORDS:
        if verified_repos == original_repos:
            break  # all repos verified, stop early

        max_pages = 3 if keyword in strong_keywords else 2
        query = f'"{model_id}" "{keyword}"'
        results = search_code(query, max_pages=max_pages)

        for repo, fpath in results:
            if repo in original_repos:
                verified_repos.add(repo)
                verified_files[repo].add(fpath)

    unverified_repos = original_repos - verified_repos

    return {
        "model_id": model_id,
        "original_repo_count": len(original_repos),
        "verified_repo_count": len(verified_repos),
        "unverified_repo_count": len(unverified_repos),
        "verified_repos": {
            repo: {
                "files": sorted(verified_files[repo]),
            }
            for repo in sorted(verified_repos)
        },
        "unverified_repos": sorted(unverified_repos),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify ambiguous model GitHub results with quantization context"
    )
    parser.add_argument("--ambiguous-models", required=True,
                        help="File with ambiguous model IDs (one per line)")
    parser.add_argument("--original-results", required=True,
                        help="Original search_results.jsonl from Code Search")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load ambiguous model set
    with open(args.ambiguous_models) as f:
        ambiguous_set = {line.strip() for line in f if line.strip()}

    # Load original results — extract repos per ambiguous model
    model_original_repos = {}
    with open(args.original_results) as f:
        for line in f:
            d = json.loads(line)
            mid = d["model_id"]
            if mid in ambiguous_set and d["num_repos"] > 0:
                model_original_repos[mid] = set(d["unique_repos"])

    print(f"Ambiguous Model Verification")
    print(f"{'='*60}")
    print(f"Total ambiguous models: {len(ambiguous_set):,}")
    print(f"Ambiguous models with GitHub results: {len(model_original_repos):,}")
    print(f"Tokens: {len(TOKENS)}")
    print(f"Workers: {args.workers}")
    print(f"Context keywords: {len(QUANT_CONTEXT_KEYWORDS)}")
    print(f"Max queries: ~{len(model_original_repos) * len(QUANT_CONTEXT_KEYWORDS):,} (1 per keyword per model, with early stop)")
    print()

    # Checkpoint — append-only processed log
    processed_path = os.path.join(args.output_dir, "verified_processed.txt")
    results_path = os.path.join(args.output_dir, "verification_results.jsonl")

    processed_set = set()
    if os.path.exists(processed_path):
        with open(processed_path) as f:
            processed_set = {line.strip() for line in f if line.strip()}

    remaining = {mid: repos for mid, repos in model_original_repos.items()
                 if mid not in processed_set}
    print(f"Already processed: {len(processed_set):,}")
    print(f"Remaining: {len(remaining):,}")
    print()

    _write_lock = threading.Lock()

    t0 = time.time()

    def process_one(mid_repos_tuple):
        mid, orig_repos = mid_repos_tuple
        return verify_model(mid, orig_repos)

    items = list(remaining.items())

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Process in chunks for periodic stats
        chunk_size = 50
        for chunk_start in range(0, len(items), chunk_size):
            chunk = items[chunk_start:chunk_start + chunk_size]

            futures = {executor.submit(process_one, item): item[0] for item in chunk}

            for future in as_completed(futures):
                mid = futures[future]
                try:
                    result = future.result()

                    with _write_lock:
                        with open(results_path, "a") as f:
                            f.write(json.dumps(result, ensure_ascii=False, default=list) + "\n")
                        with open(processed_path, "a") as f:
                            f.write(mid + "\n")
                        processed_set.add(mid)

                    if result["verified_repo_count"] > 0:
                        log_msg(f"  ✓ {mid} -> {result['verified_repo_count']} verified, "
                                f"{result['unverified_repo_count']} unverified "
                                f"(of {result['original_repo_count']})")

                except Exception as e:
                    log_msg(f"  ✗ Error for {mid}: {e}")
                    with _write_lock:
                        with open(processed_path, "a") as f:
                            f.write(mid + "\n")
                        processed_set.add(mid)

            # Progress
            elapsed = time.time() - t0
            done_this_run = chunk_start + len(chunk)
            rate = done_this_run / elapsed if elapsed > 0 else 0
            remaining_count = len(items) - done_this_run
            eta_h = remaining_count / rate / 3600 if rate > 0 else 0
            log_msg(f"  Progress: {done_this_run}/{len(items)} | "
                    f"Rate: {rate:.2f} models/s | ETA: {eta_h:.1f}h")

    # ==========================================
    # Generate summary
    # ==========================================
    total_verified = 0
    total_unverified = 0
    all_verified_repos = set()
    all_unverified_repos = set()

    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                total_verified += r["verified_repo_count"]
                total_unverified += r["unverified_repo_count"]
                all_verified_repos.update(r["verified_repos"].keys())
                all_unverified_repos.update(r["unverified_repos"])

    # Repos only in unverified (not verified by any model)
    only_unverified = all_unverified_repos - all_verified_repos

    stats = {
        "total_ambiguous_models_checked": len(model_original_repos),
        "total_verified_repos": len(all_verified_repos),
        "total_unverified_repos": len(only_unverified),
        "total_overlap": len(all_verified_repos & all_unverified_repos),
    }

    stats_path = os.path.join(args.output_dir, "verification_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Verified repo list
    verified_list_path = os.path.join(args.output_dir, "verified_repos.txt")
    with open(verified_list_path, "w") as f:
        for repo in sorted(all_verified_repos):
            f.write(repo + "\n")

    # Unverified repo list
    unverified_list_path = os.path.join(args.output_dir, "unverified_repos.txt")
    with open(unverified_list_path, "w") as f:
        for repo in sorted(only_unverified):
            f.write(repo + "\n")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Models checked: {len(model_original_repos):,}")
    print(f"Verified repos (confirmed quantized usage): {len(all_verified_repos):,}")
    print(f"Unverified repos (ambiguous): {len(only_unverified):,}")
    print(f"Time: {elapsed/3600:.1f}h")
    print(f"\nOutput files:")
    print(f"  {results_path}")
    print(f"  {verified_list_path}")
    print(f"  {unverified_list_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
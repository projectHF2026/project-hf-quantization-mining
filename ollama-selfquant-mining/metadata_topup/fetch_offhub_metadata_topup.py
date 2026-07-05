"""Off-hub metadata top-up: add num_contributors_api_approx and
num_commits_default_branch to the 8,424 off-hub repos.

Adapted from output_dir/rq_analysis/rq0/scripts/collect_repo_metadata.py
(hub RQ1 = server-side rq0 — name kept on disk). Same method:
  contributors:  GET /repos/{full_name}/contributors?per_page=1&anon=true
                 parse Link header `rel="last"` → integer count.
                 Special: HTTP 204 → 0 (empty repo).
  commits:       GET /repos/{full_name}/commits?per_page=1&sha=<default_branch>
                 parse Link header `rel="last"` → integer count.
                 Special: HTTP 409 → 0 (empty/conflict).
  default_branch: GET /repos/{full_name} first to obtain it (identical to hub).
  Missing-value convention: integer on success, None on failure
  (mirrors analysis_set_repo_details.jsonl, which has int|None for both fields).

Multi-token pool: GH_TOKEN_1..GH_TOKEN_9, per-token cooldown + 429/403 backoff
+ Retry-After + X-RateLimit-Reset awareness. Same TokenState mechanism as the
hub fetcher. Threaded — one worker per token.

INPUTS
  ollama-selfquant-mining/missed_repos.jsonl        (8,424 repos with stars etc.)
  ollama-selfquant-mining/analysis_offhub/offhub_obtain_repo_ids.txt  (not used
                                                  by this script; available
                                                  for cross-checks)

OUTPUTS (new directory: ollama-selfquant-mining/metadata_topup/)
  offhub_repo_details.jsonl       — one record per repo, fully replaces nothing;
                                    DOES NOT overwrite missed_repos.jsonl
  offhub_processed_repos.txt      — checkpoint: one full_name per line, fsync'd
  offhub_collection_errors.log    — per-error log
  checkpoint.json                 — cumulative counters + last-seen timestamp

USAGE (after exporting GH_TOKEN_1..9):
  python3 fetch_offhub_metadata_topup.py             # full run, resumable
  python3 fetch_offhub_metadata_topup.py --dry-run 50  # process first 50 only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE             = Path("/scratch/oldhome/user/projects/JAW/scripts/icpc-approch")
SIDE             = BASE / "ollama-selfquant-mining"
INPUT_JSONL      = SIDE / "missed_repos.jsonl"

OUT_DIR          = SIDE / "metadata_topup"
OUT_JSONL        = OUT_DIR / "offhub_repo_details.jsonl"
PROCESSED_PATH   = OUT_DIR / "offhub_processed_repos.txt"
ERROR_LOG_PATH   = OUT_DIR / "offhub_collection_errors.log"
CHECKPOINT       = OUT_DIR / "checkpoint.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tunables (mirror hub collect_repo_metadata.py)
# ---------------------------------------------------------------------------

API_BASE            = "https://api.github.com"
GITHUB_API_VERSION  = "2022-11-28"
TIMEOUT             = 30
MAX_RETRIES         = 5
RATE_LIMIT_FLOOR    = 10
PROGRESS_EVERY      = 100


# ---------------------------------------------------------------------------
# Mode classification — same priority rule as analyze_offhub.py
# (self_quantized > ollama_obtains_model > ollama_backend_only)
# ---------------------------------------------------------------------------

OLLAMA_OBTAIN_SIGNALS = {
    "ollama_modelfile_file",
    "ollama_modelfile_FROM",
    "ollama_cli_pull",
    "ollama_cli_create",
    "ollama_cli_push",
    "ollama_registry_old",
    "ollama_registry_library",
}


def classify_mode(repo: dict) -> str | None:
    cats = set(repo.get("categories") or [])
    if "B" in cats:
        return "self_quantized"
    snames = {s.get("signal", "") for s in repo.get("signals") or []}
    if snames & OLLAMA_OBTAIN_SIGNALS:
        return "ollama_obtains_model"
    if "A" in cats:
        return "ollama_backend_only"
    return None


# ---------------------------------------------------------------------------
# Token pool (identical mechanism to hub collect_repo_metadata.py)
# ---------------------------------------------------------------------------

def load_tokens() -> list[tuple[str, str]]:
    out = []
    for i in range(1, 10):
        name = f"GH_TOKEN_{i}"
        val = os.environ.get(name, "").strip()
        if val:
            out.append((name, val))
    return out


def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization":         f"Bearer {token}",
        "Accept":                "application/vnd.github+json",
        "X-GitHub-Api-Version":  GITHUB_API_VERSION,
        "User-Agent":            "jaw-offhub-metadata-topup/1.0",
    })
    return s


class TokenState:
    __slots__ = ("name", "session", "remaining", "reset_at")

    def __init__(self, name: str, token: str) -> None:
        self.name = name
        self.session = make_session(token)
        self.remaining: int | None = None
        self.reset_at: float = 0.0


def update_rate_state(state: TokenState, resp: requests.Response) -> None:
    rem = resp.headers.get("X-RateLimit-Remaining")
    reset = resp.headers.get("X-RateLimit-Reset")
    if rem is not None:
        try:
            state.remaining = int(rem)
        except ValueError:
            pass
    if reset is not None:
        try:
            state.reset_at = float(reset)
        except ValueError:
            pass


def wait_if_low(state: TokenState) -> None:
    if state.remaining is None or state.remaining > RATE_LIMIT_FLOOR:
        return
    sleep_for = max(state.reset_at - time.time() + 5, 1.0)
    if sleep_for > 0:
        print(f"[{state.name}] rate-limit low (remaining={state.remaining}), "
              f"sleeping {sleep_for:.0f}s until reset")
        time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Link header parsing (identical to hub)
# ---------------------------------------------------------------------------

LINK_LAST_RE  = re.compile(r'<([^>]+)>;\s*rel="last"')
PAGE_PARAM_RE = re.compile(r"[?&]page=(\d+)")


def parse_last_page(link_header: str | None) -> int | None:
    if not link_header:
        return None
    m = LINK_LAST_RE.search(link_header)
    if not m:
        return None
    pm = PAGE_PARAM_RE.search(m.group(1))
    if pm:
        try:
            return int(pm.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Request wrapper (identical retry/backoff semantics as hub)
# ---------------------------------------------------------------------------

def github_request(state: TokenState, method: str, url: str, **kwargs):
    kwargs.setdefault("timeout", TIMEOUT)
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        wait_if_low(state)
        try:
            resp = state.session.request(method, url, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            backoff = min(2 ** attempt, 30) + random.random()
            time.sleep(backoff)
            continue

        update_rate_state(state, resp)

        if resp.status_code in (403, 429):
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_for = float(retry_after) + 1.0
                except ValueError:
                    sleep_for = 60.0
            elif state.remaining == 0 and state.reset_at:
                sleep_for = max(state.reset_at - time.time() + 5, 5)
            else:
                sleep_for = min(2 ** attempt * 2, 60)
            print(f"[{state.name}] HTTP {resp.status_code} backoff {sleep_for:.0f}s "
                  f"(attempt {attempt+1}/{MAX_RETRIES}) {url}")
            time.sleep(sleep_for)
            continue

        return resp

    if last_exc is not None:
        raise last_exc
    return resp


# ---------------------------------------------------------------------------
# Per-repo fetch — three calls (same 3-call flow as hub)
#   1) /repos              → repo_status + default_branch (and 404/403 catch)
#   2) /repos/.../contributors?per_page=1&anon=true   → Link header trick
#   3) /repos/.../commits?per_page=1&sha=<default>     → Link header trick
# Missing-value convention: integer on success, None on failure (matches the
# downstream parse_int_or_none() behaviour in build_analysis_set.py).
# ---------------------------------------------------------------------------

def fetch_metadata(state: TokenState, full_name: str) -> dict:
    """Return a dict with the two new fields + status fields. Carries no
    fields from the existing missed_repos.jsonl; the caller merges."""
    row: dict[str, Any] = {
        "num_contributors_api_approx":  None,
        "num_commits_default_branch":   None,
        "default_branch":               None,
        "repo_status":                  None,
        "repo_http_status":             None,
        "repo_error":                   None,
        "full_name_resolved":           None,
        "contributors_status":          None,
        "contributors_http_status":     None,
        "contributors_error":           None,
        "commits_status":               None,
        "commits_http_status":          None,
        "commits_error":                None,
        "topup_collected_at_utc":       datetime.now(timezone.utc).isoformat(
                                            timespec="seconds"),
        "github_api_version":           GITHUB_API_VERSION,
    }

    # ---- (1) primary metadata to get default_branch + detect 404/403 ----
    try:
        r = github_request(state, "GET", f"{API_BASE}/repos/{full_name}",
                           allow_redirects=False)
    except Exception as e:
        row["repo_status"] = "error"
        row["repo_error"] = f"{type(e).__name__}: {e}"
        return row
    if r is None:
        row["repo_status"] = "error"
        row["repo_error"] = "no response"
        return row

    row["repo_http_status"] = r.status_code

    if r.status_code == 301:
        new_url = r.headers.get("Location", "")
        # Try the resolved repo: parse the /repos/{owner}/{name} path
        m = re.search(r"/repos/([^/]+/[^/?#]+)", new_url)
        if m:
            resolved = m.group(1)
            row["full_name_resolved"] = resolved
            try:
                r = github_request(state, "GET",
                                   f"{API_BASE}/repos/{resolved}",
                                   allow_redirects=False)
            except Exception as e:
                row["repo_status"] = "error"
                row["repo_error"] = f"{type(e).__name__}: {e}"
                return row
            if r is None:
                row["repo_status"] = "error"
                row["repo_error"] = "no response after 301"
                return row
            row["repo_http_status"] = r.status_code
        else:
            row["repo_status"] = "301_no_location"
            return row

    if r.status_code == 404:
        row["repo_status"] = "404_not_found"
        return row
    if r.status_code == 403:
        row["repo_status"] = "403_forbidden"
        row["repo_error"] = (r.text or "")[:200]
        return row
    if r.status_code != 200:
        row["repo_status"] = f"http_{r.status_code}"
        row["repo_error"] = (r.text or "")[:200]
        return row

    try:
        data = r.json()
    except Exception as e:
        row["repo_status"] = "json_decode_error"
        row["repo_error"] = f"{type(e).__name__}: {e}"
        return row

    row["repo_status"]         = "ok"
    row["default_branch"]      = data.get("default_branch") or None
    row["full_name_resolved"]  = row["full_name_resolved"] or data.get("full_name")
    target_full_name           = row["full_name_resolved"] or full_name
    default_branch             = row["default_branch"]

    # ---- (2) contributors ----
    try:
        r2 = github_request(
            state, "GET",
            f"{API_BASE}/repos/{target_full_name}/contributors",
            params={"per_page": 1, "anon": "true"},
        )
    except Exception as e:
        row["contributors_status"] = "error"
        row["contributors_error"] = f"{type(e).__name__}: {e}"
    else:
        if r2 is None:
            row["contributors_status"] = "error"
            row["contributors_error"] = "no response"
        else:
            row["contributors_http_status"] = r2.status_code
            if r2.status_code == 204:
                row["num_contributors_api_approx"] = 0
                row["contributors_status"] = "empty"
            elif r2.status_code == 404:
                row["contributors_status"] = "404_not_found"
            elif r2.status_code == 403:
                row["contributors_status"] = "403_forbidden"
                row["contributors_error"] = (r2.text or "")[:200]
            elif r2.status_code == 429:
                row["contributors_status"] = "429_rate_limited"
            elif r2.status_code == 200:
                last = parse_last_page(r2.headers.get("Link", ""))
                if last is not None:
                    row["num_contributors_api_approx"] = last
                    row["contributors_status"] = "ok"
                else:
                    try:
                        arr = r2.json()
                        row["num_contributors_api_approx"] = (
                            len(arr) if isinstance(arr, list) else 0
                        )
                    except Exception:
                        row["num_contributors_api_approx"] = 0
                    row["contributors_status"] = "no_link_header"
            else:
                row["contributors_status"] = "error"
                row["contributors_error"] = (
                    f"HTTP {r2.status_code}: {(r2.text or '')[:200]}"
                )

    # ---- (3) commits ----
    if not default_branch:
        row["commits_status"] = "empty"
        return row

    try:
        r3 = github_request(
            state, "GET",
            f"{API_BASE}/repos/{target_full_name}/commits",
            params={"per_page": 1, "sha": default_branch},
        )
    except Exception as e:
        row["commits_status"] = "error"
        row["commits_error"] = f"{type(e).__name__}: {e}"
        return row
    if r3 is None:
        row["commits_status"] = "error"
        row["commits_error"] = "no response"
        return row

    row["commits_http_status"] = r3.status_code
    if r3.status_code == 409:
        row["num_commits_default_branch"] = 0
        row["commits_status"] = "empty_or_conflict"
    elif r3.status_code == 404:
        row["commits_status"] = "404_not_found"
    elif r3.status_code == 403:
        row["commits_status"] = "403_forbidden"
        row["commits_error"] = (r3.text or "")[:200]
    elif r3.status_code == 429:
        row["commits_status"] = "429_rate_limited"
    elif r3.status_code == 200:
        last = parse_last_page(r3.headers.get("Link", ""))
        if last is not None:
            row["num_commits_default_branch"] = last
            row["commits_status"] = "ok"
        else:
            try:
                arr = r3.json()
                row["num_commits_default_branch"] = (
                    len(arr) if isinstance(arr, list) else 0
                )
            except Exception:
                row["num_commits_default_branch"] = 0
            row["commits_status"] = "no_link_header"
    else:
        row["commits_status"] = "error"
        row["commits_error"] = f"HTTP {r3.status_code}: {(r3.text or '')[:200]}"

    return row


# ---------------------------------------------------------------------------
# Unified single fetch_status field (summarises the per-call statuses)
# ---------------------------------------------------------------------------

def overall_fetch_status(top: dict) -> str:
    rs = top.get("repo_status")
    cs = top.get("contributors_status")
    ms = top.get("commits_status")
    if rs in ("404_not_found", "403_forbidden"):
        return rs
    if rs and rs not in ("ok",):
        if rs.startswith("301") or rs.startswith("http_") or rs in ("error", "json_decode_error", "301_no_location"):
            return rs
    if cs == "ok" and ms == "ok":
        return "ok"
    if cs == "ok" or ms == "ok":
        return f"partial:contrib={cs}|commits={ms}"
    return f"failed:contrib={cs}|commits={ms}"


# ---------------------------------------------------------------------------
# Input loader: missed_repos.jsonl → (full_name, mode, carry-over dict)
# ---------------------------------------------------------------------------

CARRY_FIELDS = (
    "full_name", "html_url", "stars", "forks", "created_at", "pushed_at",
    "archived", "fork", "language", "categories", "signals", "first_seen",
    "already_counted", "repo_status",  # repo_status from side-study fetch
)


def load_inputs() -> list[tuple[str, str, dict]]:
    if not INPUT_JSONL.exists():
        sys.exit(f"ERROR: missing input {INPUT_JSONL}")
    out: list[tuple[str, str, dict]] = []
    n_no_mode = 0
    with INPUT_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            fn = r.get("full_name")
            if not fn:
                continue
            mode = classify_mode(r)
            if mode is None:
                n_no_mode += 1
                mode = "unclassified"
            carry = {k: r.get(k) for k in CARRY_FIELDS if k in r}
            # Rename collision: side-study's "repo_status" overlaps with
            # our top-up repo_status. Keep both via a side_study_ prefix.
            if "repo_status" in carry:
                carry["side_study_repo_status"] = carry.pop("repo_status")
            out.append((fn, mode, carry))
    if n_no_mode:
        print(f"[init] WARNING: {n_no_mode} repos have no classifiable mode "
              f"(should be 0; defaulting to 'unclassified')")
    return out


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", type=int, default=0,
                    help="Process only the first N repos (testing).")
    args = ap.parse_args()

    tokens = load_tokens()
    if not tokens:
        sys.exit("ERROR: no GH_TOKEN_1..GH_TOKEN_9 env vars set. "
                 "Export your tokens and re-run.")
    print(f"Loaded {len(tokens)} GitHub token(s): {[n for n, _ in tokens]}")
    states = [TokenState(n, t) for n, t in tokens]

    inputs = load_inputs()
    print(f"Loaded {len(inputs):,} repos from {INPUT_JSONL.name}")
    if args.dry_run > 0:
        inputs = inputs[:args.dry_run]
        print(f"DRY-RUN: limiting to first {args.dry_run} repos")

    processed: set[str] = set()
    if PROCESSED_PATH.exists():
        processed = {
            line.strip() for line in PROCESSED_PATH.read_text().splitlines()
            if line.strip()
        }
    print(f"Already in checkpoint: {len(processed):,}")
    todo = [(fn, m, c) for (fn, m, c) in inputs if fn not in processed]
    print(f"To fetch: {len(todo):,}")
    if not todo:
        print("Nothing to do. Exiting.")
        return

    # Open output file APPEND mode so resume works
    out_fh = OUT_JSONL.open("a", encoding="utf-8")
    proc_fh = PROCESSED_PATH.open("a", encoding="utf-8")
    err_fh = ERROR_LOG_PATH.open("a", encoding="utf-8")
    write_lock = threading.Lock()

    def write_row(row: dict) -> None:
        with write_lock:
            out_fh.write(json.dumps(row) + "\n")
            proc_fh.write(row["full_name"] + "\n")
            out_fh.flush()
            proc_fh.flush()
            os.fsync(out_fh.fileno())
            os.fsync(proc_fh.fileno())

    def log_error(repo: str, exc: BaseException) -> None:
        with write_lock:
            err_fh.write(
                f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
                f"{repo} | {type(exc).__name__}: {exc}\n"
            )
            err_fh.flush()

    queue: list[tuple[str, str, dict]] = list(reversed(todo))
    queue_lock = threading.Lock()

    progress = {
        "n": 0, "ok": 0, "failed": 0, "lock": threading.Lock(),
        "start": time.time(), "total": len(todo),
    }

    def take_one():
        with queue_lock:
            if queue:
                return queue.pop()
            return None

    def print_progress() -> None:
        n = progress["n"]
        elapsed = time.time() - progress["start"]
        rate = n / elapsed if elapsed > 0 else 0.0
        remaining = progress["total"] - n
        etr = (f"{remaining/rate/60:.1f}min" if rate > 0 else "?min")
        token_str = ", ".join(
            f"{s.name.replace('GH_TOKEN_', 'T')}="
            f"{s.remaining if s.remaining is not None else '?'}"
            for s in states
        )
        print(f"  [progress] {n:,}/{progress['total']:,} "
              f"({100*n/progress['total']:.1f}%) "
              f"OK={progress['ok']:,} FAIL={progress['failed']:,} "
              f"rate={rate:.2f}/s elapsed={elapsed/60:.1f}min ETR={etr} "
              f"tokens=[{token_str}]")

    def worker(state: TokenState) -> None:
        while True:
            item = take_one()
            if item is None:
                return
            full_name, mode, carry = item
            try:
                new_fields = fetch_metadata(state, full_name)
            except Exception as e:
                log_error(full_name, e)
                row = dict(carry)
                row["full_name"] = full_name
                row["mode"] = mode
                row["num_contributors_api_approx"] = None
                row["num_commits_default_branch"] = None
                row["fetch_status"] = "exception"
                row["fetch_error"] = f"{type(e).__name__}: {e}"
                row["topup_collected_at_utc"] = datetime.now(
                    timezone.utc).isoformat(timespec="seconds")
                with progress["lock"]:
                    progress["n"] += 1
                    progress["failed"] += 1
                    if progress["n"] % PROGRESS_EVERY == 0:
                        print_progress()
                write_row(row)
                continue

            row = dict(carry)
            row["full_name"] = full_name
            row["mode"] = mode
            row.update(new_fields)
            row["fetch_status"] = overall_fetch_status(new_fields)

            with progress["lock"]:
                progress["n"] += 1
                if (new_fields.get("contributors_status") == "ok" and
                    new_fields.get("commits_status") == "ok"):
                    progress["ok"] += 1
                else:
                    progress["failed"] += 1
                if progress["n"] % PROGRESS_EVERY == 0:
                    print_progress()
            write_row(row)

    with ThreadPoolExecutor(max_workers=len(states)) as ex:
        futs = [ex.submit(worker, s) for s in states]
        for fut in futs:
            fut.result()

    out_fh.close()
    proc_fh.close()
    err_fh.close()

    elapsed = time.time() - progress["start"]
    print()
    print("=" * 72)
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"  total processed:       {progress['n']:,}")
    print(f"  OK (both calls ok):    {progress['ok']:,}")
    print(f"  FAILED or partial:     {progress['failed']:,}")
    print(f"  outputs:")
    print(f"    {OUT_JSONL}")
    print(f"    {PROCESSED_PATH}")
    print(f"    {ERROR_LOG_PATH}")
    # Write a small checkpoint with totals (no resume state — that's in
    # PROCESSED_PATH and OUT_JSONL).
    cp = {
        "completed_at_utc":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_processed":   progress["n"],
        "ok":                progress["ok"],
        "failed_or_partial": progress["failed"],
        "wallclock_seconds": int(elapsed),
        "tokens_used":       len(states),
        "input_repos":       INPUT_JSONL.name,
    }
    CHECKPOINT.write_text(json.dumps(cp, indent=2))


if __name__ == "__main__":
    main()

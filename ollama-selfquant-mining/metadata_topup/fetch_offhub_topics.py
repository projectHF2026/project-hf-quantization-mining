"""Off-hub topics top-up — small dedicated fetcher.

Single API call per repo (GET /repos/{full_name}) → persist the `topics` array
the side-study's original fetcher dropped. Resumable, multi-token pool +
rate-limit handling identical to fetch_offhub_metadata_topup.py.

INPUT
  offhub_repo_details.jsonl   (8,424 records; reads only full_name)

OUTPUT
  offhub_topics.jsonl         (one record/repo: full_name, topics, fetch_status,
                              http_status, fetched_at_utc)
  offhub_topics_processed.txt (checkpoint)
  offhub_topics_errors.log    (per-failure log)

After this runs, a separate merge step folds topics back into
offhub_repo_details.jsonl.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT       = Path(__file__).resolve().parent
INPUT      = ROOT / "offhub_repo_details.jsonl"
OUTPUT     = ROOT / "offhub_topics.jsonl"
PROCESSED  = ROOT / "offhub_topics_processed.txt"
ERRORS     = ROOT / "offhub_topics_errors.log"

API_BASE           = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
TIMEOUT            = 30
MAX_RETRIES        = 5
RATE_LIMIT_FLOOR   = 10
PROGRESS_EVERY     = 250


def load_tokens() -> list[tuple[str, str]]:
    out = []
    for i in range(1, 10):
        n = f"GH_TOKEN_{i}"
        v = os.environ.get(n, "").strip()
        if v:
            out.append((n, v))
    return out


def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        # mercy-preview historically required for topics; modern API returns
        # them by default but the header is harmless and explicit.
        "Accept": ("application/vnd.github+json, "
                   "application/vnd.github.mercy-preview+json"),
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "jaw-offhub-topics-topup/1.0",
    })
    return s


class TokenState:
    __slots__ = ("name", "session", "remaining", "reset_at")

    def __init__(self, name, token):
        self.name = name
        self.session = make_session(token)
        self.remaining = None
        self.reset_at = 0.0


def update_rate(state, resp):
    rem = resp.headers.get("X-RateLimit-Remaining")
    rst = resp.headers.get("X-RateLimit-Reset")
    if rem is not None:
        try: state.remaining = int(rem)
        except ValueError: pass
    if rst is not None:
        try: state.reset_at = float(rst)
        except ValueError: pass


def wait_if_low(state):
    if state.remaining is None or state.remaining > RATE_LIMIT_FLOOR:
        return
    sleep_for = max(state.reset_at - time.time() + 5, 1.0)
    if sleep_for > 0:
        print(f"[{state.name}] low rate (remaining={state.remaining}), "
              f"sleeping {sleep_for:.0f}s")
        time.sleep(sleep_for)


def gh_get(state, url):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        wait_if_low(state)
        try:
            r = state.session.get(url, timeout=TIMEOUT)
        except requests.RequestException as e:
            last_exc = e
            time.sleep(min(2**attempt, 30) + random.random())
            continue
        update_rate(state, r)
        if r.status_code in (403, 429):
            ra = r.headers.get("Retry-After")
            if ra:
                try: s = float(ra) + 1.0
                except ValueError: s = 60.0
            elif state.remaining == 0 and state.reset_at:
                s = max(state.reset_at - time.time() + 5, 5)
            else:
                s = min(2**attempt * 2, 60)
            print(f"[{state.name}] HTTP {r.status_code} backoff {s:.0f}s "
                  f"(attempt {attempt+1}/{MAX_RETRIES}) {url}")
            time.sleep(s)
            continue
        return r
    if last_exc:
        raise last_exc
    return r


def fetch_topics(state, full_name):
    row = {
        "full_name":       full_name,
        "topics":          None,
        "fetch_status":    None,
        "http_status":     None,
        "fetched_at_utc":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        r = gh_get(state, f"{API_BASE}/repos/{full_name}")
    except Exception as e:
        row["fetch_status"] = "exception"
        row["fetch_error"]  = f"{type(e).__name__}: {e}"
        return row
    if r is None:
        row["fetch_status"] = "no_response"
        return row
    row["http_status"] = r.status_code
    if r.status_code == 200:
        try:
            d = r.json()
            row["topics"]       = d.get("topics") or []
            row["fetch_status"] = "ok"
        except Exception as e:
            row["fetch_status"] = "json_decode_error"
            row["fetch_error"]  = f"{type(e).__name__}: {e}"
        return row
    if r.status_code == 404:
        row["fetch_status"] = "404_not_found"
    elif r.status_code == 403:
        row["fetch_status"] = "403_forbidden"
        row["fetch_error"]  = (r.text or "")[:200]
    else:
        row["fetch_status"] = f"http_{r.status_code}"
        row["fetch_error"]  = (r.text or "")[:200]
    return row


def main():
    tokens = load_tokens()
    if not tokens:
        sys.exit("ERROR: export GH_TOKEN_1..GH_TOKEN_9 first")
    print(f"Loaded {len(tokens)} tokens: {[n for n, _ in tokens]}")
    states = [TokenState(n, t) for n, t in tokens]

    full_names = []
    seen = set()
    with INPUT.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            fn = r.get("full_name")
            if fn and fn not in seen:
                full_names.append(fn)
                seen.add(fn)
    print(f"Loaded {len(full_names):,} full_names from {INPUT.name}")

    processed = set()
    if PROCESSED.exists():
        processed = {ln.strip() for ln in PROCESSED.read_text().splitlines() if ln.strip()}
    todo = [fn for fn in full_names if fn not in processed]
    print(f"Already processed: {len(processed):,}; to fetch: {len(todo):,}")
    if not todo:
        print("Nothing to do.")
        return

    out_fh  = OUTPUT.open("a", encoding="utf-8")
    proc_fh = PROCESSED.open("a", encoding="utf-8")
    err_fh  = ERRORS.open("a", encoding="utf-8")
    lock = threading.Lock()

    def write(row):
        with lock:
            out_fh.write(json.dumps(row) + "\n")
            proc_fh.write(row["full_name"] + "\n")
            out_fh.flush(); proc_fh.flush()

    queue = list(reversed(todo))
    qlock = threading.Lock()
    prog = {"n": 0, "ok": 0, "fail": 0, "with_topics": 0,
            "start": time.time(), "total": len(todo), "lock": threading.Lock()}

    def take():
        with qlock:
            return queue.pop() if queue else None

    def print_progress():
        n = prog["n"]
        el = time.time() - prog["start"]
        rate = n / el if el > 0 else 0
        rem = prog["total"] - n
        etr = f"{rem/rate/60:.1f}min" if rate > 0 else "?"
        tk = ", ".join(f"{s.name.replace('GH_TOKEN_', 'T')}="
                       f"{s.remaining if s.remaining is not None else '?'}"
                       for s in states)
        print(f"  [progress] {n:,}/{prog['total']:,} "
              f"({100*n/prog['total']:.1f}%)  OK={prog['ok']:,} "
              f"FAIL={prog['fail']:,} with_topics={prog['with_topics']:,} "
              f"rate={rate:.2f}/s ETR={etr}  tokens=[{tk}]")

    def worker(state):
        while True:
            fn = take()
            if fn is None: return
            row = fetch_topics(state, fn)
            with prog["lock"]:
                prog["n"] += 1
                if row.get("fetch_status") == "ok":
                    prog["ok"] += 1
                    if row.get("topics"):
                        prog["with_topics"] += 1
                else:
                    prog["fail"] += 1
                if prog["n"] % PROGRESS_EVERY == 0:
                    print_progress()
            write(row)
            if row.get("fetch_status") not in ("ok", "404_not_found"):
                err_fh.write(
                    f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
                    f" | {fn} | {row.get('fetch_status')} | "
                    f"{(row.get('fetch_error') or '')[:120]}\n")
                err_fh.flush()

    with ThreadPoolExecutor(max_workers=len(states)) as ex:
        futs = [ex.submit(worker, s) for s in states]
        for f in futs:
            f.result()

    out_fh.close(); proc_fh.close(); err_fh.close()
    el = time.time() - prog["start"]
    print()
    print("=" * 60)
    print(f"DONE in {el/60:.1f} min")
    print(f"  total processed:    {prog['n']:,}")
    print(f"  OK (200):           {prog['ok']:,}")
    print(f"  FAILED:             {prog['fail']:,}")
    print(f"  with non-empty topics: {prog['with_topics']:,}")
    print(f"  outputs: {OUTPUT.name}, {PROCESSED.name}, {ERRORS.name}")


if __name__ == "__main__":
    main()

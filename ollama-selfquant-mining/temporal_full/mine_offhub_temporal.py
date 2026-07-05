"""Off-hub temporal mining — adapted from the hub RQ2 phase1 miner.

Mirrors output_dir/rq_analysis/rq2/phase1_full/mine_full.py:
  - default-branch-only first-commit recovery via git log -S pickaxe
  - rate-limited cloning (60/60s rolling window + exponential backoff)
  - per-repo timeout + oversized-history auto-skip on first pickaxe timeout
  - resumable via checkpoint.json + skip-list from existing JSONL output
  - aggressive disk strategy (delete clone after mining)

Inputs (relative to icpc-approch/):
  ollama-selfquant-mining/analysis_offhub/offhub_obtain_repo_ids.txt
    (3,830 obtain-or-produce repo IDs)
  ollama-selfquant-mining/missed_repos.jsonl
    (full per-repo records; used to re-derive self_quantized vs
     ollama_obtains_model mode with the SAME priority rule analyze_offhub.py
     used: self_quantized first, then ollama_obtains_model)

Outputs (in ollama-selfquant-mining/temporal_full/):
  results/offhub_signal_first_dates.jsonl
    — one record per (repo_id, signal_family) with first-commit details
  offhub_temporal_first_adoption.csv
    — collapsed: one row per repo with the EARLIEST signal across families
      (columns: repo_id, mode, first_commit_date, fired_signal, method_family,
      pulled_model_id)
  offhub_clone_log.jsonl
    — one record per clone attempt (status: success/404/auth_required/...)
  checkpoint.json
    — resume marker (last_batch_completed + cumulative counters)
  logs/batch_<N>.log
    — per-batch summary
  logs/oversized_skip_log.jsonl
    — first-pickaxe-timeout entries (oversized histories)
  logs/rate_limit_events.jsonl
    — 429 / abuse backoffs

Launch:
  cd /scratch/oldhome/user/projects/JAW/scripts/icpc-approch/ollama-selfquant-mining/temporal_full
  nohup python3 -u mine_offhub_temporal.py > pipeline.log 2>&1 &
  echo $! > mine.pid
  tail -f pipeline.log
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths (server-side names stay; only paper prose uses RQ3)
# ---------------------------------------------------------------------------

BASE = Path("/scratch/oldhome/user/projects/JAW/scripts/icpc-approch")
SIDE = BASE / "ollama-selfquant-mining"

REPO_LIST_TXT  = SIDE / "analysis_offhub" / "offhub_obtain_repo_ids.txt"
MISSED_JSONL   = SIDE / "missed_repos.jsonl"

FULL_DIR       = SIDE / "temporal_full"
CLONES_DIR     = FULL_DIR / "clones"
RESULTS_DIR    = FULL_DIR / "results"
LOGS_DIR       = FULL_DIR / "logs"
SIGNALS_JSONL  = RESULTS_DIR / "offhub_signal_first_dates.jsonl"
OUT_CSV        = FULL_DIR / "offhub_temporal_first_adoption.csv"
CLONE_LOG      = FULL_DIR / "offhub_clone_log.jsonl"
CHECKPOINT     = FULL_DIR / "checkpoint.json"
RATE_LIMIT_LOG = LOGS_DIR / "rate_limit_events.jsonl"
OVERSIZED_LOG  = LOGS_DIR / "oversized_skip_log.jsonl"

for d in (FULL_DIR, CLONES_DIR, RESULTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tunables (mirror mine_full.py)
# ---------------------------------------------------------------------------

BATCH_SIZE                       = 1000
CLONE_WORKERS                    = 12
CLONE_TIMEOUT_S                  = 30 * 60   # 30 min per repo
PICKAXE_TIMEOUT_S                = 90        # match hub L1_PICKAXE_TIMEOUT_S
HEAD_FALLBACK_TIMEOUT_S          = 90
MODELFILE_LOOKUP_TIMEOUT_S       = 30
MODEL_NAME_EXTRACT_TIMEOUT_S     = 30

RATE_WINDOW_MAX     = 60
RATE_WINDOW_S       = 60
RATE_BACKOFF_SCHEDULE = [60, 120, 240, 480, 960]

DISK_LIMIT_TB                       = 15.0
MAX_PROJECTED_HOURS                 = 96
MIN_BATCH_CLONE_RATE                = 0.0
GUARDRAIL_BATCHES_FOR_PROJECTION    = 2   # only 4 batches total; lower bar


# ---------------------------------------------------------------------------
# Signal families — Category B self-quant tools + Ollama-obtain signals
# Mirrors ollama_selfquant_miner.py's keyword set. The View B EXPANDED
# LLM-stack families (STRICT + SmoothQuant + AMD-Quark) plus torchao, plus
# the three Ollama obtain channels.
# ---------------------------------------------------------------------------

# Content-based signals: detected via `git log --reverse -S <keyword> -i`
# Each family is a list of literal substrings. First-adoption for the family
# = earliest commit where any of its substrings first enters the history.
KEYWORD_SIGNALS: dict[str, list[str]] = {
    # Self-quantize tool families (Category B, View B EXPANDED + torchao)
    "GPTQ": [
        "AutoGPTQForCausalLM", "auto_gptq", "GPTQModel", "gptqmodel",
        "BaseQuantizeConfig", "GPTQQuantizer", "GPTQConfig",
        "save_pretrained_gptq", "save_quantized",
    ],
    "AWQ": [
        "AutoAWQForCausalLM", "from awq import", "AwqConfig", "llm-awq",
    ],
    "llama.cpp/GGUF-convert": [
        "convert_hf_to_gguf", "convert-hf-to-gguf", "convert_lora_to_gguf",
        "llama-quantize", "GGUFWriter", "from gguf import",
    ],
    "ExLlamaV2": [
        "from exllamav2", "exllamav2/convert",
    ],
    "compressed-tensors": [
        "from compressed_tensors", "compressed_tensors",
    ],
    "SmoothQuant": [
        "smoothquant",
    ],
    "AMD-Quark": [
        "from quark.torch", "quark.torch",
    ],
    "torchao": [
        "from torchao", "int4_weight_only", "int8_weight_only",
    ],
    # Ollama obtain signals (Category A obtain-or-produce)
    "Ollama-pull": [
        "ollama pull", "ollama create", "ollama push",
        "registry.ollama.ai", "ollama.com/library",
    ],
    "Ollama-gguf-ref": [
        ".gguf",
    ],
}

# Path-based signals: detected via git ls-files + first-add commit per path.
# `Modelfile` is conventionally a filename; pickaxe-S on `"FROM "` is too
# noisy (Dockerfile, SQL), so we trigger this family on the presence of a
# file named Modelfile (or "*.Modelfile" / "*/Modelfile") at HEAD.
PATH_SIGNAL_FAMILY = "Ollama-Modelfile"
MODELFILE_NAME_RE  = re.compile(
    r"(?:^|/)(?:[Mm]odelfile|Modelfile\.[A-Za-z0-9_.-]+)$"
)

ALL_FAMILIES = list(KEYWORD_SIGNALS.keys()) + [PATH_SIGNAL_FAMILY]


# Method-family mapping (View B comparable) for the final CSV's
# `method_family` column. Self-quant families map to themselves; the three
# Ollama families collapse into `Ollama-obtain` for downstream consistency
# with the side study's taxonomy.
FAMILY_GROUP: dict[str, str] = {
    "GPTQ":                   "GPTQ",
    "AWQ":                    "AWQ",
    "llama.cpp/GGUF-convert": "llama.cpp/GGUF-convert",
    "ExLlamaV2":              "ExLlamaV2",
    "compressed-tensors":     "compressed-tensors",
    "SmoothQuant":            "SmoothQuant",
    "AMD-Quark":              "AMD-Quark",
    "torchao":                "torchao",
    "Ollama-pull":            "Ollama-obtain",
    "Ollama-gguf-ref":        "Ollama-obtain",
    "Ollama-Modelfile":       "Ollama-obtain",
}


# ---------------------------------------------------------------------------
# pulled_model_id extraction patterns (Ollama obtain only)
# ---------------------------------------------------------------------------

PULLED_MODEL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("modelfile_FROM",      re.compile(r"^\s*FROM\s+(\S+)", re.MULTILINE)),
    ("ollama_pull",         re.compile(r"\bollama\s+pull\s+(\S+)", re.IGNORECASE)),
    ("ollama_create",       re.compile(r"\bollama\s+create\s+(\S+)", re.IGNORECASE)),
    ("ollama_push",         re.compile(r"\bollama\s+push\s+(\S+)", re.IGNORECASE)),
    ("registry_ollama_ai",  re.compile(r"registry\.ollama\.ai/(\S+)", re.IGNORECASE)),
    ("ollama_com_library",  re.compile(r"ollama\.com/library/(\S+)", re.IGNORECASE)),
]


# ---------------------------------------------------------------------------
# Mode classification — re-derive from missed_repos.jsonl with the SAME
# priority rule analyze_offhub.py uses: self_quantized > ollama_obtains_model.
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
    """Mirror analyze_offhub.py:classify_mode. self_quantized > ollama_obtains."""
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
# Helpers
# ---------------------------------------------------------------------------

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def fsync_append_jsonl(path: Path, records: list[dict], lock: threading.Lock) -> None:
    if not records:
        return
    with lock:
        with path.open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())


def load_done_repo_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rid = r.get("repo_id")
                if rid:
                    out.add(rid)
            except Exception:
                pass
    return out


def disk_used_tb(path: Path = BASE) -> float:
    try:
        out = subprocess.run(
            ["df", "-B1", str(path)], capture_output=True, text=True, timeout=10,
        ).stdout.strip().split("\n")
        used_bytes = int(out[-1].split()[2])
        return used_bytes / (1024 ** 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Rate limiter (identical to hub mine_full.py)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Allow at most `max_events` per `window_s` rolling window."""

    def __init__(self, max_events: int, window_s: float) -> None:
        self.max_events = max_events
        self.window_s = window_s
        self.events: collections.deque = collections.deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.time()
                while self.events and self.events[0] < now - self.window_s:
                    self.events.popleft()
                if len(self.events) < self.max_events:
                    self.events.append(now)
                    return
                wait_s = self.events[0] + self.window_s - now + 0.05
            if wait_s > 0:
                time.sleep(wait_s)


rate_limiter = RateLimiter(RATE_WINDOW_MAX, RATE_WINDOW_S)
rate_log_lock = threading.Lock()


def log_rate_event(repo_id: str, attempt: int, backoff_s: int, error: str) -> None:
    with rate_log_lock:
        with RATE_LIMIT_LOG.open("a") as f:
            f.write(json.dumps({
                "timestamp": utcnow(),
                "repo_id": repo_id,
                "attempt": attempt,
                "backoff_seconds": backoff_s,
                "error": error[:300],
            }) + "\n")
            f.flush()
            os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Clone (mirrors hub clone_one)
# ---------------------------------------------------------------------------

def clone_one(repo_id: str, mode: str) -> dict:
    safe = safe_name(repo_id)
    clone_path = CLONES_DIR / safe
    if clone_path.exists():
        shutil.rmtree(clone_path, ignore_errors=True)
    url = f"https://github.com/{repo_id}.git"

    t0 = time.time()
    proc = None
    for attempt_idx, backoff_s in enumerate([0] + RATE_BACKOFF_SCHEDULE):
        if backoff_s > 0:
            time.sleep(backoff_s)

        rate_limiter.acquire()
        try:
            proc = subprocess.run(
                ["git", "clone", "--quiet", "--no-tags", url, str(clone_path)],
                capture_output=True, text=True, timeout=CLONE_TIMEOUT_S, errors="replace",
            )
        except subprocess.TimeoutExpired:
            if clone_path.exists():
                shutil.rmtree(clone_path, ignore_errors=True)
            return {
                "repo_id": repo_id, "mode": mode,
                "clone_status": "timeout",
                "clone_error": f"timed out after {CLONE_TIMEOUT_S}s",
                "clone_seconds": time.time() - t0,
                "timestamp": utcnow(),
            }

        if proc.returncode == 0:
            break
        err = (proc.stderr or "").strip()[:500]
        low = err.lower()
        if "429" in err or "abuse" in low or "secondary rate limit" in low:
            if clone_path.exists():
                shutil.rmtree(clone_path, ignore_errors=True)
            next_backoff = (
                RATE_BACKOFF_SCHEDULE[attempt_idx] if attempt_idx < len(RATE_BACKOFF_SCHEDULE)
                else RATE_BACKOFF_SCHEDULE[-1]
            )
            log_rate_event(repo_id, attempt_idx + 1, next_backoff, err)
            continue
        break

    elapsed = time.time() - t0
    if proc is None or proc.returncode != 0:
        err = (proc.stderr or "").strip()[:500] if proc is not None else "no response"
        low = err.lower()
        if "not found" in low or "repository not found" in low:
            status = "404"
        elif "authentication" in low or "could not read username" in low:
            status = "auth_required"
        elif "abuse" in low or "429" in err or "secondary rate limit" in low:
            status = "rate_limited"
        else:
            status = "other_error"
        if clone_path.exists():
            shutil.rmtree(clone_path, ignore_errors=True)
        return {
            "repo_id": repo_id, "mode": mode,
            "clone_status": status,
            "clone_error": err,
            "clone_seconds": elapsed,
            "timestamp": utcnow(),
        }

    def git_str(args, timeout=60):
        try:
            return subprocess.run(
                args, cwd=str(clone_path), capture_output=True, text=True,
                timeout=timeout, errors="replace",
            ).stdout.strip()
        except Exception:
            return ""

    default_branch = git_str(["git", "symbolic-ref", "--short", "HEAD"], 30)
    try:
        commit_count = int(git_str(["git", "rev-list", "--count", "HEAD"], 60))
    except Exception:
        commit_count = 0

    return {
        "repo_id": repo_id, "mode": mode,
        "clone_status": "success",
        "default_branch": default_branch,
        "default_branch_commit_count": commit_count,
        "clone_seconds": elapsed,
        "timestamp": utcnow(),
    }


# ---------------------------------------------------------------------------
# Pickaxe with first-pickaxe-timeout oversized-skip guardrail
# ---------------------------------------------------------------------------

def run_git(args: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, errors="replace",
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"


def mine_repo(repo_id: str, mode: str, clone_path: Path) -> list[dict]:
    """For each signal family, find the earliest commit (default branch only)
    where any of its substrings first enters the history. If the VERY FIRST
    pickaxe of the repo times out (oversized history), abort the whole repo
    and emit a single skipped_too_large record so resume skips it forever."""
    out_records: list[dict] = []
    _first_pickaxe_done = False

    for family, keywords in KEYWORD_SIGNALS.items():
        earliest_commit = None
        earliest_date = None
        matched_kw = None
        had_error = False
        err_msg = ""

        for kw in keywords:
            rc, out, err = run_git(
                ["git", "log", "--reverse", "-S", kw, "-i", "--format=%H,%cI"],
                clone_path, PICKAXE_TIMEOUT_S,
            )
            if rc == 124 and not _first_pickaxe_done:
                # Whole-repo abort
                try:
                    with OVERSIZED_LOG.open("a") as f:
                        f.write(json.dumps({
                            "timestamp":       utcnow(),
                            "repo_id":         repo_id,
                            "mode":            mode,
                            "reason":          "first_pickaxe_timeout",
                            "keyword":         kw,
                            "timeout_seconds": PICKAXE_TIMEOUT_S,
                        }) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    pass
                return [{
                    "repo_id":      repo_id,
                    "mode":         mode,
                    "signal_family": "ALL",
                    "first_method_signal_method": "skipped_too_large",
                    "mining_status":              "skipped_too_large",
                    "mining_error":               f"first pickaxe timeout on '{kw}'",
                }]
            _first_pickaxe_done = True
            if rc == 124:
                had_error = True
                err_msg = f"pickaxe timeout on '{kw}'"
                continue
            if rc != 0:
                had_error = True
                err_msg = (err or "").strip()[:200]
                continue
            lines = [l for l in out.strip().split("\n") if l]
            if not lines:
                continue
            try:
                commit, date = lines[0].split(",", 1)
            except ValueError:
                continue
            if earliest_date is None or date < earliest_date:
                earliest_date = date
                earliest_commit = commit
                matched_kw = kw

        if earliest_commit:
            out_records.append({
                "repo_id":                    repo_id,
                "mode":                       mode,
                "signal_family":              family,
                "first_method_signal_commit": earliest_commit,
                "first_method_signal_date":   earliest_date,
                "matched_keyword":            matched_kw,
                "first_method_signal_method": "pickaxe_exact",
                "mining_status":              "success",
            })
            continue

        # head_fallback: any HEAD file still contains the keyword?
        head_kw = None
        for kw in keywords:
            rc, out, _ = run_git(
                ["git", "grep", "-i", "-l", "-F", kw, "HEAD"],
                clone_path, HEAD_FALLBACK_TIMEOUT_S,
            )
            if rc == 0 and out.strip():
                head_kw = kw
                break
        if head_kw:
            rc, out, _ = run_git(
                ["git", "log", "-1", "--format=%H,%cI", "HEAD"], clone_path, 30,
            )
            try:
                commit, date = out.strip().split(",", 1)
                out_records.append({
                    "repo_id":                    repo_id,
                    "mode":                       mode,
                    "signal_family":              family,
                    "first_method_signal_commit": commit,
                    "first_method_signal_date":   date,
                    "matched_keyword":            head_kw,
                    "first_method_signal_method": "head_fallback",
                    "mining_status":              "success",
                })
                continue
            except Exception:
                pass

        rec = {
            "repo_id":                    repo_id,
            "mode":                       mode,
            "signal_family":              family,
            "first_method_signal_method": "not_found_in_history",
            "mining_status":              "success" if not had_error else "other_error",
        }
        if had_error:
            rec["mining_error"] = err_msg
        out_records.append(rec)

    # Path-based: Ollama-Modelfile (file presence at HEAD + earliest-add commit)
    rc, ls_out, _ = run_git(["git", "ls-files"], clone_path, MODELFILE_LOOKUP_TIMEOUT_S)
    modelfile_paths = []
    if rc == 0:
        for p in ls_out.split("\n"):
            p = p.strip()
            if p and MODELFILE_NAME_RE.search(p):
                modelfile_paths.append(p)

    if not modelfile_paths:
        out_records.append({
            "repo_id":                    repo_id,
            "mode":                       mode,
            "signal_family":              PATH_SIGNAL_FAMILY,
            "first_method_signal_method": "not_found_in_history",
            "mining_status":              "success",
        })
    else:
        earliest_commit = None
        earliest_date = None
        matched_path = None
        for path in modelfile_paths:
            rc, out, _ = run_git(
                ["git", "log", "--reverse", "--diff-filter=A",
                 "--format=%H,%cI", "--", path],
                clone_path, PICKAXE_TIMEOUT_S,
            )
            if rc != 0:
                continue
            line = out.strip().split("\n")[0] if out else ""
            if not line or "," not in line:
                continue
            commit, date = line.split(",", 1)
            if earliest_date is None or date < earliest_date:
                earliest_date = date
                earliest_commit = commit
                matched_path = path
        if earliest_commit:
            out_records.append({
                "repo_id":                    repo_id,
                "mode":                       mode,
                "signal_family":              PATH_SIGNAL_FAMILY,
                "first_method_signal_commit": earliest_commit,
                "first_method_signal_date":   earliest_date,
                "matched_path":               matched_path,
                "first_method_signal_method": "path_first_add",
                "mining_status":              "success",
            })
        else:
            out_records.append({
                "repo_id":                    repo_id,
                "mode":                       mode,
                "signal_family":              PATH_SIGNAL_FAMILY,
                "first_method_signal_method": "not_found_in_history",
                "mining_status":              "success",
            })
    return out_records


# ---------------------------------------------------------------------------
# pulled_model_id extraction (Ollama-obtain repos only)
# ---------------------------------------------------------------------------

def extract_pulled_model(clone_path: Path, commit: str) -> tuple[str, str]:
    """Return (model_id, source) extracted from the diff added at `commit`.
    Returns ("", "") if nothing recognisable is found."""
    rc, out, _ = run_git(
        ["git", "show", "--unified=0", "--no-color", commit],
        clone_path, MODEL_NAME_EXTRACT_TIMEOUT_S,
    )
    if rc != 0 or not out:
        return "", ""
    # Restrict to added lines (those starting with '+', excluding diff headers '+++')
    added = "\n".join(
        line[1:] for line in out.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    )
    for name, rgx in PULLED_MODEL_PATTERNS:
        m = rgx.search(added)
        if m:
            return m.group(1).rstrip(",;)").strip(), name
    return "", ""


# ---------------------------------------------------------------------------
# Disk cleanup (aggressive default)
# ---------------------------------------------------------------------------

def cleanup_repo(clone_path: Path, strategy: str) -> None:
    if not clone_path.exists():
        return
    if strategy == "aggressive":
        shutil.rmtree(clone_path, ignore_errors=True)
    elif strategy == "conservative":
        pack_dir = clone_path / ".git" / "objects" / "pack"
        if pack_dir.exists():
            shutil.rmtree(pack_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-repo work
# ---------------------------------------------------------------------------

def process_repo(repo_id: str, mode: str, disk_strategy: str,
                 signal_lock: threading.Lock,
                 clone_lock: threading.Lock) -> dict:
    safe = safe_name(repo_id)
    clone_path = CLONES_DIR / safe

    clone_rec = clone_one(repo_id, mode)
    fsync_append_jsonl(CLONE_LOG, [clone_rec], clone_lock)

    if clone_rec["clone_status"] != "success":
        return {"clone_status": clone_rec["clone_status"], "signal_count": 0}

    try:
        records = mine_repo(repo_id, mode, clone_path)
    except Exception as e:
        records = [{
            "repo_id":                    repo_id,
            "mode":                       mode,
            "signal_family":              "ALL",
            "mining_status":              "other_error",
            "mining_error":               f"{type(e).__name__}: {e}",
            "first_method_signal_method": "not_found_in_history",
        }]

    # pulled_model_id: only for ollama_obtains_model mode + only on the
    # repo's overall earliest signal commit if it's from an Ollama family.
    if mode == "ollama_obtains_model":
        ollama_families = {"Ollama-pull", "Ollama-gguf-ref", "Ollama-Modelfile"}
        ollama_records_with_date = [
            r for r in records
            if r.get("signal_family") in ollama_families
            and r.get("first_method_signal_date")
        ]
        if ollama_records_with_date:
            earliest = min(ollama_records_with_date,
                           key=lambda r: r["first_method_signal_date"])
            commit = earliest.get("first_method_signal_commit", "")
            if commit:
                try:
                    pulled, source = extract_pulled_model(clone_path, commit)
                except Exception:
                    pulled, source = "", ""
                if pulled:
                    earliest["pulled_model_id"] = pulled
                    earliest["pulled_model_source"] = source

    fsync_append_jsonl(SIGNALS_JSONL, records, signal_lock)
    cleanup_repo(clone_path, disk_strategy)

    return {"clone_status": "success", "signal_count": len(records)}


# ---------------------------------------------------------------------------
# Inputs: repo list + mode classification
# ---------------------------------------------------------------------------

def load_repos_with_mode() -> list[tuple[str, str]]:
    """Read the obtain-or-produce repo list and re-derive mode from missed_repos.jsonl
    using analyze_offhub.py's priority rule. Returns sorted list of (repo_id, mode)."""
    if not REPO_LIST_TXT.exists():
        sys.exit(f"ERROR: repo list missing: {REPO_LIST_TXT}")
    if not MISSED_JSONL.exists():
        sys.exit(f"ERROR: missed_repos.jsonl missing: {MISSED_JSONL}")

    repos = []
    with REPO_LIST_TXT.open() as f:
        for line in f:
            line = line.strip()
            if line:
                repos.append(line)
    print(f"[init] {len(repos):,} repo IDs in offhub_obtain_repo_ids.txt", flush=True)

    mode_by_repo: dict[str, str] = {}
    with MISSED_JSONL.open() as f:
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
            m = classify_mode(r)
            if m in ("self_quantized", "ollama_obtains_model"):
                mode_by_repo[fn] = m

    out = []
    n_missing = 0
    for rid in repos:
        m = mode_by_repo.get(rid)
        if m is None:
            n_missing += 1
            continue
        out.append((rid, m))
    if n_missing:
        print(f"[init] WARNING: {n_missing} repo IDs absent from missed_repos.jsonl; skipping",
              flush=True)

    out.sort()
    n_self = sum(1 for _, m in out if m == "self_quantized")
    n_obt  = sum(1 for _, m in out if m == "ollama_obtains_model")
    print(f"[init] mode split: self_quantized={n_self:,}  "
          f"ollama_obtains_model={n_obt:,}  total={len(out):,}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def write_checkpoint(state: dict) -> None:
    state["timestamp"] = utcnow()
    with CHECKPOINT.open("w") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def load_checkpoint() -> dict | None:
    if not CHECKPOINT.exists():
        return None
    try:
        return json.loads(CHECKPOINT.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Final CSV: collapse to one row per repo with earliest signal
# ---------------------------------------------------------------------------

def build_final_csv() -> None:
    """Read offhub_signal_first_dates.jsonl, pick the EARLIEST signal per repo
    across all families, and emit offhub_temporal_first_adoption.csv."""
    by_repo: dict[str, dict] = {}
    if not SIGNALS_JSONL.exists():
        print(f"[csv] {SIGNALS_JSONL.name} missing -- skipping CSV build")
        return
    with SIGNALS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rid = r.get("repo_id")
            date = r.get("first_method_signal_date") or ""
            if not rid or not date:
                continue
            family = r.get("signal_family", "")
            cur = by_repo.get(rid)
            if cur is None or date < cur["first_method_signal_date"]:
                by_repo[rid] = r

    rows = []
    for rid, rec in by_repo.items():
        fam = rec.get("signal_family", "")
        rows.append({
            "repo_id":          rid,
            "mode":             rec.get("mode", ""),
            "first_commit_date": rec.get("first_method_signal_date", ""),
            "fired_signal":     rec.get("matched_keyword") or rec.get("matched_path", ""),
            "method_family":    FAMILY_GROUP.get(fam, fam),
            "signal_family":    fam,
            "detection_method": rec.get("first_method_signal_method", ""),
            "pulled_model_id":  rec.get("pulled_model_id", ""),
            "pulled_model_source": rec.get("pulled_model_source", ""),
        })
    rows.sort(key=lambda r: r["repo_id"])
    cols = ["repo_id", "mode", "first_commit_date", "fired_signal",
            "method_family", "signal_family", "detection_method",
            "pulled_model_id", "pulled_model_source"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] wrote {OUT_CSV}  ({len(rows):,} repos)")


# ---------------------------------------------------------------------------
# Main: batched orchestration (mirrors hub mine_full.py)
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disk", choices=["aggressive", "conservative"], default="aggressive")
    ap.add_argument("--batches", type=int, default=0,
                    help="If >0, stop after this many batches from the resume point.")
    ap.add_argument("--start-batch", type=int, default=-1,
                    help="If >=0, force start at this batch index (caution).")
    ap.add_argument("--csv-only", action="store_true",
                    help="Skip mining; just rebuild offhub_temporal_first_adoption.csv "
                         "from the existing signals JSONL and exit.")
    args = ap.parse_args()

    if args.csv_only:
        build_final_csv()
        return

    pairs = load_repos_with_mode()
    if not pairs:
        sys.exit("ERROR: no repos to mine")

    # Build batches (sorted by repo_id from load_repos_with_mode)
    batches: list[list[tuple[str, str]]] = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batches.append(pairs[i:i + BATCH_SIZE])
    print(f"[init] {len(batches)} batches of up to {BATCH_SIZE:,}", flush=True)

    # Resume
    cp = load_checkpoint()
    if args.start_batch >= 0:
        start = args.start_batch
        print(f"[init] FORCED start_batch={start}", flush=True)
    elif cp:
        start = cp.get("last_batch_completed", -1) + 1
        print(f"[init] RESUMING — last_batch_completed={cp.get('last_batch_completed')}, "
              f"starting at batch {start}", flush=True)
    else:
        start = 0
        print(f"[init] starting fresh at batch 0", flush=True)

    if start >= len(batches):
        print("[init] all batches already done; building final CSV", flush=True)
        build_final_csv()
        return

    signal_lock = threading.Lock()
    clone_lock = threading.Lock()

    cum_clones = cp.get("cumulative_clones", 0) if cp else 0
    cum_signals = cp.get("cumulative_signal_records", 0) if cp else 0
    batch_wallclock_seconds: list[float] = []

    end_batch = len(batches) if args.batches <= 0 else min(len(batches), start + args.batches)

    for bi in range(start, end_batch):
        batch = batches[bi]

        # Skip already-done repos (resume safety net)
        done = load_done_repo_ids(SIGNALS_JSONL)
        todo = [(rid, m) for (rid, m) in batch if rid not in done]

        batch_log = LOGS_DIR / f"batch_{bi:02d}.log"
        with batch_log.open("a") as f:
            f.write(f"=== batch {bi} start {utcnow()} ===\n")
            f.write(f"size={len(batch)} todo={len(todo)} disk={args.disk}\n")

        print(f"[batch {bi}/{len(batches)-1}] size={len(batch)} todo={len(todo)}",
              flush=True)
        if not todo:
            write_checkpoint({
                "last_batch_completed":      bi,
                "last_repo_completed":       batch[-1][0],
                "cumulative_clones":         cum_clones,
                "cumulative_signal_records": cum_signals,
                "disk_strategy":             args.disk,
            })
            continue

        batch_start = time.time()
        clone_success_count = 0
        signals_emitted = 0
        per_repo_done = 0

        with ThreadPoolExecutor(max_workers=CLONE_WORKERS) as ex:
            futures = {
                ex.submit(process_repo, rid, m, args.disk, signal_lock, clone_lock): rid
                for rid, m in todo
            }
            for f in as_completed(futures):
                rid = futures[f]
                try:
                    result = f.result()
                except Exception as e:
                    result = {"clone_status": "other_error",
                              "signal_count": 0,
                              "error": f"{type(e).__name__}: {e}"}
                if result["clone_status"] == "success":
                    clone_success_count += 1
                signals_emitted += result.get("signal_count", 0)
                per_repo_done += 1
                if per_repo_done % 50 == 0:
                    print(f"  [batch {bi}] {per_repo_done}/{len(todo)} "
                          f"clones_ok={clone_success_count}", flush=True)

        batch_elapsed = time.time() - batch_start
        batch_wallclock_seconds.append(batch_elapsed)
        cum_clones += clone_success_count
        cum_signals += signals_emitted
        clone_rate = clone_success_count / max(1, len(todo))

        with batch_log.open("a") as f:
            f.write(f"=== batch {bi} done {utcnow()} ===\n")
            f.write(f"todo={len(todo)} success={clone_success_count} "
                    f"rate={clone_rate:.3f}\n")
            f.write(f"signals_emitted={signals_emitted}\n")
            f.write(f"wallclock_seconds={batch_elapsed:.0f}\n")
            f.write(f"per_repo_avg_seconds={batch_elapsed / max(1, len(todo)):.2f}\n")

        write_checkpoint({
            "last_batch_completed":          bi,
            "last_repo_completed":           batch[-1][0],
            "cumulative_clones":             cum_clones,
            "cumulative_signal_records":     cum_signals,
            "disk_strategy":                 args.disk,
            "last_batch_wallclock_seconds":  batch_elapsed,
            "last_batch_clone_rate":         clone_rate,
        })

        print(f"[batch {bi} done] {batch_elapsed:.0f}s  "
              f"clones={clone_success_count}/{len(todo)} ({clone_rate*100:.1f}%)  "
              f"signals={signals_emitted}  "
              f"cum: clones={cum_clones} signals={cum_signals}",
              flush=True)

        # ----- Guardrails (mirror hub) -----
        used_tb = disk_used_tb()
        if used_tb > DISK_LIMIT_TB:
            print(f"GUARDRAIL: disk used {used_tb:.1f} TB > {DISK_LIMIT_TB} TB. PAUSING.",
                  flush=True)
            return

        if clone_rate < MIN_BATCH_CLONE_RATE:
            print(f"GUARDRAIL: batch {bi} clone rate {clone_rate*100:.1f}% < "
                  f"{MIN_BATCH_CLONE_RATE*100:.0f}%. PAUSING.", flush=True)
            return

        if len(batch_wallclock_seconds) >= GUARDRAIL_BATCHES_FOR_PROJECTION:
            avg_batch = sum(batch_wallclock_seconds[-GUARDRAIL_BATCHES_FOR_PROJECTION:]) \
                / GUARDRAIL_BATCHES_FOR_PROJECTION
            remaining_batches = len(batches) - (bi + 1)
            projected_hours = avg_batch * remaining_batches / 3600 + \
                sum(batch_wallclock_seconds) / 3600
            if projected_hours > MAX_PROJECTED_HOURS:
                print(f"GUARDRAIL: projected total wall-clock {projected_hours:.1f}h > "
                      f"{MAX_PROJECTED_HOURS}h after {bi+1} batches. PAUSING.",
                      flush=True)
                return

    print(f"[done] all batches complete. clones={cum_clones} signals={cum_signals}",
          flush=True)

    # Build the final CSV once all batches are in.
    build_final_csv()


if __name__ == "__main__":
    main()

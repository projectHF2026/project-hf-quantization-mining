"""
RQ2 Phase 1 pilot — orchestrated clone + Layer 1 + Layer 2 mining over the
100 pilot repos. Resumable: each stage skips repos that already appear in its
output JSONL with a terminal status.

Usage:
  python3 mine_pilot.py --stage clone     # clone 100 repos
  python3 mine_pilot.py --stage layer1    # method-level pickaxe
  python3 mine_pilot.py --stage layer2    # model-level pickaxe per matched_file
  python3 mine_pilot.py --stage all       # run all three in sequence
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path("/scratch/oldhome/user/projects/JAW/scripts/icpc-approch")
PILOT_DIR = BASE / "output_dir/rq_analysis/rq2/phase1_pilot"
PILOT_CSV = PILOT_DIR / "pilot_repos.csv"
CLONES_DIR = PILOT_DIR / "clones"
CLONE_LOG = PILOT_DIR / "pilot_clone_log.jsonl"
RESULTS_DIR = PILOT_DIR / "results"
LAYER1_JSONL = RESULTS_DIR / "layer1_method_signals.jsonl"
LAYER2_JSONL = RESULTS_DIR / "layer2_model_signals.jsonl"
ANALYSIS_JSONL = BASE / "output_dir/rq_analysis/shared/results/analysis_set_repo_details.jsonl"
ALL_MODELS_JSONL = BASE / "output_dir/quantized_filtered/quantized_models_all.jsonl"

CLONES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Tunables ---
CLONE_TIMEOUT = 600        # 10 min per repo
CLONE_MAX_WORKERS = 8
PICKAXE_TIMEOUT = 300      # 5 min per git log invocation
MINE_MAX_WORKERS = 4

# --- Canonical method keywords (case-insensitive) ---
METHOD_KEYWORDS: dict[str, list[str]] = {
    "GGUF":         ["gguf", "ggml", "q4_k_m", "q5_0", "q8_0", "q3_k_s"],
    "BitsAndBytes": ["bitsandbytes", "bnb", "load_in_4bit", "load_in_8bit", "nf4"],
    "GPTQ":         ["gptq", "autogptq", "gptqmodel"],
    "AWQ":          ["awq", "autoawq"],
    "Other":        ["fp8", "mxfp4", "aqlm", "hqq", "eetq", "vptq", "compressedtensors",
                     "torchao", "exl2", "marlin", "spqr", "fbgemm-fp8", "nvfp4",
                     "autoround", "bitnet", "quark", "higgs"],
}


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def run_git(args: list[str], cwd: Path, timeout: int = 60) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, errors="replace",
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"


def load_done(jsonl: Path, key_fn) -> set:
    done = set()
    if not jsonl.exists():
        return done
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                k = key_fn(r)
                if k is not None:
                    done.add(k)
            except Exception:
                pass
    return done


def append_jsonl(path: Path, records: list[dict], lock: threading.Lock) -> None:
    if not records:
        return
    with lock:
        with path.open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Stage 1: clone
# ---------------------------------------------------------------------------


def clone_one(repo_id: str, stratum: str) -> dict:
    safe = safe_name(repo_id)
    clone_path = CLONES_DIR / safe
    url = f"https://github.com/{repo_id}.git"

    if clone_path.exists():
        shutil.rmtree(clone_path, ignore_errors=True)

    t0 = time.time()
    try:
        proc = subprocess.run(
            ["git", "clone", "--quiet", "--no-tags", url, str(clone_path)],
            capture_output=True, text=True, timeout=CLONE_TIMEOUT, errors="replace",
        )
    except subprocess.TimeoutExpired:
        if clone_path.exists():
            shutil.rmtree(clone_path, ignore_errors=True)
        return {
            "repo_id": repo_id, "stratum": stratum,
            "clone_status": "timeout",
            "clone_error": f"timed out after {CLONE_TIMEOUT}s",
            "clone_seconds": time.time() - t0,
        }
    elapsed = time.time() - t0

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()[:500]
        low = err.lower()
        if "not found" in low or "could not read" in low or "repository not found" in low:
            status = "404"
        elif "authentication" in low or "could not read username" in low:
            status = "auth_required"
        else:
            status = "other_error"
        if clone_path.exists():
            shutil.rmtree(clone_path, ignore_errors=True)
        return {
            "repo_id": repo_id, "stratum": stratum,
            "clone_status": status,
            "clone_error": err,
            "clone_seconds": elapsed,
        }

    # Success — collect metadata
    rc, out, _ = run_git(["git", "symbolic-ref", "--short", "HEAD"], clone_path, 30)
    default_branch = out.strip() if rc == 0 else ""

    rc, out, _ = run_git(["git", "rev-list", "--count", "HEAD"], clone_path, 60)
    try:
        commit_count = int(out.strip())
    except Exception:
        commit_count = 0

    rc, out, _ = run_git(
        ["git", "log", "--reverse", "--format=%cI", "HEAD"], clone_path, 60
    )
    first_commit_date = out.strip().split("\n")[0] if rc == 0 and out.strip() else ""

    try:
        size_proc = subprocess.run(
            ["du", "-sb", str(clone_path)],
            capture_output=True, text=True, timeout=60,
        )
        clone_size = int(size_proc.stdout.split()[0])
    except Exception:
        clone_size = 0

    return {
        "repo_id": repo_id,
        "stratum": stratum,
        "clone_status": "success",
        "default_branch": default_branch,
        "default_branch_commit_count": commit_count,
        "default_branch_first_commit_date": first_commit_date,
        "clone_seconds": elapsed,
        "clone_size_bytes": clone_size,
    }


def stage_clone() -> None:
    pilot = pd.read_csv(PILOT_CSV)
    done = load_done(CLONE_LOG, lambda r: r.get("repo_id"))
    todo = [(r["repo_id"], r["stratum"]) for _, r in pilot.iterrows()
            if r["repo_id"] not in done]
    print(f"[clone] pilot={len(pilot)}  done={len(done)}  todo={len(todo)}", flush=True)
    if not todo:
        return

    lock = threading.Lock()
    n = len(done)
    start = time.time()
    with ThreadPoolExecutor(max_workers=CLONE_MAX_WORKERS) as ex:
        futures = {ex.submit(clone_one, rid, strat): (rid, strat) for rid, strat in todo}
        for f in as_completed(futures):
            rid, strat = futures[f]
            try:
                rec = f.result()
            except Exception as e:
                rec = {"repo_id": rid, "stratum": strat,
                       "clone_status": "other_error",
                       "clone_error": f"{type(e).__name__}: {e}"}
            append_jsonl(CLONE_LOG, [rec], lock)
            n += 1
            print(
                f"[clone {n:>3}/{len(pilot)}] {rid} ({strat}): "
                f"{rec['clone_status']:<14} {rec.get('clone_seconds', 0):>5.1f}s "
                f"{rec.get('default_branch_commit_count', '?')} commits",
                flush=True,
            )
    print(f"[clone] elapsed: {(time.time()-start):.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage 2A: Layer 1 (method-level)
# ---------------------------------------------------------------------------


def mine_layer1_one(repo_id: str, stratum: str, clone_path: Path) -> list[dict]:
    out_records: list[dict] = []
    for method, keywords in METHOD_KEYWORDS.items():
        earliest_commit: str | None = None
        earliest_date: str | None = None
        matched_kw: str | None = None
        had_error = False
        err_msg = ""

        for kw in keywords:
            rc, out, err = run_git(
                ["git", "log", "--reverse", "-S", kw, "-i", "--format=%H,%cI"],
                clone_path, PICKAXE_TIMEOUT,
            )
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
                "repo_id": repo_id, "stratum": stratum, "method": method,
                "first_method_signal_commit": earliest_commit,
                "first_method_signal_date": earliest_date,
                "matched_keyword": matched_kw,
                "first_method_signal_method": "pickaxe_exact",
                "mining_status": "success",
            })
            continue

        # head_fallback: any keyword present in HEAD content?
        head_kw = None
        for kw in keywords:
            rc, out, _ = run_git(
                ["git", "grep", "-i", "-l", "-F", kw, "HEAD"],
                clone_path, PICKAXE_TIMEOUT,
            )
            if rc == 124:
                continue
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
                    "repo_id": repo_id, "stratum": stratum, "method": method,
                    "first_method_signal_commit": commit,
                    "first_method_signal_date": date,
                    "matched_keyword": head_kw,
                    "first_method_signal_method": "head_fallback",
                    "mining_status": "success",
                })
                continue
            except Exception:
                pass

        rec = {
            "repo_id": repo_id, "stratum": stratum, "method": method,
            "first_method_signal_method": "not_found_in_history",
            "mining_status": "success" if not had_error else "other_error",
        }
        if had_error:
            rec["mining_error"] = err_msg
        out_records.append(rec)
    return out_records


def stage_layer1() -> None:
    # Build success-cloned repo list
    clones: dict[str, dict] = {}
    if CLONE_LOG.exists():
        with CLONE_LOG.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("clone_status") == "success":
                    clones[r["repo_id"]] = r

    done_repos = load_done(LAYER1_JSONL, lambda r: r.get("repo_id"))
    todo = [(rid, rec["stratum"]) for rid, rec in clones.items() if rid not in done_repos]
    print(f"[layer1] cloned={len(clones)}  done={len(done_repos)}  todo={len(todo)}",
          flush=True)
    if not todo:
        return

    lock = threading.Lock()
    n = len(done_repos)
    start = time.time()
    with ThreadPoolExecutor(max_workers=MINE_MAX_WORKERS) as ex:
        futures = {
            ex.submit(mine_layer1_one, rid, strat, CLONES_DIR / safe_name(rid)):
                (rid, strat)
            for rid, strat in todo
        }
        for f in as_completed(futures):
            rid, strat = futures[f]
            try:
                recs = f.result()
            except Exception as e:
                recs = [{
                    "repo_id": rid, "stratum": strat, "method": m,
                    "mining_status": "other_error",
                    "mining_error": f"{type(e).__name__}: {e}",
                    "first_method_signal_method": "not_found_in_history",
                } for m in METHOD_KEYWORDS]
            append_jsonl(LAYER1_JSONL, recs, lock)
            n += 1
            found = sum(1 for r in recs if r.get("first_method_signal_method") == "pickaxe_exact")
            print(f"[layer1 {n:>3}/{len(clones)}] {rid} ({strat}): "
                  f"found {found}/{len(METHOD_KEYWORDS)} methods via pickaxe",
                  flush=True)
    print(f"[layer1] elapsed: {(time.time()-start):.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage 2B: Layer 2 (model-level)
# ---------------------------------------------------------------------------


def load_pilot_loader_evidence(pilot_ids: set[str]) -> dict[str, dict]:
    """Return repo_id -> dict with primary_method and list of (model_id, file) pairs
    from loader_evidence."""
    out: dict[str, dict] = {}
    with ANALYSIS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["repo"] not in pilot_ids:
                continue
            evidence = r.get("loader_evidence") or []
            pairs = []
            seen = set()
            for ev in evidence:
                mid = ev.get("model_id")
                fp = ev.get("file")
                if not mid or not fp:
                    continue
                key = (mid, fp)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"model_id": mid, "matched_file": fp})
            out[r["repo"]] = {"pairs": pairs}
    return out


def load_hf_created_at(model_ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    with ALL_MODELS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            mid = r.get("model_id")
            if mid in model_ids:
                ca = r.get("created_at") or ""
                out[mid] = ca
    return out


def compute_lag_days(repo_date: str, hf_date: str) -> float | None:
    try:
        rd = datetime.fromisoformat(repo_date.replace("Z", "+00:00"))
        hd = datetime.fromisoformat(hf_date.replace("Z", "+00:00"))
        return (rd - hd).total_seconds() / 86400.0
    except Exception:
        return None


def mine_layer2_one(repo_id: str, stratum: str, primary_method: str,
                    clone_path: Path, pairs: list[dict],
                    hf_meta: dict[str, str]) -> list[dict]:
    records: list[dict] = []
    for pair in pairs:
        mid = pair["model_id"]
        mf = pair["matched_file"]
        hf_ca = hf_meta.get(mid, "")
        rec = {
            "repo_id": repo_id, "stratum": stratum,
            "model_id": mid, "matched_file": mf,
            "primary_method": primary_method,
            "hf_createdAt": hf_ca,
        }

        # File on default branch?
        rc, _, _ = run_git(
            ["git", "cat-file", "-e", f"HEAD:{mf}"], clone_path, 15,
        )
        if rc != 0:
            rec.update({
                "first_model_signal_method": "not_found_in_history",
                "mining_status": "file_missing_on_default",
            })
            records.append(rec)
            continue

        # Pickaxe with --follow on the specific file
        rc, out, err = run_git(
            ["git", "log", "--follow", "--reverse", "-S", mid,
             "--format=%H,%cI", "--", mf],
            clone_path, PICKAXE_TIMEOUT,
        )
        if rc == 124:
            rec.update({
                "first_model_signal_method": "not_found_in_history",
                "mining_status": "other_error",
                "mining_error": "pickaxe timeout",
            })
            records.append(rec)
            continue

        lines = [l for l in out.strip().split("\n") if l] if rc == 0 else []
        if lines:
            try:
                commit, date = lines[0].split(",", 1)
            except ValueError:
                commit, date = None, None
            if commit and date:
                rec.update({
                    "first_model_signal_commit": commit,
                    "first_model_signal_date": date,
                    "first_model_signal_method": "pickaxe_exact",
                    "mining_status": "success",
                })
                if hf_ca:
                    lag = compute_lag_days(date, hf_ca)
                    if lag is not None:
                        rec["adoption_lag_days"] = lag
                records.append(rec)
                continue

        # head_fallback
        rc, out, _ = run_git(
            ["git", "grep", "-l", "-F", mid, "HEAD", "--", mf],
            clone_path, PICKAXE_TIMEOUT,
        )
        if rc == 0 and out.strip():
            rc2, out2, _ = run_git(
                ["git", "log", "-1", "--format=%H,%cI", "HEAD"], clone_path, 30,
            )
            if rc2 == 0 and out2.strip():
                try:
                    commit, date = out2.strip().split(",", 1)
                    rec.update({
                        "first_model_signal_commit": commit,
                        "first_model_signal_date": date,
                        "first_model_signal_method": "head_fallback",
                        "mining_status": "success",
                    })
                    records.append(rec)
                    continue
                except Exception:
                    pass

        rec.update({
            "first_model_signal_method": "not_found_in_history",
            "mining_status": "success",
        })
        records.append(rec)
    return records


def stage_layer2() -> None:
    pilot = pd.read_csv(PILOT_CSV)
    pilot_by_repo = {r["repo_id"]: r for _, r in pilot.iterrows()}

    # Successfully cloned repos
    clones: dict[str, dict] = {}
    if CLONE_LOG.exists():
        with CLONE_LOG.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("clone_status") == "success" and r["repo_id"] in pilot_by_repo:
                    clones[r["repo_id"]] = r

    print(f"[layer2] loading pilot loader_evidence ...", flush=True)
    pilot_evidence = load_pilot_loader_evidence(set(clones.keys()))
    n_total_pairs = sum(len(v["pairs"]) for v in pilot_evidence.values())
    print(f"[layer2] {n_total_pairs} (model_id, matched_file) pairs across "
          f"{len(pilot_evidence)} cloned repos", flush=True)

    # Pre-load HF createdAt for all model_ids in pairs
    all_mids: set[str] = set()
    for v in pilot_evidence.values():
        for p in v["pairs"]:
            all_mids.add(p["model_id"])
    print(f"[layer2] loading HF createdAt for {len(all_mids)} unique models ...",
          flush=True)
    hf_meta = load_hf_created_at(all_mids)
    print(f"[layer2] resolved createdAt for {len(hf_meta)}/{len(all_mids)} models",
          flush=True)

    done_repos = load_done(LAYER2_JSONL, lambda r: r.get("repo_id"))
    todo = [(rid, clones[rid]["stratum"], pilot_by_repo[rid]["primary_method"])
            for rid in clones if rid not in done_repos]
    print(f"[layer2] done={len(done_repos)}  todo={len(todo)}", flush=True)

    if not todo:
        return

    lock = threading.Lock()
    n = len(done_repos)
    start = time.time()
    with ThreadPoolExecutor(max_workers=MINE_MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                mine_layer2_one, rid, strat, pri,
                CLONES_DIR / safe_name(rid),
                pilot_evidence.get(rid, {"pairs": []})["pairs"],
                hf_meta,
            ): (rid, strat)
            for rid, strat, pri in todo
        }
        for f in as_completed(futures):
            rid, strat = futures[f]
            try:
                recs = f.result()
            except Exception as e:
                recs = [{
                    "repo_id": rid, "stratum": strat,
                    "mining_status": "other_error",
                    "mining_error": f"{type(e).__name__}: {e}",
                    "first_model_signal_method": "not_found_in_history",
                }]
            append_jsonl(LAYER2_JSONL, recs, lock)
            n += 1
            n_exact = sum(1 for r in recs if r.get("first_model_signal_method") == "pickaxe_exact")
            print(f"[layer2 {n:>3}/{len(clones)}] {rid} ({strat}): "
                  f"{len(recs)} pairs, {n_exact} pickaxe_exact", flush=True)
    print(f"[layer2] elapsed: {(time.time()-start):.0f}s", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", required=True,
        choices=["clone", "layer1", "layer2", "all"],
    )
    args = ap.parse_args()

    if args.stage in ("clone", "all"):
        stage_clone()
    if args.stage in ("layer1", "all"):
        stage_layer1()
    if args.stage in ("layer2", "all"):
        stage_layer2()


if __name__ == "__main__":
    main()

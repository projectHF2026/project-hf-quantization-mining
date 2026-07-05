#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_usage_repos_with_loader_confirmation.py
=============================================
1) Runs your existing Tier A/B/C/D classification based on matched file TYPES.
2) Adds loader-pattern confirmation (Tier A2) by fetching and scanning matched
   code/config files from GitHub for loader evidence.

Loader confirmation is "practical + defensible":
- Requires BOTH:
  (a) the exact model_id string appears in the file content, AND
  (b) at least one loader pattern appears in the same file content.
This reduces false positives from catalogs or unrelated code.

Notebook caveat:
- .ipynb is Tier A by file-type, but becomes Tier A2 only if loader patterns
  are found in *code cells* ("cell_type":"code") and the model_id string is in those code cells.

Inputs:
  - repo_details.csv (repo,num_models,num_files,models,files)
  - GitHub tokens via GH_TOKEN_1..GH_TOKEN_N or GITHUB_TOKEN

Outputs:
  - Original Tier A/B/C/D files
  - Tier A2 loader-confirmed files + evidence
  - loader_stats.json

Usage:
  python3 filter_usage_repos_with_loader_confirmation.py \
    --repo-details /path/to/final_data/repo_details.csv \
    --output-dir /path/to/final_data/usage_filtered_loader \
    --max-files-per-repo 25 \
    --workers 8
"""

import os
import re
import csv
import json
import time
import base64
import argparse
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import requests

# ============================================================================
# 0) FILE-TYPE TIERS (your existing logic, unchanged)
# ============================================================================

CODE_EXTENSIONS = {
    ".py", ".pyx", ".pyi",
    ".ipynb",
    ".sh", ".bash", ".zsh",
    ".dockerfile",
    ".ps1", ".bat", ".cmd",
    ".go", ".rs", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".kt", ".scala", ".gradle",
    ".cpp", ".c", ".h", ".hpp",
    ".swift", ".m",
    ".rb", ".php", ".lua",
    ".r", ".R",
    ".jl",
    ".makefile", ".mk",
    ".nix",
    ".jsonnet",
}

CONFIG_EXTENSIONS = {
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env",
}

CODE_FILENAMES = {
    "dockerfile",
    "makefile",
    "docker-compose.yml",
    "docker-compose.yaml",
}

AMBIGUOUS_FILENAMES = {
    "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
    "pipfile", "gemfile",
}

DOC_EXTENSIONS = {
    ".md", ".rst", ".txt", ".html", ".htm",
    ".csv", ".tsv",
    ".pdf", ".doc", ".docx",
    ".tex", ".bib",
}

DOC_PATH_PATTERNS = [
    re.compile(r'(^|/)docs?/', re.IGNORECASE),
    re.compile(r'(^|/)documentation/', re.IGNORECASE),
    re.compile(r'(^|/)wiki/', re.IGNORECASE),
    re.compile(r'(^|/)papers?/', re.IGNORECASE),
    re.compile(r'(^|/)blog/', re.IGNORECASE),
    re.compile(r'(^|/)examples?/.*\.md$', re.IGNORECASE),
    re.compile(r'(^|/)(model[-_]?zoo|model[-_]?list|model[-_]?map|modelzoo|leaderboard)(s)?(/|$)', re.IGNORECASE),
    re.compile(r'(^|/)gh-pages/', re.IGNORECASE),
    re.compile(r'(^|/)_posts/', re.IGNORECASE),
    re.compile(r'(^|/)site/', re.IGNORECASE),
    re.compile(r'(^|/)docs?/models?/', re.IGNORECASE),
]

USAGE_JSON_FILENAMES = {
    "quantize_config.json", "quant_config.json",
    "quantization_config.json",
}

def classify_file(file_path: str) -> str:
    basename = os.path.basename(file_path).lower()
    _, ext = os.path.splitext(basename)

    for pattern in DOC_PATH_PATTERNS:
        if pattern.search(file_path):
            return "doc"

    if ext in DOC_EXTENSIONS:
        return "doc"

    if basename in CODE_FILENAMES:
        return "code"

    if basename == "dockerfile" or basename.startswith("dockerfile."):
        return "code"

    if basename in AMBIGUOUS_FILENAMES:
        return "ambiguous"

    if ext in CODE_EXTENSIONS:
        return "code"

    if ext in CONFIG_EXTENSIONS:
        return "config"

    if ext == ".json":
        if basename in USAGE_JSON_FILENAMES:
            return "config"
        return "ambiguous"

    return "ambiguous"

def classify_repo(files: List[str]) -> Dict:
    file_class = {f: classify_file(f) for f in files}

    code_files = [f for f, c in file_class.items() if c == "code"]
    config_files = [f for f, c in file_class.items() if c == "config"]
    doc_files = [f for f, c in file_class.items() if c == "doc"]
    ambiguous_files = [f for f, c in file_class.items() if c == "ambiguous"]

    if code_files:
        tier = "A_usage"
    elif config_files:
        tier = "B_config_only"
    elif ambiguous_files and not doc_files:
        tier = "D_ambiguous"
    else:
        tier = "C_mention_only"

    return {
        "tier": tier,
        "code_files": code_files,
        "config_files": config_files,
        "doc_files": doc_files,
        "ambiguous_files": ambiguous_files,
        "file_classifications": file_class,
    }

# ============================================================================
# 1) LOADER-PATTERN CONFIRMATION (Tier A2)
# ============================================================================

# Practical + defensible loader patterns (regex):
# We treat "from_pretrained" as weak unless paired with quant signal or model_id in same file (we enforce model_id presence).
LOADER_REGEXES = {
    # transformers / HF loading
    "transformers_from_pretrained": re.compile(r"\bfrom_pretrained\s*\(", re.IGNORECASE),
    "auto_model_from_pretrained": re.compile(r"\bAutoModelForCausalLM\.from_pretrained\s*\(", re.IGNORECASE),
    "auto_tokenizer_from_pretrained": re.compile(r"\bAutoTokenizer\.from_pretrained\s*\(", re.IGNORECASE),
    "pipeline_model": re.compile(r"\bpipeline\s*\(.*model\s*=", re.IGNORECASE),

    # bnb quant loading (stronger quant signal)
    "bitsandbytes_config": re.compile(r"\bBitsAndBytesConfig\s*\(", re.IGNORECASE),
    "load_in_4bit": re.compile(r"\bload_in_4bit\s*=\s*True\b", re.IGNORECASE),
    "load_in_8bit": re.compile(r"\bload_in_8bit\s*=\s*True\b", re.IGNORECASE),

    # huggingface_hub
    "snapshot_download": re.compile(r"\bsnapshot_download\s*\(", re.IGNORECASE),
    "hf_hub_download": re.compile(r"\bhf_hub_download\s*\(", re.IGNORECASE),

    # quant-specific loaders
    "autogptq_from_quantized": re.compile(r"\bAutoGPTQForCausalLM\.from_quantized\s*\(", re.IGNORECASE),
    "autoawq_from_quantized": re.compile(r"\bAutoAWQForCausalLM\.from_quantized\s*\(", re.IGNORECASE),
    "ctransformers_from_pretrained": re.compile(r"\bctransformers\..*from_pretrained\s*\(", re.IGNORECASE),

    # GGUF / llama.cpp style
    "llama_cpp_llama": re.compile(r"\bllama_cpp\.Llama\s*\(", re.IGNORECASE),
    "import_llama_cpp": re.compile(r"\bfrom\s+llama_cpp\s+import\s+Llama\b|\bimport\s+llama_cpp\b", re.IGNORECASE),
    "gguf_path": re.compile(r"\.gguf\b", re.IGNORECASE),

    # CLI-ish (we keep as supporting)
    "llama_cli": re.compile(r"\bllama(-cli)?\b|\bllama\.cpp\b", re.IGNORECASE),
    "cli_model_flag": re.compile(r"(\s--model\s+|\s-m\s+)", re.IGNORECASE),
}

# Strong “quantization evidence” signals (helps ensure we're not just loading base fp16)
QUANT_EVIDENCE_REGEXES = {
    "bitsandbytes_4bit_8bit": re.compile(r"\b(load_in_4bit|load_in_8bit|BitsAndBytesConfig)\b", re.IGNORECASE),
    "gptq_awq": re.compile(r"\b(AutoGPTQForCausalLM|AutoAWQForCausalLM|gptq|awq)\b", re.IGNORECASE),
    "gguf_llama": re.compile(r"\b(gguf|llama_cpp|llama\.cpp)\b|\.(gguf)\b", re.IGNORECASE),
    "exl2": re.compile(r"\bexllama\b|\bexllamav2\b|\bexl2\b", re.IGNORECASE),
}

def _ipynb_code_cell_text(nb: dict) -> str:
    parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            parts.append("".join(src))
        elif isinstance(src, str):
            parts.append(src)
    return "\n".join(parts)

def detect_loader_evidence(text: str) -> Tuple[List[str], List[str]]:
    """
    Return (matched_loaders, matched_quant_evidence).
    """
    loaders = [name for name, rgx in LOADER_REGEXES.items() if rgx.search(text)]
    qev = [name for name, rgx in QUANT_EVIDENCE_REGEXES.items() if rgx.search(text)]
    return loaders, qev

# ============================================================================
# 2) GITHUB FETCH (contents API)
# ============================================================================

def load_github_tokens() -> List[str]:
    toks = []
    for i in range(1, 50):
        t = os.environ.get(f"GH_TOKEN_{i}", "").strip()
        if t:
            toks.append(t)
    if not toks:
        t = os.environ.get("GITHUB_TOKEN", "").strip()
        if t:
            toks.append(t)
    return toks

TOKENS = load_github_tokens()
if not TOKENS:
    raise RuntimeError("No GitHub tokens found. Export GH_TOKEN_1.. or GITHUB_TOKEN.")

_token_lock = threading.Lock()
_token_idx = 0

def next_token() -> str:
    global _token_idx
    with _token_lock:
        t = TOKENS[_token_idx % len(TOKENS)]
        _token_idx += 1
        return t

_session_local = threading.local()

def get_session(token: str) -> requests.Session:
    s = getattr(_session_local, "session", None)
    if s is None or getattr(_session_local, "token", None) != token:
        s = requests.Session()
        s.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        _session_local.session = s
        _session_local.token = token
    return s

def fetch_github_file_text(repo: str, path: str, timeout: int = 30) -> Optional[str]:
    """
    Fetch file via GitHub contents API.
    Returns decoded text or None.
    """
    token = next_token()
    s = get_session(token)
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    for _ in range(3):
        r = s.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and data.get("encoding") == "base64" and "content" in data:
                raw = base64.b64decode(data["content"])
                try:
                    return raw.decode("utf-8", errors="replace")
                except Exception:
                    return raw.decode(errors="replace")
            return None
        if r.status_code in (403, 429):
            # backoff a bit and rotate token
            time.sleep(2.0)
            token = next_token()
            s = get_session(token)
            continue
        if r.status_code == 404:
            return None
        time.sleep(1.0)
    return None

# ============================================================================
# 3) CONFIRMATION LOGIC
# ============================================================================

def confirm_repo_usage_by_loaders(repo: str, models: List[str], candidate_files: List[str],
                                  max_files: int) -> Dict:
    """
    Confirm usage by scanning up to max_files matched code/config files.

    Rule (defensible):
      A repo is loader-confirmed if ∃ (model_id, file) such that:
        - model_id string appears in the scanned content AND
        - at least one loader regex matches in that same content
      Stronger confidence if quant evidence also appears (bnb/gptq/gguf/etc).

    Returns:
      {
        confirmed: bool,
        evidence: [ {model_id, file, loaders, quant_evidence, notebook: bool}, ... ],
        scanned_files: int,
        skipped_files: int
      }
    """
    evidence = []
    scanned = 0
    skipped = 0

    # Prefer code files over config files
    prioritized = candidate_files[:]
    # keep stable order; you can sort if you want.
    prioritized = prioritized[:max_files]

    for fp in prioritized:
        text = fetch_github_file_text(repo, fp)
        if text is None:
            skipped += 1
            continue

        is_nb = fp.lower().endswith(".ipynb")
        if is_nb:
            try:
                nb = json.loads(text)
                text_for_scan = _ipynb_code_cell_text(nb)
            except Exception:
                # can't parse notebook -> treat as skipped for confirmation
                skipped += 1
                continue
        else:
            text_for_scan = text

        scanned += 1

        # Fast reject: if none of the model ids appear, skip heavy regex
        # (still O(len(models)) but okay; models per repo is usually small except catalogs)
        present_models = [m for m in models if m and (m in text_for_scan)]
        if not present_models:
            continue

        loaders, qev = detect_loader_evidence(text_for_scan)
        if not loaders:
            continue

        # confirm each present model (cap evidence to avoid huge output)
        for mid in present_models[:25]:
            ev = {
                "model_id": mid,
                "file": fp,
                "loaders": loaders,
                "quant_evidence": qev,
                "notebook": is_nb,
            }
            evidence.append(ev)

        # Early stop: once we have some evidence, we can stop (configurable)
        if evidence:
            return {
                "confirmed": True,
                "evidence": evidence,
                "scanned_files": scanned,
                "skipped_files": skipped,
            }

    return {
        "confirmed": False,
        "evidence": evidence,
        "scanned_files": scanned,
        "skipped_files": skipped,
    }

# ============================================================================
# 4) MAIN
# ============================================================================

def write_list(path: str, items: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for x in sorted(items):
            f.write(x + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-details", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-files-per-repo", type=int, default=25,
                    help="Max matched code/config files to fetch per repo for loader confirmation")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv.field_size_limit(50 * 1024 * 1024)

    # Load repo_details.csv
    repos = {}
    with open(args.repo_details, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row["repo"]
            files = row["files"].split(";") if row.get("files") else []
            models = row["models"].split(";") if row.get("models") else []
            repos[repo] = {
                "repo": repo,
                "num_models": int(row["num_models"]),
                "num_files": int(row["num_files"]),
                "models": models,
                "files": files,
            }

    # Stage 1: A/B/C/D by file type
    tierA, tierB, tierC, tierD = {}, {}, {}, {}
    tier_counts = Counter()

    for repo, d in repos.items():
        cls = classify_repo(d["files"])
        tier = cls["tier"]
        tier_counts[tier] += 1
        entry = {
            "repo": repo,
            "num_models": d["num_models"],
            "models": d["models"],
            "code_files": cls["code_files"],
            "config_files": cls["config_files"],
            "doc_files": cls["doc_files"],
            "ambiguous_files": cls["ambiguous_files"],
        }
        if tier == "A_usage":
            tierA[repo] = entry
        elif tier == "B_config_only":
            tierB[repo] = entry
        elif tier == "C_mention_only":
            tierC[repo] = entry
        else:
            tierD[repo] = entry

    # Stage 2: Loader-confirmation on Tier A + (optionally) Tier B
    # (We do Tier A only by default for runtime usage; Tier B can be scanned too if you want.)
    tierA2_confirmed = {}
    tierA2_unconfirmed = {}

    lock = threading.Lock()

    def worker(repo: str, entry: dict) -> Tuple[str, dict]:
        candidate_files = entry["code_files"] + entry["config_files"]
        # If this repo has too many models (catalog-like), confirmation is expensive.
        # We still try, but models per repo can be huge for catalogs. You can cap model checks.
        models = entry["models"][:200]  # cap to keep scanning feasible
        res = confirm_repo_usage_by_loaders(repo, models, candidate_files, args.max_files_per_repo)
        out = dict(entry)
        out["loader_confirmed"] = res["confirmed"]
        out["loader_evidence"] = res["evidence"]
        out["loader_scanned_files"] = res["scanned_files"]
        out["loader_skipped_files"] = res["skipped_files"]
        return repo, out

    tierA_items = list(tierA.items())

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, repo, entry) for repo, entry in tierA_items]
        done = 0
        for fut in as_completed(futures):
            repo, out = fut.result()
            with lock:
                if out["loader_confirmed"]:
                    tierA2_confirmed[repo] = out
                else:
                    tierA2_unconfirmed[repo] = out
                done += 1
                if done % 250 == 0:
                    print(f"  Loader-check progress: {done}/{len(tierA_items)}")

    # Write outputs
    def outp(name: str) -> str:
        return os.path.join(args.output_dir, name)

    # Original tiers lists
    write_list(outp("tierA1_usage_filetype_repos.txt"), list(tierA.keys()))
    write_list(outp("tierB_config_only_repos.txt"), list(tierB.keys()))
    write_list(outp("tierC_mention_only_repos.txt"), list(tierC.keys()))
    write_list(outp("tierD_ambiguous_repos.txt"), list(tierD.keys()))

    # Loader-confirmed
    write_list(outp("tierA2_usage_loader_confirmed_repos.txt"), list(tierA2_confirmed.keys()))
    write_list(outp("tierA2_usage_loader_unconfirmed_repos.txt"), list(tierA2_unconfirmed.keys()))

    # Evidence JSONL
    with open(outp("tierA2_usage_loader_confirmed_repo_details.jsonl"), "w", encoding="utf-8") as f:
        for repo in sorted(tierA2_confirmed.keys()):
            f.write(json.dumps(tierA2_confirmed[repo], ensure_ascii=False) + "\n")

    # Evidence CSV (compact)
    with open(outp("tierA2_usage_loader_confirmed_repo_details.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repo", "num_models", "num_code_files", "num_config_files", "evidence_count",
                    "evidence_examples"])
        for repo in sorted(tierA2_confirmed.keys()):
            d = tierA2_confirmed[repo]
            examples = []
            for ev in d.get("loader_evidence", [])[:5]:
                examples.append(f"{ev['model_id']}@{ev['file']} loaders={','.join(ev['loaders'])}")
            w.writerow([
                repo,
                d["num_models"],
                len(d["code_files"]),
                len(d["config_files"]),
                len(d.get("loader_evidence", [])),
                " | ".join(examples),
            ])

    stats = {
        "total_repos": len(repos),
        "tier_counts": dict(tier_counts),
        "tierA1_usage_filetype_repos": len(tierA),
        "tierA2_usage_loader_confirmed_repos": len(tierA2_confirmed),
        "tierA2_usage_loader_unconfirmed_repos": len(tierA2_unconfirmed),
        "tokens_loaded": len(TOKENS),
        "workers": args.workers,
        "max_files_per_repo": args.max_files_per_repo,
        "loader_patterns": list(LOADER_REGEXES.keys()),
        "quant_evidence_patterns": list(QUANT_EVIDENCE_REGEXES.keys()),
        "confirmation_rule": "model_id string AND loader regex in same file; notebooks scan code cells only",
    }
    with open(outp("loader_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDONE")
    print(f"Tier A1 (file-type usage): {len(tierA):,}")
    print(f"Tier A2 (loader-confirmed): {len(tierA2_confirmed):,}")
    print(f"Unconfirmed within Tier A1: {len(tierA2_unconfirmed):,}")
    print(f"Outputs: {args.output_dir}")

if __name__ == "__main__":
    main()
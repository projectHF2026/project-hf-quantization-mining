#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_usage_repos.py
=====================
Classifies repos into 4 tiers based on which matched file types contain
quantized model references.

Tier A (usage):        at least one executable code file match
Tier B (config-only):  config matches exist, but no executable code matches
Tier C (mention-only): matches only in docs/catalog files
Tier D (ambiguous):    only unclear file types

Inputs:
  - repo_details.csv produced by your final_data step
    columns: repo,num_models,num_files,models,files

Outputs (in --output-dir):
  - tierA_usage_repos.txt + CSV + JSONL
  - tierB_config_only_repos.txt + CSV + JSONL
  - tierC_mention_only_repos.txt + CSV + JSONL
  - tierD_ambiguous_repos.txt + CSV + JSONL
  - tier_stats.json

Usage:
  python3 filter_usage_repos.py \
    --repo-details /path/to/final_data/repo_details.csv \
    --output-dir /path/to/final_data/usage_filtered
"""

import os
import re
import csv
import json
import argparse
from collections import Counter

# ============================================================================
# FILE CLASSIFICATION RULES
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

# Config extensions: classified separately as "config".
# A repo with ONLY config files (no code) = "config-only" (Tier B).
# Config files do NOT get promoted to "code" — they stay as "config".
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
    # Catalog/listing paths (specific, NOT generic "models/" which is common in code)
    re.compile(r'(^|/)(model[-_]?zoo|model[-_]?list|model[-_]?map|modelzoo|leaderboard)(s)?(/|$)', re.IGNORECASE),
    re.compile(r'(^|/)gh-pages/', re.IGNORECASE),
    re.compile(r'(^|/)_posts/', re.IGNORECASE),
    re.compile(r'(^|/)site/', re.IGNORECASE),
    # docs/models/ is a catalog, but src/models/ is code
    re.compile(r'(^|/)docs?/models?/', re.IGNORECASE),
]

USAGE_JSON_FILENAMES = {
    "quantize_config.json", "quant_config.json",
    "quantization_config.json",
}

# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_file(file_path: str) -> str:
    """
    Return one of: code | config | doc | ambiguous
    """
    basename = os.path.basename(file_path).lower()
    _, ext = os.path.splitext(basename)

    # Doc path patterns override everything
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


def classify_repo(files: list) -> dict:
    """
    Tier assignment:
      Tier A: code_files exists
      Tier B: no code_files, config_files exists
      Tier D: no code/config, and no doc (only ambiguous)
      Tier C: otherwise (doc-only or doc+ambiguous)
    """
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
# OUTPUT HELPERS
# ============================================================================

def write_repo_list(path: str, repos: dict):
    with open(path, "w", encoding="utf-8") as f:
        for repo in sorted(repos.keys()):
            f.write(repo + "\n")

def write_repo_csv(path: str, repos: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "repo",
            "num_models",
            "num_code_files",
            "num_config_files",
            "num_doc_files",
            "num_ambiguous_files",
            "models",
            "code_files",
            "config_files",
            "doc_files",
            "ambiguous_files",
        ])
        for repo in sorted(repos.keys()):
            d = repos[repo]
            w.writerow([
                repo,
                d["num_models"],
                len(d["code_files"]),
                len(d["config_files"]),
                len(d["doc_files"]),
                len(d["ambiguous_files"]),
                ";".join(d["models"]),
                ";".join(d["code_files"]),
                ";".join(d["config_files"]),
                ";".join(d["doc_files"]),
                ";".join(d["ambiguous_files"]),
            ])

def write_repo_jsonl(path: str, repos: dict):
    with open(path, "w", encoding="utf-8") as f:
        for repo in sorted(repos.keys()):
            f.write(json.dumps(repos[repo], ensure_ascii=False) + "\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Classify repos into 4 tiers based on file types of matched references"
    )
    ap.add_argument("--repo-details", required=True,
                    help="Path to repo_details.csv")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory for tier results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Increase CSV field size limit for repos with many models/files
    csv.field_size_limit(10 * 1024 * 1024)  # 10MB

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
                "models": models,
                "num_models": int(row["num_models"]),
                "files": files,
            }

    print(f"File-based Tier Classification")
    print(f"{'='*60}")
    print(f"Total repos to classify: {len(repos):,}")
    print()

    # Buckets
    tierA = {}
    tierB = {}
    tierC = {}
    tierD = {}

    ext_counter = Counter()
    tier_counter = Counter()
    model_sum = Counter()

    for repo, base in repos.items():
        cls = classify_repo(base["files"])
        tier = cls["tier"]
        tier_counter[tier] += 1
        model_sum[tier] += base["num_models"]

        entry = {
            "repo": repo,
            "num_models": base["num_models"],
            "models": base["models"],
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

        for fp in base["files"]:
            _, ext = os.path.splitext(fp.lower())
            ext_counter[ext or "(no ext)"] += 1

    # Write outputs
    def out(name):
        return os.path.join(args.output_dir, name)

    write_repo_list(out("tierA_usage_repos.txt"), tierA)
    write_repo_csv(out("tierA_usage_repo_details.csv"), tierA)
    write_repo_jsonl(out("tierA_usage_repo_details.jsonl"), tierA)

    write_repo_list(out("tierB_config_only_repos.txt"), tierB)
    write_repo_csv(out("tierB_config_only_repo_details.csv"), tierB)
    write_repo_jsonl(out("tierB_config_only_repo_details.jsonl"), tierB)

    write_repo_list(out("tierC_mention_only_repos.txt"), tierC)
    write_repo_csv(out("tierC_mention_only_repo_details.csv"), tierC)
    write_repo_jsonl(out("tierC_mention_only_repo_details.jsonl"), tierC)

    write_repo_list(out("tierD_ambiguous_repos.txt"), tierD)
    write_repo_csv(out("tierD_ambiguous_repo_details.csv"), tierD)
    write_repo_jsonl(out("tierD_ambiguous_repo_details.jsonl"), tierD)

    # Stats
    total = len(repos)
    stats = {
        "total_repos": total,
        "tier_counts": {k: v for k, v in tier_counter.items()},
        "tier_percentages": {k: round(v / total * 100, 2) for k, v in tier_counter.items()},
        "sum_num_models_by_tier": {k: int(v) for k, v in model_sum.items()},
        "top_file_extensions_in_matches": dict(ext_counter.most_common(30)),
    }
    with open(out("tier_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"{'='*60}")
    print(f"TIER CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Total repos:     {total:>10,}")
    for k in ["A_usage", "B_config_only", "C_mention_only", "D_ambiguous"]:
        c = tier_counter.get(k, 0)
        m = model_sum.get(k, 0)
        print(f"  {k:<16s}: {c:>8,} repos  ({c/total*100:.1f}%)  | {m:>8,} model refs")
    print(f"\nTop 15 file extensions in matches:")
    for ext, count in ext_counter.most_common(15):
        print(f"  {ext:<15s}: {count:>8,}")
    print(f"\nOutputs written to: {args.output_dir}")
    print(f"  (see tier_stats.json for full summary)")


if __name__ == "__main__":
    main()
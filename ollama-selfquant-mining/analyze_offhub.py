#!/usr/bin/env python3
"""Offline analysis of the off-hub (missed) repos.

Reads missed_repos.jsonl in this folder (read-only) and writes analysis
artefacts to analysis_offhub/. No GitHub API calls. Stdlib only.

Two analyses:
  1) Adoption taxonomy + method preference
  2) Repo characteristics (popularity & activity)

Outputs:
  analysis_offhub/taxonomy_method.csv         (section, group, key, count, pct)
  analysis_offhub/repo_characteristics.csv    (one row per cohort)
  analysis_offhub/offhub_obtain_repo_ids.txt  (self_quant + ollama_obtain repos)
  analysis_offhub/analysis_summary.md         (human-readable)
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE        = Path(__file__).resolve().parent
MISSED      = HERE / "missed_repos.jsonl"
OUT_DIR     = HERE / "analysis_offhub"

# Reference date for pushed_at recency. Today per the session context.
REFERENCE_DATE = datetime(2026, 6, 16, tzinfo=timezone.utc)


# ===========================================================================
# Adoption-mode classification
# ===========================================================================

# Ollama signals that indicate the repo OBTAINS or PRODUCES a model artifact
# (vs. merely calling a running Ollama server as a backend).
OLLAMA_OBTAIN_SIGNALS = {
    "ollama_modelfile_file",
    "ollama_modelfile_FROM",
    "ollama_cli_pull",
    "ollama_cli_create",
    "ollama_cli_push",
    "ollama_registry_old",
    "ollama_registry_library",
}


def has_category(repo: dict, c: str) -> bool:
    return c in (repo.get("categories") or [])


def signal_names(repo: dict) -> set[str]:
    return {s.get("signal", "") for s in repo.get("signals") or []}


def has_ollama_obtain(repo: dict) -> bool:
    return bool(signal_names(repo) & OLLAMA_OBTAIN_SIGNALS)


def classify_mode(repo: dict) -> str | None:
    """Priority assignment:
      1) self_quantized        if any Category B signal
      2) ollama_obtains_model  else if any Ollama obtain/produce signal
      3) ollama_backend_only   else if any Category A signal
      4) None                  unclassifiable (expected 0)
    """
    if has_category(repo, "B"):
        return "self_quantized"
    if has_ollama_obtain(repo):
        return "ollama_obtains_model"
    if has_category(repo, "A"):
        return "ollama_backend_only"
    return None


# ===========================================================================
# Tool family + precision parsing
# ===========================================================================

# Map the mining-tool field -> consolidated method family.
TOOL_FAMILY: dict[str, str] = {
    # GPTQ
    "GPTQ:AutoGPTQ":              "GPTQ",
    "GPTQ:GPTQModel":             "GPTQ",
    "GPTQ":                       "GPTQ",
    "HF GPTQ":                    "GPTQ",
    "Optimum GPTQ":               "GPTQ",
    "GPTQ-for-LLaMa":             "GPTQ",
    "save_pretrained_gptq":       "GPTQ",
    # AWQ
    "AWQ:AutoAWQ":                "AWQ",
    "HF AWQ":                     "AWQ",
    "MIT llm-awq":                "AWQ",
    # llama.cpp / GGUF convert+quantize
    "llama.cpp convert":          "llama.cpp/GGUF-convert",
    "llama.cpp quantize":         "llama.cpp/GGUF-convert",
    "gguf-py":                    "llama.cpp/GGUF-convert",
    # torchao
    "torchao":                    "torchao",
    # ONNX Runtime (incl. Optimum ORT, which wraps it)
    "ONNX Runtime":               "ONNXRuntime",
    "Optimum ORT":                "ONNXRuntime",
    # OpenVINO / NNCF
    "OpenVINO NNCF":              "OpenVINO/NNCF",
    "OpenVINO":                   "OpenVINO/NNCF",
    # ExLlamaV2
    "ExLlamaV2":                  "ExLlamaV2",
    # compressed-tensors
    "compressed-tensors":         "compressed-tensors",
    # Intel Neural Compressor
    "Intel Neural Compressor":    "Intel-NC",
    # AMD Quark
    "AMD Quark":                  "AMD-Quark",
    # SmoothQuant
    "SmoothQuant":                "SmoothQuant",
    # Quanto (HF optimum-quanto + Quanto core)
    "optimum-quanto":              "quanto",
    "Quanto":                     "quanto",
    # Everything else falls through to "other"
    # (EETQ, TensorRT, llm-compressor, ...)
}


def tool_families_for_repo(repo: dict) -> set[str]:
    fams: set[str] = set()
    for s in repo.get("signals") or []:
        if s.get("category") != "B":
            continue
        tool = s.get("tool") or ""
        fams.add(TOOL_FAMILY.get(tool, "other"))
    return fams


# ---- Precision parsing ----------------------------------------------------

# Specific precision tokens (order matters only for clarity in reports).
PRECISION_PATTERNS: list[tuple[str, str]] = [
    # GGUF K-quants
    ("Q2_K",     r"\bQ2_K\b"),
    ("Q3_K_S",   r"\bQ3_K_S\b"),
    ("Q3_K_M",   r"\bQ3_K_M\b"),
    ("Q3_K_L",   r"\bQ3_K_L\b"),
    ("Q4_K_S",   r"\bQ4_K_S\b"),
    ("Q4_K_M",   r"\bQ4_K_M\b"),
    ("Q5_K_S",   r"\bQ5_K_S\b"),
    ("Q5_K_M",   r"\bQ5_K_M\b"),
    ("Q6_K",     r"\bQ6_K\b"),
    # Legacy GGUF
    ("Q4_0",     r"\bQ4_0\b"),
    ("Q4_1",     r"\bQ4_1\b"),
    ("Q5_0",     r"\bQ5_0\b"),
    ("Q5_1",     r"\bQ5_1\b"),
    ("Q8_0",     r"\bQ8_0\b"),
    ("Q8_1",     r"\bQ8_1\b"),
    # IQ family
    ("IQ1_S",    r"\bIQ1_S\b"),
    ("IQ2_XS",   r"\bIQ2_XS\b"),
    ("IQ2_XXS",  r"\bIQ2_XXS\b"),
    ("IQ3_XS",   r"\bIQ3_XS\b"),
    ("IQ3_XXS",  r"\bIQ3_XXS\b"),
    ("IQ4_NL",   r"\bIQ4_NL\b"),
    ("IQ4_XS",   r"\bIQ4_XS\b"),
    # Floating-point
    ("F32",      r"\bF32\b"),
    ("F16",      r"\bF16\b"),
    ("BF16",     r"\bBF16\b"),
    ("FP32",     r"\bFP32\b"),
    ("FP16",     r"\bFP16\b"),
    ("FP8",      r"\bFP8\b"),
    # PyTorch-style integer (require explicit "bit" suffix so we don't catch
    # model-size tags like "8b"/"7b")
    ("4bit",     r"\b(?:int4|4bit|4-bit|load_in_4bit|int4_weight_only)\b"),
    ("8bit",     r"\b(?:int8|8bit|8-bit|load_in_8bit|int8_weight_only|"
                 r"int8_dynamic_activation_int8_weight)\b"),
]
PRECISION_COMPILED = [(label, re.compile(p, re.IGNORECASE))
                     for label, p in PRECISION_PATTERNS]
PRECISION_Q_CATCHALL  = re.compile(r"\bQ[0-9]+[_A-Z0-9]*\b", re.IGNORECASE)
PRECISION_IQ_CATCHALL = re.compile(r"\bIQ[0-9]+[_A-Z0-9]*\b", re.IGNORECASE)

# Size-tag detector: 7b, 8b, 1.5B, 70b, 1k -- model size, NOT precision.
SIZE_TAG_RE = re.compile(r"^\s*\d+(?:\.\d+)?[bBkKmM]\s*$")


def _scan_one_text(text: str, found: set[str]) -> None:
    """Add any precision labels matched in `text` to `found`."""
    matched_q = matched_iq = False
    for label, rgx in PRECISION_COMPILED:
        if rgx.search(text):
            found.add(label)
            if label.startswith("Q"):
                matched_q = True
            if label.startswith("IQ"):
                matched_iq = True
    # Catchall buckets for less-common Q-/IQ-quants -- only if no specific
    # match was found in this same text.
    if not matched_q and PRECISION_Q_CATCHALL.search(text):
        # Make sure the catchall hit isn't a false positive on "Q4_K_M" etc.
        # (it would only get here if the specific patterns didn't already
        # match, so this captures genuinely unrecognised Qn variants.)
        found.add("Q-other")
    if not matched_iq and PRECISION_IQ_CATCHALL.search(text):
        found.add("IQ-other")


def extract_precisions(repo: dict) -> set[str]:
    """Return the set of precision-tag labels present in this repo's
    signal fragments and quant-tag fields. Size tags (8b/7b/1.5B/...) are
    excluded by construction (specific regexes require 'bit' suffix or
    Q/IQ/F prefix; ollama_quant_tag values that match SIZE_TAG_RE are
    dropped from scanning)."""
    found: set[str] = set()
    for s in repo.get("signals") or []:
        # quant-tag string fields
        for k in ("gguf_quant_tag", "ollama_quant_tag"):
            v = s.get(k)
            if not v:
                continue
            v_str = str(v).strip()
            if SIZE_TAG_RE.match(v_str):
                continue  # drop size tags like "4b", "7b", "1.5B"
            _scan_one_text(v_str, found)
        # raw fragment text
        for frag in s.get("fragments") or []:
            if frag:
                _scan_one_text(str(frag), found)
    return found


# ===========================================================================
# Repo characteristics
# ===========================================================================

def parse_iso(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def percentile(data: list[float], p: float) -> float | None:
    """Linear-interpolation percentile. p in [0, 100]."""
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def repo_char_stats(repos: list[dict]) -> dict[str, Any]:
    n = len(repos)
    if n == 0:
        return {"n": 0}
    stars = [int(r.get("stars") or 0) for r in repos]
    forks = [int(r.get("forks") or 0) for r in repos]
    n_archived = sum(1 for r in repos if r.get("archived") is True)
    n_fork     = sum(1 for r in repos if r.get("fork") is True)
    languages  = Counter((r.get("language") or "Unknown") for r in repos)

    created_year_hist: Counter = Counter()
    for r in repos:
        dt = parse_iso(r.get("created_at"))
        if dt:
            created_year_hist[dt.year] += 1

    pushed = {"active_last_6mo": 0, "active_last_12mo": 0,
              "older_than_12mo": 0, "unknown": 0}
    for r in repos:
        dt = parse_iso(r.get("pushed_at"))
        if dt is None:
            pushed["unknown"] += 1
            continue
        days = (REFERENCE_DATE - dt).days
        if days <= 183:
            pushed["active_last_6mo"] += 1
        elif days <= 365:
            pushed["active_last_12mo"] += 1
        else:
            pushed["older_than_12mo"] += 1

    return {
        "n":              n,
        "stars_median":   statistics.median(stars),
        "stars_mean":     statistics.mean(stars),
        "stars_p90":      percentile(stars, 90),
        "stars_p99":      percentile(stars, 99),
        "stars_eq0_pct":  100.0 * sum(1 for s in stars if s == 0) / n,
        "stars_ge1_pct":  100.0 * sum(1 for s in stars if s >= 1) / n,
        "stars_ge5_pct":  100.0 * sum(1 for s in stars if s >= 5) / n,
        "stars_ge10_pct": 100.0 * sum(1 for s in stars if s >= 10) / n,
        "forks_median":   statistics.median(forks),
        "forks_mean":     statistics.mean(forks),
        "archived_pct":   100.0 * n_archived / n,
        "fork_pct":       100.0 * n_fork / n,
        "lang_top15":     languages.most_common(15),
        "created_years":  dict(sorted(created_year_hist.items())),
        "pushed":         pushed,
    }


# ===========================================================================
# Reporting helpers
# ===========================================================================

def pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.2f}%" if den else "n/a"


def fmt_n(x: Any, places: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{places}f}"
    return str(x)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MISSED.exists():
        sys.exit(f"ERROR: {MISSED} not found")

    print(f"[load] {MISSED}", flush=True)
    repos: list[dict] = []
    with MISSED.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                repos.append(json.loads(line))
            except Exception as e:
                print(f"  bad json: {e!r}", file=sys.stderr)
    n_total = len(repos)
    print(f"[load] {n_total:,} repos", flush=True)

    # --- Analysis 1: Adoption taxonomy --------------------------------------
    print("[classify] adoption mode (priority assignment)", flush=True)
    mode_counts: Counter = Counter()
    unclassified: list[str] = []
    n_self_quant_also_ollama = 0
    obtain_subset: list[dict] = []   # self_quant + ollama_obtains_model
    mode_to_repos: dict[str, list[dict]] = defaultdict(list)
    for r in repos:
        m = classify_mode(r)
        if m is None:
            unclassified.append(r.get("full_name", "?"))
            continue
        mode_counts[m] += 1
        mode_to_repos[m].append(r)
        if m == "self_quantized" and has_ollama_obtain(r):
            n_self_quant_also_ollama += 1
        if m in ("self_quantized", "ollama_obtains_model"):
            obtain_subset.append(r)

    print(f"  mode_counts: {dict(mode_counts)}", flush=True)
    print(f"  unclassified: {len(unclassified)} "
          f"(expected 0)", flush=True)
    print(f"  n_self_quant_also_ollama: {n_self_quant_also_ollama}", flush=True)
    print(f"  obtain-or-produce subset: {len(obtain_subset):,}", flush=True)

    # Tool families across obtain-or-produce repos (only self-quant repos
    # actually contribute B-tool families; ollama_obtains_model repos
    # contribute nothing here by construction, but we still count over the
    # union so the table answers "how do the obtain-or-produce repos quantize?")
    print("[classify] tool families (Category B)", flush=True)
    tool_family_counts: Counter = Counter()
    for r in obtain_subset:
        for fam in tool_families_for_repo(r):
            tool_family_counts[fam] += 1
    print(f"  tool families: {dict(tool_family_counts)}", flush=True)

    # Precision distribution among obtain-or-produce repos.
    print("[classify] precision distribution", flush=True)
    precision_counts: Counter = Counter()
    n_with_precision = 0
    for r in obtain_subset:
        precs = extract_precisions(r)
        if precs:
            n_with_precision += 1
            for p in precs:
                precision_counts[p] += 1
    print(f"  obtain-or-produce repos with identifiable precision: "
          f"{n_with_precision:,}/{len(obtain_subset):,} "
          f"({pct(n_with_precision, len(obtain_subset))})", flush=True)
    print(f"  top precisions: {precision_counts.most_common(8)}", flush=True)

    # --- Analysis 2: Repo characteristics -----------------------------------
    print("[stats] repo characteristics by cohort", flush=True)
    cohorts: list[tuple[str, list[dict]]] = [
        ("all_offhub",          repos),
        ("self_quantized",      mode_to_repos["self_quantized"]),
        ("ollama_obtains_model",mode_to_repos["ollama_obtains_model"]),
        ("ollama_backend_only", mode_to_repos["ollama_backend_only"]),
    ]
    cohort_stats: dict[str, dict] = {}
    for name, sub in cohorts:
        cohort_stats[name] = repo_char_stats(sub)
        s = cohort_stats[name]
        print(f"  {name:<22}: n={s['n']:,}  "
              f"stars med={fmt_n(s.get('stars_median'))} "
              f"mean={fmt_n(s.get('stars_mean'))}  "
              f"%0={fmt_n(s.get('stars_eq0_pct'))}%", flush=True)

    # --- Write outputs ------------------------------------------------------
    print(f"[write] {OUT_DIR}", flush=True)

    # taxonomy_method.csv  (section, group, key, count, pct_of_basis, basis)
    tx_path = OUT_DIR / "taxonomy_method.csv"
    with tx_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "group", "key", "count",
                    "pct_of_basis", "basis"])
        # mode_counts
        for mode, n in mode_counts.most_common():
            w.writerow(["mode_counts", "adoption_mode", mode, n,
                        pct(n, n_total), f"all_offhub={n_total}"])
        # also record the overlap + unclassified
        w.writerow(["mode_counts", "overlap",
                    "self_quant_AND_ollama_obtain",
                    n_self_quant_also_ollama,
                    pct(n_self_quant_also_ollama, mode_counts['self_quantized']),
                    f"self_quantized={mode_counts['self_quantized']}"])
        w.writerow(["mode_counts", "unclassified",
                    "unclassified", len(unclassified),
                    pct(len(unclassified), n_total),
                    f"all_offhub={n_total}"])
        # tool_family_counts
        for fam, n in tool_family_counts.most_common():
            w.writerow(["tool_family_counts", "method_family", fam, n,
                        pct(n, mode_counts["self_quantized"]),
                        f"self_quantized={mode_counts['self_quantized']}"])
        # precision_distribution
        w.writerow(["precision_distribution", "coverage",
                    "repos_with_identifiable_precision",
                    n_with_precision,
                    pct(n_with_precision, len(obtain_subset)),
                    f"obtain_or_produce={len(obtain_subset)}"])
        for prec, n in precision_counts.most_common():
            w.writerow(["precision_distribution", "precision_tag",
                        prec, n,
                        pct(n, n_with_precision),
                        f"with_identifiable_precision={n_with_precision}"])
    print(f"  wrote {tx_path}")

    # repo_characteristics.csv  (one row per cohort)
    rc_path = OUT_DIR / "repo_characteristics.csv"
    rc_cols = [
        "cohort", "n",
        "stars_median", "stars_mean", "stars_p90", "stars_p99",
        "stars_eq0_pct", "stars_ge1_pct", "stars_ge5_pct", "stars_ge10_pct",
        "forks_median", "forks_mean",
        "archived_pct", "fork_pct",
        "pushed_active_last_6mo", "pushed_active_last_12mo",
        "pushed_older_than_12mo", "pushed_unknown",
    ]
    with rc_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(rc_cols)
        for name, _sub in cohorts:
            s = cohort_stats[name]
            row = [
                name, s.get("n", 0),
                fmt_n(s.get("stars_median")), fmt_n(s.get("stars_mean")),
                fmt_n(s.get("stars_p90")), fmt_n(s.get("stars_p99")),
                fmt_n(s.get("stars_eq0_pct")), fmt_n(s.get("stars_ge1_pct")),
                fmt_n(s.get("stars_ge5_pct")), fmt_n(s.get("stars_ge10_pct")),
                fmt_n(s.get("forks_median")), fmt_n(s.get("forks_mean")),
                fmt_n(s.get("archived_pct")), fmt_n(s.get("fork_pct")),
                s.get("pushed", {}).get("active_last_6mo", 0),
                s.get("pushed", {}).get("active_last_12mo", 0),
                s.get("pushed", {}).get("older_than_12mo", 0),
                s.get("pushed", {}).get("unknown", 0),
            ]
            w.writerow(row)
    print(f"  wrote {rc_path}")

    # offhub_obtain_repo_ids.txt
    ids_path = OUT_DIR / "offhub_obtain_repo_ids.txt"
    with ids_path.open("w") as f:
        for r in obtain_subset:
            fn = r.get("full_name")
            if fn:
                f.write(f"{fn}\n")
    print(f"  wrote {ids_path} ({len(obtain_subset):,} ids)")

    # analysis_summary.md
    md_path = OUT_DIR / "analysis_summary.md"
    L: list[str] = []
    L.append("# Off-hub repo analysis")
    L.append("")
    L.append(f"Reference date for activity buckets: "
             f"{REFERENCE_DATE.date().isoformat()}.")
    L.append(f"Input: `{MISSED.name}` ({n_total:,} repos missed by the "
             f"HF-anchored pipeline).")
    L.append("")
    # Analysis 1
    L.append("## 1. Adoption taxonomy & method preference")
    L.append("")
    L.append("Each repo is assigned ONE adoption mode using a priority rule:")
    L.append("")
    L.append("1. `self_quantized` — any Category B signal")
    L.append("2. `ollama_obtains_model` — any Ollama obtain/produce signal "
             "(`Modelfile FROM`, `ollama pull`, `ollama create`, `ollama push`, "
             "registry refs)")
    L.append("3. `ollama_backend_only` — Ollama usage only via client / REST / "
             "LangChain / LlamaIndex / `ollama run`")
    L.append("")
    L.append("| mode | count | pct of all off-hub |")
    L.append("|---|---:|---:|")
    for mode in ("self_quantized", "ollama_obtains_model", "ollama_backend_only"):
        n = mode_counts.get(mode, 0)
        L.append(f"| `{mode}` | {n:,} | {pct(n, n_total)} |")
    L.append(f"| unclassified | {len(unclassified):,} | "
             f"{pct(len(unclassified), n_total)} |")
    L.append(f"| **total** | **{n_total:,}** | 100% |")
    L.append("")
    if unclassified:
        L.append(f"**Unclassified repos** ({len(unclassified)}):")
        for fn in unclassified[:50]:
            L.append(f"- `{fn}`")
        if len(unclassified) > 50:
            L.append(f"... +{len(unclassified) - 50} more")
        L.append("")
    else:
        L.append("All repos were assigned to one of the three modes "
                 "(0 unclassified, as expected).")
        L.append("")
    L.append(f"**Overlap (`self_quantize AND serve via Ollama`):** "
             f"`n_self_quant_also_ollama` = **{n_self_quant_also_ollama:,}** "
             f"(= {pct(n_self_quant_also_ollama, mode_counts.get('self_quantized', 0))}"
             f" of `self_quantized`). These are repos that would have been "
             f"`ollama_obtains_model` without the priority rule.")
    L.append("")
    L.append(f"**Obtain-or-produce subset:** "
             f"`self_quantized` + `ollama_obtains_model` = "
             f"{len(obtain_subset):,} repos. This is the universe for the "
             f"method/precision tables below and the input for the later "
             f"temporal pass (`offhub_obtain_repo_ids.txt`).")
    L.append("")
    L.append("### Self-quant method-family distribution")
    L.append("")
    L.append(f"Across `self_quantized` repos "
             f"({mode_counts.get('self_quantized', 0):,}); each repo counted "
             f"once per family it uses (a repo can use >1).")
    L.append("")
    L.append("| method family | repos | pct of self_quantized |")
    L.append("|---|---:|---:|")
    sq_n = max(mode_counts.get("self_quantized", 0), 1)
    for fam, n in tool_family_counts.most_common():
        L.append(f"| {fam} | {n:,} | {pct(n, sq_n)} |")
    L.append("")
    L.append("### Precision distribution (where identifiable)")
    L.append("")
    L.append(f"Computed across the obtain-or-produce subset "
             f"({len(obtain_subset):,} repos). "
             f"**{n_with_precision:,}** repos "
             f"({pct(n_with_precision, len(obtain_subset))}) carry an "
             f"identifiable precision tag. Size tags such as `7b`/`8b`/`1.5B` "
             f"are NOT counted as precisions (they're model size).")
    L.append("")
    L.append("| precision | repos | pct of repos w/ identifiable precision |")
    L.append("|---|---:|---:|")
    for prec, n in precision_counts.most_common():
        L.append(f"| {prec} | {n:,} | {pct(n, max(n_with_precision, 1))} |")
    L.append("")
    # Analysis 2
    L.append("## 2. Repo characteristics by adoption mode")
    L.append("")
    L.append("Stars (median / mean / p90 / p99 / share at each cut), forks "
             "(median, mean), archived & fork share, and pushed-at recency "
             f"(reference date {REFERENCE_DATE.date().isoformat()}).")
    L.append("")
    L.append("| cohort | n | stars med | stars mean | p90 | p99 | "
             "%=0 | %>=1 | %>=5 | %>=10 | forks med | forks mean | "
             "%archived | %fork | active <6mo | <12mo | older |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
             "---:|---:|---:|---:|---:|---:|")
    for name, _sub in cohorts:
        s = cohort_stats[name]
        pu = s.get("pushed", {})
        L.append(
            f"| `{name}` | {s.get('n', 0):,} | "
            f"{fmt_n(s.get('stars_median'))} | {fmt_n(s.get('stars_mean'))} | "
            f"{fmt_n(s.get('stars_p90'))} | {fmt_n(s.get('stars_p99'))} | "
            f"{fmt_n(s.get('stars_eq0_pct'))}% | "
            f"{fmt_n(s.get('stars_ge1_pct'))}% | "
            f"{fmt_n(s.get('stars_ge5_pct'))}% | "
            f"{fmt_n(s.get('stars_ge10_pct'))}% | "
            f"{fmt_n(s.get('forks_median'))} | {fmt_n(s.get('forks_mean'))} | "
            f"{fmt_n(s.get('archived_pct'))}% | "
            f"{fmt_n(s.get('fork_pct'))}% | "
            f"{pu.get('active_last_6mo', 0):,} | "
            f"{pu.get('active_last_12mo', 0):,} | "
            f"{pu.get('older_than_12mo', 0):,} |"
        )
    L.append("")
    # Language distributions
    L.append("### Top-15 primary languages by cohort")
    L.append("")
    for name, _sub in cohorts:
        s = cohort_stats[name]
        L.append(f"**`{name}`** ({s.get('n', 0):,} repos):")
        L.append("")
        L.append("| language | count |")
        L.append("|---|---:|")
        for lang, n in s.get("lang_top15", []):
            L.append(f"| {lang} | {n:,} |")
        L.append("")
    # Created-year histograms
    L.append("### Created_at by year")
    L.append("")
    years_seen: set[int] = set()
    for name, _sub in cohorts:
        years_seen.update(cohort_stats[name].get("created_years", {}).keys())
    years_sorted = sorted(years_seen)
    header = "| cohort | " + " | ".join(str(y) for y in years_sorted) + " |"
    sep = "|---|" + "|".join(["---:"] * len(years_sorted)) + "|"
    L.append(header)
    L.append(sep)
    for name, _sub in cohorts:
        cy = cohort_stats[name].get("created_years", {})
        cells = [str(cy.get(y, 0)) for y in years_sorted]
        L.append(f"| `{name}` | " + " | ".join(cells) + " |")
    L.append("")
    md_path.write_text("\n".join(L) + "\n")
    print(f"  wrote {md_path}")

    # --- Stdout key numbers ------------------------------------------------
    print()
    print("=" * 72)
    print(f"TOTAL off-hub repos: {n_total:,}")
    print(f"  self_quantized         : {mode_counts.get('self_quantized', 0):,}")
    print(f"  ollama_obtains_model   : {mode_counts.get('ollama_obtains_model', 0):,}")
    print(f"  ollama_backend_only    : {mode_counts.get('ollama_backend_only', 0):,}")
    print(f"  unclassified           : {len(unclassified):,}")
    print(f"  obtain-or-produce      : {len(obtain_subset):,}")
    print(f"  n_self_quant_also_ollama: {n_self_quant_also_ollama:,}")
    print()
    print(f"  precision-identified   : {n_with_precision:,} of {len(obtain_subset):,} "
          f"({pct(n_with_precision, len(obtain_subset))})")
    print()
    print("Top 5 method families:")
    for fam, n in tool_family_counts.most_common(5):
        print(f"  {fam:<24} {n:>5,}")
    print()
    print("Top 5 precisions:")
    for p, n in precision_counts.most_common(5):
        print(f"  {p:<14} {n:>5,}")


if __name__ == "__main__":
    main()

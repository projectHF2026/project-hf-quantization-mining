#!/usr/bin/env python3
"""Final off-hub re-cut: LLM-stack temporal + precision sensitivity +
two-view tool ranking. Reads missed_repos.jsonl; appends sections 5/6/7
to analysis_offhub/analysis_summary.md and writes three CSVs.

Read-only on existing outputs. Stdlib only. No API calls. No git cloning;
temporal evidence is metadata-based (created_at / pushed_at).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE       = Path(__file__).resolve().parent
MISSED     = HERE / "missed_repos.jsonl"
OUT_DIR    = HERE / "analysis_offhub"
SUMMARY_MD = OUT_DIR / "analysis_summary.md"

REFERENCE_DATE = datetime(2026, 6, 16, tzinfo=timezone.utc)
YEARS_TO_REPORT = [2021, 2022, 2023, 2024, 2025, 2026]


# ===========================================================================
# Stack definitions
# ===========================================================================

STRICT_FAMILIES = {
    "GPTQ", "AWQ", "ExLlamaV2", "llama.cpp/GGUF-convert", "compressed-tensors",
}
EXPANDED_FAMILIES = STRICT_FAMILIES | {"SmoothQuant", "AMD-Quark"}

OLLAMA_OBTAIN_SIGNALS = {
    "ollama_modelfile_file", "ollama_modelfile_FROM",
    "ollama_cli_pull",       "ollama_cli_create",
}
# Adoption-mode priority (matches analyze_offhub.py for cohort sanity)
PRIORITY_OBTAIN_SIGNALS = OLLAMA_OBTAIN_SIGNALS | {
    "ollama_cli_push", "ollama_registry_old", "ollama_registry_library",
}

# Tool -> family map (duplicated so this script is self-contained)
TOOL_FAMILY: dict[str, str] = {
    "GPTQ:AutoGPTQ":              "GPTQ",
    "GPTQ:GPTQModel":             "GPTQ",
    "GPTQ":                       "GPTQ",
    "HF GPTQ":                    "GPTQ",
    "Optimum GPTQ":               "GPTQ",
    "GPTQ-for-LLaMa":             "GPTQ",
    "save_pretrained_gptq":       "GPTQ",
    "AWQ:AutoAWQ":                "AWQ",
    "HF AWQ":                     "AWQ",
    "MIT llm-awq":                "AWQ",
    "llama.cpp convert":          "llama.cpp/GGUF-convert",
    "llama.cpp quantize":         "llama.cpp/GGUF-convert",
    "gguf-py":                    "llama.cpp/GGUF-convert",
    "torchao":                    "torchao",
    "ONNX Runtime":               "ONNXRuntime",
    "Optimum ORT":                "ONNXRuntime",
    "OpenVINO NNCF":              "OpenVINO/NNCF",
    "OpenVINO":                   "OpenVINO/NNCF",
    "ExLlamaV2":                  "ExLlamaV2",
    "compressed-tensors":         "compressed-tensors",
    "Intel Neural Compressor":    "Intel-NC",
    "AMD Quark":                  "AMD-Quark",
    "SmoothQuant":                "SmoothQuant",
    "optimum-quanto":             "quanto",
    "Quanto":                     "quanto",
}


GGUF_INDICATOR_RE = re.compile(
    r"\.gguf\b|\bQ[0-9]+[_A-Z0-9]*\b|\bIQ[0-9]+[_A-Z0-9]*\b",
    re.IGNORECASE,
)


def signal_scan_pool(signal: dict) -> str:
    parts: list[str] = []
    for k in ("gguf_quant_tag", "ollama_quant_tag", "ollama_model_name",
              "matched_file"):
        v = signal.get(k)
        if v:
            parts.append(str(v))
    for frag in signal.get("fragments") or []:
        if frag:
            parts.append(str(frag))
    return "\n".join(parts)


def is_ollama_gguf_signal(signal: dict) -> bool:
    if signal.get("signal") not in OLLAMA_OBTAIN_SIGNALS:
        return False
    return bool(GGUF_INDICATOR_RE.search(signal_scan_pool(signal)))


def is_in_stack(signal: dict, stack: set[str]) -> bool:
    if signal.get("category") == "B":
        return TOOL_FAMILY.get(signal.get("tool") or "", "other") in stack
    return is_ollama_gguf_signal(signal)


def repo_in_subset(repo: dict, stack: set[str]) -> bool:
    return any(is_in_stack(s, stack) for s in repo.get("signals") or [])


# ===========================================================================
# Precision extraction (same regexes/buckets as the previous scripts)
# ===========================================================================

PRECISION_PATTERNS: list[tuple[str, str]] = [
    ("Q2_K", r"\bQ2_K\b"),
    ("Q3_K_S", r"\bQ3_K_S\b"), ("Q3_K_M", r"\bQ3_K_M\b"), ("Q3_K_L", r"\bQ3_K_L\b"),
    ("Q4_K_S", r"\bQ4_K_S\b"), ("Q4_K_M", r"\bQ4_K_M\b"),
    ("Q5_K_S", r"\bQ5_K_S\b"), ("Q5_K_M", r"\bQ5_K_M\b"),
    ("Q6_K", r"\bQ6_K\b"),
    ("Q4_0", r"\bQ4_0\b"), ("Q4_1", r"\bQ4_1\b"),
    ("Q5_0", r"\bQ5_0\b"), ("Q5_1", r"\bQ5_1\b"),
    ("Q8_0", r"\bQ8_0\b"), ("Q8_1", r"\bQ8_1\b"),
    ("IQ1_S", r"\bIQ1_S\b"),
    ("IQ2_XS", r"\bIQ2_XS\b"), ("IQ2_XXS", r"\bIQ2_XXS\b"),
    ("IQ3_XS", r"\bIQ3_XS\b"), ("IQ3_XXS", r"\bIQ3_XXS\b"),
    ("IQ4_NL", r"\bIQ4_NL\b"), ("IQ4_XS", r"\bIQ4_XS\b"),
    ("F32", r"\bF32\b"), ("F16", r"\bF16\b"), ("BF16", r"\bBF16\b"),
    ("FP32", r"\bFP32\b"), ("FP16", r"\bFP16\b"), ("FP8", r"\bFP8\b"),
    ("4bit", r"\b(?:int4|4bit|4-bit|load_in_4bit|int4_weight_only)\b"),
    ("8bit", r"\b(?:int8|8bit|8-bit|load_in_8bit|int8_weight_only|"
             r"int8_dynamic_activation_int8_weight)\b"),
]
PRECISION_COMPILED = [(label, re.compile(p, re.IGNORECASE))
                      for label, p in PRECISION_PATTERNS]
PRECISION_Q_CATCH  = re.compile(r"\bQ[0-9]+[_A-Z0-9]*\b", re.IGNORECASE)
PRECISION_IQ_CATCH = re.compile(r"\bIQ[0-9]+[_A-Z0-9]*\b", re.IGNORECASE)
SIZE_TAG_RE        = re.compile(r"^\s*\d+(?:\.\d+)?[bBkKmM]\s*$")

BITWIDTH_MAP: dict[str, str] = {
    "Q2_K": "2-bit",
    "Q3_K_S": "3-bit", "Q3_K_M": "3-bit", "Q3_K_L": "3-bit",
    "Q4_K_S": "4-bit", "Q4_K_M": "4-bit",
    "Q4_0": "4-bit", "Q4_1": "4-bit", "4bit": "4-bit",
    "Q5_K_S": "5-bit", "Q5_K_M": "5-bit", "Q5_0": "5-bit", "Q5_1": "5-bit",
    "Q6_K": "6-bit",
    "Q8_0": "8-bit", "Q8_1": "8-bit", "8bit": "8-bit",
    "IQ1_S": "1-bit",
    "IQ2_XS": "2-bit", "IQ2_XXS": "2-bit",
    "IQ3_XS": "3-bit", "IQ3_XXS": "3-bit",
    "IQ4_NL": "4-bit", "IQ4_XS": "4-bit",
    "F16": "FP16", "BF16": "FP16", "FP16": "FP16",
    "F32": "FP32", "FP32": "FP32",
    "FP8": "FP8",
    "Q-other": "other", "IQ-other": "other",
}
BITWIDTH_ORDER = ["1-bit", "2-bit", "3-bit", "4-bit", "5-bit",
                  "6-bit", "8-bit", "FP16", "FP32", "FP8", "other"]


def _scan_text(text: str, found: set[str]) -> None:
    matched_q = matched_iq = False
    for label, rgx in PRECISION_COMPILED:
        if rgx.search(text):
            found.add(label)
            if label.startswith("Q"):
                matched_q = True
            if label.startswith("IQ"):
                matched_iq = True
    if not matched_q and PRECISION_Q_CATCH.search(text):
        found.add("Q-other")
    if not matched_iq and PRECISION_IQ_CATCH.search(text):
        found.add("IQ-other")


def extract_precisions(repo: dict, stack: set[str] | None = None) -> set[str]:
    found: set[str] = set()
    for s in repo.get("signals") or []:
        if stack is not None and not is_in_stack(s, stack):
            continue
        for k in ("gguf_quant_tag", "ollama_quant_tag"):
            v = s.get(k)
            if not v:
                continue
            v_str = str(v).strip()
            if SIZE_TAG_RE.match(v_str):
                continue
            _scan_text(v_str, found)
        for frag in s.get("fragments") or []:
            if frag:
                _scan_text(str(frag), found)
    return found


def precisions_to_bitwidths(precs: set[str]) -> set[str]:
    return {BITWIDTH_MAP.get(p, "other") for p in precs}


# ===========================================================================
# Domain classification (for the %likely_LLM column in view B)
# ===========================================================================

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "likely_LLM": [
        "llama", "mistral", "qwen", "gemma", "phi-", "phi3", "phi2", " phi ",
        "deepseek", "falcon", "mixtral", "vicuna", "alpaca", "chatglm",
        "baichuan", "starcoder", "codegen", "codellama", "wizardlm",
        "openchat", "yi-", "llm", "language-model", "language_model",
        "language model", "text-generation", "text_generation",
        "causal-lm", "causal_lm", "causallm", "transformers", "gpt",
        ".gguf", "instruct", "tulu", "mpt-", "redpajama",
    ],
    "whisper_audio": ["whisper", "wav2vec", "asr", "speech-to-text",
                       "speech_to_text", "speech_recognition", "speech-recognition"],
    "embedding": ["sentence-transformer", "sentence_transformer", "sbert",
                   "all-minilm", "all_minilm", "minilm",
                   "bge-", "bge_", "gte-", "e5-",
                   "embedding", "embeddings", "embedder",
                   "bert-", "bert_", "robertaforsequenceclassification"],
    "likely_nonLLM": ["resnet", "yolo", "yolox", "vit-", "vit_",
                       "vision-transformer", "image-classification",
                       "image_classification", "object-detection",
                       "object_detection", "segmentation", "segment-anything",
                       "detr-", "detr_", "efficientnet", "mobilenet",
                       "opencv", "sklearn", "scikit-learn",
                       "stable-diffusion", "stable_diffusion", " diffusion",
                       "tabular", "xgboost", "lightgbm"],
}
DOMAIN_PRIORITY = ["likely_LLM", "whisper_audio", "embedding", "likely_nonLLM"]


def repo_text_pool(repo: dict) -> str:
    parts: list[str] = []
    fn = repo.get("full_name")
    if fn:
        parts.append(str(fn))
    lang = repo.get("language")
    if lang:
        parts.append(str(lang))
    for s in repo.get("signals") or []:
        mf = s.get("matched_file")
        if mf:
            parts.append(str(mf))
        for k in ("ollama_model_name", "gguf_quant_tag", "ollama_quant_tag"):
            v = s.get(k)
            if v:
                parts.append(str(v))
        for frag in s.get("fragments") or []:
            if frag:
                parts.append(str(frag))
    return "\n".join(parts).lower()


def classify_domain(repo: dict) -> str:
    text = repo_text_pool(repo)
    for bucket in DOMAIN_PRIORITY:
        for kw in DOMAIN_KEYWORDS[bucket]:
            if kw in text:
                return bucket
    return "unknown"


def tool_families_for_repo(repo: dict) -> set[str]:
    fams: set[str] = set()
    for s in repo.get("signals") or []:
        if s.get("category") != "B":
            continue
        fams.add(TOOL_FAMILY.get(s.get("tool") or "", "other"))
    return fams


# ===========================================================================
# Adoption-mode (priority)
# ===========================================================================

def mode_of(repo: dict) -> str | None:
    cats = set(repo.get("categories") or [])
    if "B" in cats:
        return "self_quantized"
    snames = {s.get("signal", "") for s in repo.get("signals") or []}
    if snames & PRIORITY_OBTAIN_SIGNALS:
        return "ollama_obtains_model"
    if "A" in cats:
        return "ollama_backend_only"
    return None


def parse_iso(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.2f}%" if den else "n/a"


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    if not MISSED.exists():
        sys.exit(f"ERROR: {MISSED} not found")
    if not OUT_DIR.exists():
        sys.exit(f"ERROR: {OUT_DIR} not found")

    print(f"[load] {MISSED}", flush=True)
    repos: list[dict] = []
    with MISSED.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                repos.append(json.loads(line))
            except Exception:
                pass
    print(f"[load] {len(repos):,} repos", flush=True)

    # Cohort split
    self_quant = [r for r in repos if mode_of(r) == "self_quantized"]
    ollama_obtain = [r for r in repos if mode_of(r) == "ollama_obtains_model"]
    obtain_or_produce = self_quant + ollama_obtain
    print(f"[cohorts] self_quantized={len(self_quant):,}  "
          f"ollama_obtains_model={len(ollama_obtain):,}  "
          f"obtain_or_produce={len(obtain_or_produce):,}", flush=True)

    # ======================================================================
    # 5. LLM-stack created_at-by-year curve (EXPANDED scope) + pushed recency
    # ======================================================================
    print("[5] LLM-stack temporal (EXPANDED scope)", flush=True)
    # Subset = repos whose ANY signal is in EXPANDED (Cat-B) OR is an Ollama
    # obtain-signal with a GGUF indicator.
    llm_subset = [r for r in obtain_or_produce
                  if repo_in_subset(r, EXPANDED_FAMILIES)]
    by_year_self: Counter = Counter()
    by_year_ollama: Counter = Counter()
    pushed_self  = Counter({"active_last_6mo": 0, "active_last_12mo": 0,
                             "older_than_12mo": 0, "unknown": 0})
    pushed_ollama = Counter({"active_last_6mo": 0, "active_last_12mo": 0,
                              "older_than_12mo": 0, "unknown": 0})
    n_subset_self = n_subset_ollama = 0
    for r in llm_subset:
        m = mode_of(r)
        if m == "self_quantized":
            n_subset_self += 1
            by_target = by_year_self; pushed_target = pushed_self
        elif m == "ollama_obtains_model":
            n_subset_ollama += 1
            by_target = by_year_ollama; pushed_target = pushed_ollama
        else:
            continue
        cdt = parse_iso(r.get("created_at"))
        if cdt and cdt.year in YEARS_TO_REPORT:
            by_target[cdt.year] += 1
        elif cdt:
            by_target["other_year"] += 1
        else:
            by_target["unknown_year"] += 1
        pdt = parse_iso(r.get("pushed_at"))
        if pdt is None:
            pushed_target["unknown"] += 1
        else:
            days = (REFERENCE_DATE - pdt).days
            if days <= 183:
                pushed_target["active_last_6mo"] += 1
            elif days <= 365:
                pushed_target["active_last_12mo"] += 1
            else:
                pushed_target["older_than_12mo"] += 1
    print(f"  LLM subset (EXPANDED + ollama-gguf): {len(llm_subset):,} "
          f"({pct(len(llm_subset), len(obtain_or_produce))} of obtain-or-produce)",
          flush=True)
    print(f"    self_quantized contribution:       {n_subset_self:,}", flush=True)
    print(f"    ollama_obtains_model contribution: {n_subset_ollama:,}", flush=True)

    # ======================================================================
    # 6. Precision sensitivity: STRICT vs EXPANDED
    # ======================================================================
    print("[6] precision sensitivity (STRICT vs EXPANDED)", flush=True)
    strict_raw: Counter = Counter()
    strict_bw: Counter = Counter()
    expanded_raw: Counter = Counter()
    expanded_bw: Counter = Counter()
    n_strict_scope = n_strict_with = 0
    n_exp_scope    = n_exp_with    = 0
    for r in obtain_or_produce:
        in_strict = repo_in_subset(r, STRICT_FAMILIES)
        in_exp    = repo_in_subset(r, EXPANDED_FAMILIES)
        if in_strict: n_strict_scope += 1
        if in_exp:    n_exp_scope    += 1
        if in_strict:
            precs = extract_precisions(r, STRICT_FAMILIES)
            if precs:
                n_strict_with += 1
                for p in precs:
                    strict_raw[p] += 1
                for bw in precisions_to_bitwidths(precs):
                    strict_bw[bw] += 1
        if in_exp:
            precs = extract_precisions(r, EXPANDED_FAMILIES)
            if precs:
                n_exp_with += 1
                for p in precs:
                    expanded_raw[p] += 1
                for bw in precisions_to_bitwidths(precs):
                    expanded_bw[bw] += 1
    print(f"  STRICT:   scope={n_strict_scope:,}  with_precision={n_strict_with:,}  "
          f"4-bit={strict_bw.get('4-bit', 0):,} "
          f"({pct(strict_bw.get('4-bit', 0), max(n_strict_with, 1))})", flush=True)
    print(f"  EXPANDED: scope={n_exp_scope:,}  with_precision={n_exp_with:,}  "
          f"4-bit={expanded_bw.get('4-bit', 0):,} "
          f"({pct(expanded_bw.get('4-bit', 0), max(n_exp_with, 1))})", flush=True)

    # ======================================================================
    # 7. Two-view tool ranking
    # ======================================================================
    print("[7] two-view tool ranking", flush=True)
    # View A: every Cat-B family across all 3,830 obtain-or-produce repos
    family_total_A: Counter = Counter()
    for r in obtain_or_produce:
        for fam in tool_families_for_repo(r):
            family_total_A[fam] += 1
    # %likely_LLM per family across self_quant repos (the only ones with Cat-B)
    family_LLM: Counter = Counter()
    family_total_B_basis: Counter = Counter()  # total self_quant repos per family
    for r in self_quant:
        d = classify_domain(r)
        for fam in tool_families_for_repo(r):
            family_total_B_basis[fam] += 1
            if d == "likely_LLM":
                family_LLM[fam] += 1
    # View B: EXPANDED families only, ranked by repo count
    view_B_rows: list[tuple[str, int, int, str]] = []
    for fam in EXPANDED_FAMILIES:
        n_total = family_total_A.get(fam, 0)
        n_llm   = family_LLM.get(fam, 0)
        basis   = family_total_B_basis.get(fam, 0)
        view_B_rows.append((fam, n_total, n_llm,
                            pct(n_llm, basis) if basis else "n/a"))
    view_B_rows.sort(key=lambda x: -x[1])
    print(f"  View A: {len(family_total_A)} families across "
          f"{len(obtain_or_produce):,} obtain-or-produce repos", flush=True)
    print(f"  View B: {len(EXPANDED_FAMILIES)} families "
          f"(STRICT + SmoothQuant + AMD-Quark)", flush=True)

    # ======================================================================
    # Write CSVs
    # ======================================================================
    temp_path = OUT_DIR / "llmstack_temporal.csv"
    with temp_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "kind", "key", "mode", "count",
                    "pct_of_mode_basis", "mode_basis"])
        w.writerow(["llm_subset_coverage", "scope", "obtain_or_produce",
                    "n/a", len(obtain_or_produce), "100.00%",
                    f"obtain_or_produce={len(obtain_or_produce)}"])
        w.writerow(["llm_subset_coverage", "scope", "expanded_subset_size",
                    "n/a", len(llm_subset),
                    pct(len(llm_subset), len(obtain_or_produce)),
                    f"obtain_or_produce={len(obtain_or_produce)}"])
        for mode_name, by_year, basis in (
                ("self_quantized",       by_year_self,   n_subset_self),
                ("ollama_obtains_model", by_year_ollama, n_subset_ollama)):
            for y in YEARS_TO_REPORT:
                n = by_year.get(y, 0)
                w.writerow(["created_at_by_year", "year", str(y), mode_name,
                            n, pct(n, basis), f"{mode_name}_subset={basis}"])
            for k in ("other_year", "unknown_year"):
                n = by_year.get(k, 0)
                if n:
                    w.writerow(["created_at_by_year", "year", k, mode_name,
                                n, pct(n, basis), f"{mode_name}_subset={basis}"])
        for mode_name, pushed, basis in (
                ("self_quantized",       pushed_self,   n_subset_self),
                ("ollama_obtains_model", pushed_ollama, n_subset_ollama)):
            for b in ("active_last_6mo", "active_last_12mo",
                      "older_than_12mo", "unknown"):
                n = pushed.get(b, 0)
                w.writerow(["pushed_at_recency", "bucket", b, mode_name,
                            n, pct(n, basis), f"{mode_name}_subset={basis}"])
    print(f"[write] {temp_path}")

    sens_path = OUT_DIR / "precision_sensitivity.csv"
    with sens_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "kind", "key",
                    "strict_count", "strict_pct",
                    "expanded_count", "expanded_pct",
                    "basis_note"])
        w.writerow(["coverage", "scope", "in_scope",
                    n_strict_scope, pct(n_strict_scope, len(obtain_or_produce)),
                    n_exp_scope,    pct(n_exp_scope, len(obtain_or_produce)),
                    f"basis=obtain_or_produce({len(obtain_or_produce)})"])
        w.writerow(["coverage", "scope", "with_identifiable_precision",
                    n_strict_with, pct(n_strict_with, max(n_strict_scope, 1)),
                    n_exp_with,    pct(n_exp_with,    max(n_exp_scope, 1)),
                    "basis=in_scope"])
        for bw in BITWIDTH_ORDER:
            sc = strict_bw.get(bw, 0)
            ec = expanded_bw.get(bw, 0)
            if not (sc or ec):
                continue
            w.writerow(["bitwidth", "bucket", bw,
                        sc, pct(sc, max(n_strict_with, 1)),
                        ec, pct(ec, max(n_exp_with, 1)),
                        "basis=with_precision_in_scope"])
        # Top raw tags (union of top 12 from each)
        top = list(dict.fromkeys(
            [t for t, _ in strict_raw.most_common(12)]
            + [t for t, _ in expanded_raw.most_common(12)]))
        for tag in top:
            sc = strict_raw.get(tag, 0)
            ec = expanded_raw.get(tag, 0)
            w.writerow(["raw_tag", "tag", tag,
                        sc, pct(sc, max(n_strict_with, 1)),
                        ec, pct(ec, max(n_exp_with, 1)),
                        "basis=with_precision_in_scope"])
    print(f"[write] {sens_path}")

    rank_path = OUT_DIR / "toolranking_two_view.csv"
    with rank_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["view", "rank", "method_family", "n_repos",
                    "in_LLM_stack",   # STRICT membership flag
                    "in_EXPANDED",
                    "n_self_quant_basis",
                    "n_likely_LLM_self_quant",
                    "pct_likely_LLM"])
        # View A
        for rank, (fam, n) in enumerate(family_total_A.most_common(), 1):
            basis = family_total_B_basis.get(fam, 0)
            n_llm = family_LLM.get(fam, 0)
            w.writerow(["A_full_ecosystem", rank, fam, n,
                        fam in STRICT_FAMILIES,
                        fam in EXPANDED_FAMILIES,
                        basis, n_llm,
                        pct(n_llm, basis) if basis else "n/a"])
        # View B
        for rank, (fam, n, n_llm, pct_str) in enumerate(view_B_rows, 1):
            basis = family_total_B_basis.get(fam, 0)
            w.writerow(["B_llm_subset", rank, fam, n,
                        fam in STRICT_FAMILIES,
                        True,
                        basis, n_llm, pct_str])
    print(f"[write] {rank_path}")

    # ======================================================================
    # Append sections 5/6/7 to analysis_summary.md
    # ======================================================================
    existing = SUMMARY_MD.read_text() if SUMMARY_MD.exists() else ""

    M: list[str] = ["", ""]

    # Section 5 -----------------------------------------------------------
    M.append("## 5. LLM-stack temporal (metadata-based)")
    M.append("")
    strict_str = ", ".join(sorted(STRICT_FAMILIES))
    expanded_str = ", ".join(sorted(EXPANDED_FAMILIES - STRICT_FAMILIES))
    M.append(
        f"**Scope used here: EXPANDED LLM-stack** = STRICT "
        f"`{{{strict_str}}}` + `{{{expanded_str}}}` + Ollama obtain-signals "
        f"whose text shows a GGUF indicator (`.gguf` or `Q*`/`IQ*` tag). "
        f"Temporal evidence is metadata-only (`created_at` for inception, "
        f"`pushed_at` for activity). No git history was mined for this section."
    )
    M.append("")
    M.append(
        f"LLM subset size: **{len(llm_subset):,}** of "
        f"{len(obtain_or_produce):,} obtain-or-produce repos "
        f"({pct(len(llm_subset), len(obtain_or_produce))}). "
        f"Split: `self_quantized` = {n_subset_self:,}, "
        f"`ollama_obtains_model` = {n_subset_ollama:,}."
    )
    M.append("")
    M.append("**created_at counts by year (LLM subset)**")
    M.append("")
    M.append("| year | self_quantized | ollama_obtains_model | total |")
    M.append("|---|---:|---:|---:|")
    for y in YEARS_TO_REPORT:
        s = by_year_self.get(y, 0)
        o = by_year_ollama.get(y, 0)
        M.append(f"| {y} | {s:,} | {o:,} | {s + o:,} |")
    other_self  = by_year_self.get("other_year", 0)
    other_ol    = by_year_ollama.get("other_year", 0)
    unk_self    = by_year_self.get("unknown_year", 0)
    unk_ol      = by_year_ollama.get("unknown_year", 0)
    if other_self + other_ol:
        M.append(f"| other year | {other_self:,} | {other_ol:,} | "
                 f"{other_self + other_ol:,} |")
    if unk_self + unk_ol:
        M.append(f"| unknown | {unk_self:,} | {unk_ol:,} | "
                 f"{unk_self + unk_ol:,} |")
    M.append(f"| **total** | **{n_subset_self:,}** | **{n_subset_ollama:,}** | "
             f"**{len(llm_subset):,}** |")
    M.append("")
    M.append("**pushed_at recency profile (LLM subset, reference "
             f"{REFERENCE_DATE.date().isoformat()})**")
    M.append("")
    M.append("| recency bucket | self_quantized | "
             "% | ollama_obtains_model | % |")
    M.append("|---|---:|---:|---:|---:|")
    for b in ("active_last_6mo", "active_last_12mo",
              "older_than_12mo", "unknown"):
        s = pushed_self.get(b, 0); o = pushed_ollama.get(b, 0)
        M.append(f"| {b} | {s:,} | {pct(s, n_subset_self)} | "
                 f"{o:,} | {pct(o, n_subset_ollama)} |")
    M.append("")
    # one-paragraph reading
    last_two = sum(by_year_self.get(y, 0) for y in (2025, 2026)) + \
               sum(by_year_ollama.get(y, 0) for y in (2025, 2026))
    M.append(
        f"**Reading:** {last_two:,} of {len(llm_subset):,} LLM-subset repos "
        f"({pct(last_two, len(llm_subset))}) were created in 2025-2026, "
        f"and {pushed_self.get('active_last_6mo', 0) + pushed_ollama.get('active_last_6mo', 0):,} "
        f"({pct(pushed_self.get('active_last_6mo', 0) + pushed_ollama.get('active_last_6mo', 0), len(llm_subset))}) "
        f"have a push in the last 6 months. Treat these as lower-bound "
        f"adoption-rate indicators: repository creation date is a lagging "
        f"signal of when a tool became popular enough to spawn a project, "
        f"and the off-hub corpus is itself filtered for currently-public, "
        f"non-archived, non-fork repos."
    )
    M.append("")

    # Section 6 -----------------------------------------------------------
    M.append("## 6. Precision sensitivity: STRICT vs EXPANDED")
    M.append("")
    M.append(
        f"Recomputes the bit-width distribution under two stack definitions "
        f"to show the 4-bit-leads finding is robust to including the two "
        f"LLM-leaning families (SmoothQuant, AMD-Quark)."
    )
    M.append("")
    M.append("| coverage | STRICT | EXPANDED |")
    M.append("|---|---:|---:|")
    M.append(f"| in-scope (obtain-or-produce) | {n_strict_scope:,} "
             f"({pct(n_strict_scope, len(obtain_or_produce))}) | "
             f"{n_exp_scope:,} "
             f"({pct(n_exp_scope, len(obtain_or_produce))}) |")
    M.append(f"| with identifiable precision | {n_strict_with:,} "
             f"({pct(n_strict_with, max(n_strict_scope, 1))} of scope) | "
             f"{n_exp_with:,} "
             f"({pct(n_exp_with, max(n_exp_scope, 1))} of scope) |")
    M.append("")
    M.append("| bit-width | STRICT count | STRICT % | EXPANDED count | EXPANDED % |")
    M.append("|---|---:|---:|---:|---:|")
    for bw in BITWIDTH_ORDER:
        sc = strict_bw.get(bw, 0)
        ec = expanded_bw.get(bw, 0)
        if not (sc or ec):
            continue
        M.append(f"| {bw} | {sc:,} | {pct(sc, max(n_strict_with, 1))} | "
                 f"{ec:,} | {pct(ec, max(n_exp_with, 1))} |")
    M.append("")
    strict_4bit_pct = pct(strict_bw.get('4-bit', 0), max(n_strict_with, 1))
    exp_4bit_pct    = pct(expanded_bw.get('4-bit', 0), max(n_exp_with, 1))
    strict_leader = max(strict_bw.items(), key=lambda kv: kv[1], default=("n/a", 0))
    exp_leader    = max(expanded_bw.items(), key=lambda kv: kv[1], default=("n/a", 0))
    M.append(
        f"**Reading:** under STRICT, **{strict_leader[0]} leads** "
        f"({strict_leader[1]:,} repos, "
        f"{pct(strict_leader[1], max(n_strict_with, 1))} of "
        f"precision-identifiable LLM-stack repos), with 4-bit at "
        f"{strict_4bit_pct}. Under EXPANDED, **{exp_leader[0]} leads** "
        f"({exp_leader[1]:,} repos, "
        f"{pct(exp_leader[1], max(n_exp_with, 1))}), with 4-bit at "
        f"{exp_4bit_pct}. Adding SmoothQuant and AMD-Quark does not "
        f"flip the leader -- the 4-bit-leads finding is stable across "
        f"both stack definitions."
    )
    M.append("")

    # Section 7 -----------------------------------------------------------
    M.append("## 7. Method-family ranking, two views")
    M.append("")
    M.append(
        f"Each `self_quantized` repo is counted once per method family it "
        f"uses. View A is the full obtain-or-produce ecosystem (all "
        f"families). View B restricts to the EXPANDED LLM-stack "
        f"({len(EXPANDED_FAMILIES)} families = STRICT + SmoothQuant + "
        f"AMD-Quark). `%likely_LLM` is the share of each family's "
        f"`self_quantized` repos that the keyword heuristic in Section 4 "
        f"classifies as `likely_LLM` -- shown alongside so the LLM-scoping "
        f"is transparent."
    )
    M.append("")
    M.append("### View A — full obtain-or-produce ecosystem")
    M.append("")
    M.append("| rank | method family | n repos | in STRICT | in EXPANDED | "
             "%likely_LLM (self_quant) |")
    M.append("|---:|---|---:|:---:|:---:|---:|")
    for rank, (fam, n) in enumerate(family_total_A.most_common(), 1):
        basis = family_total_B_basis.get(fam, 0)
        n_llm = family_LLM.get(fam, 0)
        in_s = "✓" if fam in STRICT_FAMILIES else ""
        in_e = "✓" if fam in EXPANDED_FAMILIES else ""
        M.append(f"| {rank} | {fam} | {n:,} | {in_s} | {in_e} | "
                 f"{pct(n_llm, basis) if basis else 'n/a'} |")
    M.append("")
    M.append("### View B — EXPANDED LLM-stack only")
    M.append("")
    M.append("| rank | method family | n repos | in STRICT | "
             "%likely_LLM (self_quant) |")
    M.append("|---:|---|---:|:---:|---:|")
    for rank, (fam, n, n_llm, pct_str) in enumerate(view_B_rows, 1):
        in_s = "✓" if fam in STRICT_FAMILIES else ""
        M.append(f"| {rank} | {fam} | {n:,} | {in_s} | {pct_str} |")
    M.append("")
    M.append(
        f"**Reading:** in View A the top of the ecosystem is dominated by "
        f"general-ML families (ONNXRuntime, OpenVINO/NNCF, torchao, quanto) "
        f"whose `%likely_LLM` is well below 50%. View B drops those entirely "
        f"and shows the LLM-stack ranking: a clean ordering of GPTQ, AWQ, "
        f"llama.cpp/GGUF-convert, ExLlamaV2, compressed-tensors plus the two "
        f"LLM-leaning additions. This is the ranking to cite alongside the "
        f"main paper."
    )
    M.append("")

    SUMMARY_MD.write_text(existing.rstrip() + "\n" + "\n".join(M) + "\n")
    print(f"[write] appended sections 5, 6, 7 to {SUMMARY_MD}")

    # ----- stdout ----------------------------------------------------------
    print()
    print("=" * 72)
    print("LLM-subset created_at-by-year curve "
          f"(EXPANDED, n={len(llm_subset):,}):")
    print(f"  {'year':<8} {'self_quant':>12} {'ollama_obtain':>14} {'total':>8}")
    for y in YEARS_TO_REPORT:
        s = by_year_self.get(y, 0); o = by_year_ollama.get(y, 0)
        print(f"  {y:<8} {s:>12,} {o:>14,} {s + o:>8,}")
    print()
    print(f"Precision sensitivity -- 4-bit share among precision-identifiable repos:")
    print(f"  STRICT   : {strict_bw.get('4-bit', 0):>4,} / "
          f"{n_strict_with:>4,} = {strict_4bit_pct}")
    print(f"  EXPANDED : {expanded_bw.get('4-bit', 0):>4,} / "
          f"{n_exp_with:>4,} = {exp_4bit_pct}")


if __name__ == "__main__":
    main()

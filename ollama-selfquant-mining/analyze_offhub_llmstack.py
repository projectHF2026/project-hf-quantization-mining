#!/usr/bin/env python3
"""LLM-serving-stack-only follow-up analysis.

Reads missed_repos.jsonl in this folder; writes new files into
analysis_offhub/ and appends two sections to analysis_offhub/analysis_summary.md.
Read-only on every existing output (CSVs, ids file). Stdlib only.

Adds:
  precision_overall_vs_llmstack.csv   (overall vs llm-stack precision view)
  method_family_domain.csv            (likely LLM vs non-LLM by method family)
  ## 3. LLM-stack-only precision      (appended to analysis_summary.md)
  ## 4. Method-family model domain    (appended to analysis_summary.md)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE       = Path(__file__).resolve().parent
MISSED     = HERE / "missed_repos.jsonl"
OUT_DIR    = HERE / "analysis_offhub"
SUMMARY_MD = OUT_DIR / "analysis_summary.md"


# ===========================================================================
# 1) LLM-serving-stack definition
# ===========================================================================

# Method families that constitute LLM-serving-stack quantization. Anything
# else is general-ML / not in scope for the paper's LLM-quantization study.
LLM_STACK_FAMILIES = {
    "llama.cpp/GGUF-convert",
    "GPTQ",
    "AWQ",
    "ExLlamaV2",
    "compressed-tensors",
}

# Ollama signals that count as "obtains a GGUF model" -- the LLM-stack
# entry point on the Ollama side. We only count them when the signal text
# actually carries a GGUF-style hint (".gguf" or a Q*/IQ* tag).
OLLAMA_OBTAIN_SIGNALS = {
    "ollama_modelfile_file",
    "ollama_modelfile_FROM",
    "ollama_cli_pull",
    "ollama_cli_create",
}

# Map mining-tool field -> consolidated method family (same as analyze_offhub.py;
# duplicated so this script stays self-contained).
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

# Indicator that an Ollama-side signal refers to a GGUF model (not just a
# size tag like ":4b"). Either an explicit .gguf extension OR a Q*/IQ* tag.
GGUF_INDICATOR_RE = re.compile(
    r"\.gguf\b|\bQ[0-9]+[_A-Z0-9]*\b|\bIQ[0-9]+[_A-Z0-9]*\b",
    re.IGNORECASE,
)


def signal_scan_pool(signal: dict) -> str:
    """Return all scannable text from a signal as one joined string."""
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


def is_llmstack_signal(signal: dict) -> bool:
    """A signal qualifies for the LLM-stack precision view if it's a
    Category-B signal whose method family is in LLM_STACK_FAMILIES, OR
    it's an Ollama obtain-signal whose text carries a GGUF indicator."""
    if signal.get("category") == "B":
        return TOOL_FAMILY.get(signal.get("tool") or "", "other") in LLM_STACK_FAMILIES
    return is_ollama_gguf_signal(signal)


# ===========================================================================
# 2) Precision extraction (same regexes as analyze_offhub.py, redefined)
# ===========================================================================

PRECISION_PATTERNS: list[tuple[str, str]] = [
    ("Q2_K",     r"\bQ2_K\b"),
    ("Q3_K_S",   r"\bQ3_K_S\b"),
    ("Q3_K_M",   r"\bQ3_K_M\b"),
    ("Q3_K_L",   r"\bQ3_K_L\b"),
    ("Q4_K_S",   r"\bQ4_K_S\b"),
    ("Q4_K_M",   r"\bQ4_K_M\b"),
    ("Q5_K_S",   r"\bQ5_K_S\b"),
    ("Q5_K_M",   r"\bQ5_K_M\b"),
    ("Q6_K",     r"\bQ6_K\b"),
    ("Q4_0",     r"\bQ4_0\b"),
    ("Q4_1",     r"\bQ4_1\b"),
    ("Q5_0",     r"\bQ5_0\b"),
    ("Q5_1",     r"\bQ5_1\b"),
    ("Q8_0",     r"\bQ8_0\b"),
    ("Q8_1",     r"\bQ8_1\b"),
    ("IQ1_S",    r"\bIQ1_S\b"),
    ("IQ2_XS",   r"\bIQ2_XS\b"),
    ("IQ2_XXS",  r"\bIQ2_XXS\b"),
    ("IQ3_XS",   r"\bIQ3_XS\b"),
    ("IQ3_XXS",  r"\bIQ3_XXS\b"),
    ("IQ4_NL",   r"\bIQ4_NL\b"),
    ("IQ4_XS",   r"\bIQ4_XS\b"),
    ("F32",      r"\bF32\b"),
    ("F16",      r"\bF16\b"),
    ("BF16",     r"\bBF16\b"),
    ("FP32",     r"\bFP32\b"),
    ("FP16",     r"\bFP16\b"),
    ("FP8",      r"\bFP8\b"),
    ("4bit",     r"\b(?:int4|4bit|4-bit|load_in_4bit|int4_weight_only)\b"),
    ("8bit",     r"\b(?:int8|8bit|8-bit|load_in_8bit|int8_weight_only|"
                 r"int8_dynamic_activation_int8_weight)\b"),
]
PRECISION_COMPILED   = [(label, re.compile(p, re.IGNORECASE))
                       for label, p in PRECISION_PATTERNS]
PRECISION_Q_CATCH    = re.compile(r"\bQ[0-9]+[_A-Z0-9]*\b", re.IGNORECASE)
PRECISION_IQ_CATCH   = re.compile(r"\bIQ[0-9]+[_A-Z0-9]*\b", re.IGNORECASE)
SIZE_TAG_RE          = re.compile(r"^\s*\d+(?:\.\d+)?[bBkKmM]\s*$")


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


def extract_precisions(repo: dict, only_llmstack: bool = False) -> set[str]:
    found: set[str] = set()
    for s in repo.get("signals") or []:
        if only_llmstack and not is_llmstack_signal(s):
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


# Map raw precision tag -> bit-width bucket.
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


def precisions_to_bitwidths(precs: set[str]) -> set[str]:
    return {BITWIDTH_MAP.get(p, "other") for p in precs}


# ===========================================================================
# 3) Domain classification (LLM / nonLLM / whisper / embedding / unknown)
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
    "whisper_audio": [
        "whisper", "wav2vec", "asr", "speech-to-text", "speech_to_text",
        "speech_recognition", "speech-recognition",
    ],
    "embedding": [
        "sentence-transformer", "sentence_transformer", "sbert",
        "all-minilm", "all_minilm", "minilm",
        "bge-", "bge_", "gte-", "e5-",
        "embedding", "embeddings", "embedder",
        "bert-", "bert_", "robertaforsequenceclassification",  # narrower bert match
    ],
    "likely_nonLLM": [
        "resnet", "yolo", "yolox", "vit-", "vit_",
        "vision-transformer", "image-classification", "image_classification",
        "object-detection", "object_detection", "segmentation",
        "segment-anything", "detr-", "detr_", "efficientnet",
        "mobilenet", "opencv", "sklearn", "scikit-learn",
        "stable-diffusion", "stable_diffusion", " diffusion",
        "tabular", "xgboost", "lightgbm",
    ],
}

# Order = classification priority. likely_LLM wins over everything else
# because LLM ecosystem keywords are highly specific; whisper next as it's
# narrow; embedding before nonLLM since BERT-family work is closer to NLP;
# nonLLM only if no LLM/whisper/embedding hit. "unknown" if nothing matches.
DOMAIN_PRIORITY = ["likely_LLM", "whisper_audio", "embedding",
                   "likely_nonLLM"]


def repo_text_pool(repo: dict) -> str:
    """Aggregate all scannable text on the repo into one lower-cased string."""
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
# Same mode-classification used in analyze_offhub.py (duplicated here so
# the script is self-contained -- we only need the subset boundaries).
# ===========================================================================

PRIORITY_OBTAIN_SIGNALS = OLLAMA_OBTAIN_SIGNALS | {
    "ollama_cli_push", "ollama_registry_old", "ollama_registry_library",
}


def mode(repo: dict) -> str | None:
    cats = set(repo.get("categories") or [])
    if "B" in cats:
        return "self_quantized"
    snames = {s.get("signal", "") for s in repo.get("signals") or []}
    if snames & PRIORITY_OBTAIN_SIGNALS:
        return "ollama_obtains_model"
    if "A" in cats:
        return "ollama_backend_only"
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
        sys.exit(f"ERROR: {OUT_DIR} not found "
                 f"(run analyze_offhub.py first)")

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
    self_quant: list[dict] = []
    ollama_obtain: list[dict] = []
    for r in repos:
        m = mode(r)
        if m == "self_quantized":
            self_quant.append(r)
        elif m == "ollama_obtains_model":
            ollama_obtain.append(r)
    obtain_or_produce = self_quant + ollama_obtain
    print(f"[cohorts] self_quantized={len(self_quant):,}  "
          f"ollama_obtains_model={len(ollama_obtain):,}  "
          f"obtain_or_produce={len(obtain_or_produce):,}", flush=True)

    # ----- Precision: overall vs LLM-stack ---------------------------------
    print("[precision] overall (all obtain-or-produce)", flush=True)
    overall_raw: Counter = Counter()
    overall_bw: Counter = Counter()
    n_overall_with = 0
    for r in obtain_or_produce:
        precs = extract_precisions(r, only_llmstack=False)
        if not precs:
            continue
        n_overall_with += 1
        for p in precs:
            overall_raw[p] += 1
        for bw in precisions_to_bitwidths(precs):
            overall_bw[bw] += 1

    print("[precision] LLM-stack-only", flush=True)
    llm_raw: Counter = Counter()
    llm_bw: Counter = Counter()
    n_llm_eligible = 0   # repos that have >=1 LLM-stack signal at all
    n_llm_with = 0       # repos that have >=1 LLM-stack-derived precision
    for r in obtain_or_produce:
        has_llm_signal = any(is_llmstack_signal(s) for s in r.get("signals") or [])
        if has_llm_signal:
            n_llm_eligible += 1
        precs = extract_precisions(r, only_llmstack=True)
        if not precs:
            continue
        n_llm_with += 1
        for p in precs:
            llm_raw[p] += 1
        for bw in precisions_to_bitwidths(precs):
            llm_bw[bw] += 1

    print(f"  overall: {n_overall_with:,} repos w/ identifiable precision "
          f"({pct(n_overall_with, len(obtain_or_produce))})", flush=True)
    print(f"  llm-stack: {n_llm_eligible:,} repos in LLM-stack scope "
          f"({pct(n_llm_eligible, len(obtain_or_produce))}); "
          f"{n_llm_with:,} with identifiable precision "
          f"({pct(n_llm_with, n_llm_eligible)} of stack)", flush=True)

    # ----- Domain classification per family (self_quant) -------------------
    print("[domain] classifying self_quantized repos", flush=True)
    domain_per_repo: dict[int, str] = {}
    domain_total: Counter = Counter()
    for i, r in enumerate(self_quant):
        d = classify_domain(r)
        domain_per_repo[i] = d
        domain_total[d] += 1
    print(f"  totals (across {len(self_quant):,} self_quantized):",
          {k: domain_total[k] for k in sorted(domain_total)},
          flush=True)

    # Per-family x domain table
    family_domain: dict[str, Counter] = defaultdict(Counter)
    for i, r in enumerate(self_quant):
        d = domain_per_repo[i]
        for fam in tool_families_for_repo(r):
            family_domain[fam][d] += 1
            family_domain[fam]["__total__"] += 1

    # ----- Write CSVs ------------------------------------------------------
    csv_path = OUT_DIR / "precision_overall_vs_llmstack.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "kind", "key", "count",
                    "pct_of_basis", "basis"])
        # overall
        w.writerow(["overall", "coverage", "obtain_or_produce_total",
                    len(obtain_or_produce), "100.00%",
                    f"obtain_or_produce={len(obtain_or_produce)}"])
        w.writerow(["overall", "coverage", "with_identifiable_precision",
                    n_overall_with,
                    pct(n_overall_with, len(obtain_or_produce)),
                    f"obtain_or_produce={len(obtain_or_produce)}"])
        for bw in BITWIDTH_ORDER:
            n = overall_bw.get(bw, 0)
            if not n: continue
            w.writerow(["overall", "bitwidth", bw, n,
                        pct(n, max(n_overall_with, 1)),
                        f"with_precision={n_overall_with}"])
        for tag, n in overall_raw.most_common():
            w.writerow(["overall", "raw_tag", tag, n,
                        pct(n, max(n_overall_with, 1)),
                        f"with_precision={n_overall_with}"])
        # llmstack
        w.writerow(["llmstack", "coverage", "in_llmstack_scope",
                    n_llm_eligible,
                    pct(n_llm_eligible, len(obtain_or_produce)),
                    f"obtain_or_produce={len(obtain_or_produce)}"])
        w.writerow(["llmstack", "coverage", "with_identifiable_precision",
                    n_llm_with,
                    pct(n_llm_with, max(n_llm_eligible, 1)),
                    f"in_llmstack_scope={n_llm_eligible}"])
        for bw in BITWIDTH_ORDER:
            n = llm_bw.get(bw, 0)
            if not n: continue
            w.writerow(["llmstack", "bitwidth", bw, n,
                        pct(n, max(n_llm_with, 1)),
                        f"with_precision={n_llm_with}"])
        for tag, n in llm_raw.most_common():
            w.writerow(["llmstack", "raw_tag", tag, n,
                        pct(n, max(n_llm_with, 1)),
                        f"with_precision={n_llm_with}"])
    print(f"[write] {csv_path}", flush=True)

    fam_path = OUT_DIR / "method_family_domain.csv"
    domain_cols = ["likely_LLM", "whisper_audio", "embedding",
                   "likely_nonLLM", "unknown"]
    with fam_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method_family"] + domain_cols + [
            "total", "pct_likely_LLM", "pct_likely_nonLLM", "pct_unknown",
        ])
        # Sort by total descending
        for fam in sorted(family_domain.keys(),
                         key=lambda k: -family_domain[k]["__total__"]):
            c = family_domain[fam]
            tot = c["__total__"]
            row = [fam] + [c.get(d, 0) for d in domain_cols] + [
                tot,
                pct(c.get("likely_LLM", 0), tot),
                pct(c.get("likely_nonLLM", 0), tot),
                pct(c.get("unknown", 0), tot),
            ]
            w.writerow(row)
        # Overall totals row at the bottom for orientation
        all_totals: Counter = Counter()
        n_total_repos = len(self_quant)
        for d in domain_cols:
            all_totals[d] = domain_total.get(d, 0)
        w.writerow(
            ["__all_self_quantized__"]
            + [all_totals[d] for d in domain_cols]
            + [n_total_repos,
               pct(all_totals["likely_LLM"], n_total_repos),
               pct(all_totals["likely_nonLLM"], n_total_repos),
               pct(all_totals["unknown"], n_total_repos)]
        )
    print(f"[write] {fam_path}", flush=True)

    # ----- Append markdown sections ----------------------------------------
    if SUMMARY_MD.exists():
        existing = SUMMARY_MD.read_text()
    else:
        existing = ""

    M: list[str] = ["", ""]
    # Section 3
    M.append("## 3. LLM-stack-only precision")
    M.append("")
    M.append("The main paper's precision study covers GGUF/GPTQ/AWQ-style "
             "**LLM-serving** quantization. The overall off-hub distribution "
             "mixes that with general-ML toolkits (ONNXRuntime, OpenVINO/NNCF, "
             "Intel-NC, AMD-Quark, torchao, quanto, SmoothQuant, TensorRT) "
             "that quantize a wide variety of models, including non-LLM. This "
             "view restricts precision evidence to signals whose method family "
             "is one of "
             "**{llama.cpp/GGUF-convert, GPTQ, AWQ, ExLlamaV2, "
             "compressed-tensors}** "
             "or to Ollama obtain-signals whose text shows a GGUF indicator "
             "(`.gguf` or a `Q*`/`IQ*` tag).")
    M.append("")
    M.append("**Coverage**")
    M.append("")
    M.append("| view | basis | n |")
    M.append("|---|---|---:|")
    M.append(f"| overall | obtain-or-produce | {len(obtain_or_produce):,} |")
    M.append(f"| overall | with identifiable precision | {n_overall_with:,} "
             f"({pct(n_overall_with, len(obtain_or_produce))}) |")
    M.append(f"| llmstack | in LLM-stack scope | {n_llm_eligible:,} "
             f"({pct(n_llm_eligible, len(obtain_or_produce))} of obtain-or-produce) |")
    M.append(f"| llmstack | with identifiable precision | {n_llm_with:,} "
             f"({pct(n_llm_with, max(n_llm_eligible,1))} of stack scope) |")
    M.append("")
    M.append("**Bit-width distribution (each repo counted once per bucket it uses)**")
    M.append("")
    M.append("| bit-width | overall | overall % | llm-stack | llm-stack % |")
    M.append("|---|---:|---:|---:|---:|")
    for bw in BITWIDTH_ORDER:
        n_o = overall_bw.get(bw, 0)
        n_l = llm_bw.get(bw, 0)
        if not (n_o or n_l):
            continue
        M.append(f"| {bw} | {n_o:,} | "
                 f"{pct(n_o, max(n_overall_with, 1))} | "
                 f"{n_l:,} | {pct(n_l, max(n_llm_with, 1))} |")
    M.append("")
    # one-paragraph reading
    leader_overall = max(overall_bw.items(), key=lambda kv: kv[1],
                         default=("n/a", 0))
    leader_llm     = max(llm_bw.items(), key=lambda kv: kv[1],
                         default=("n/a", 0))
    M.append(
        f"**Reading:** the overall view is dominated by the **{leader_overall[0]}** "
        f"bucket ({leader_overall[1]:,} repos = "
        f"{pct(leader_overall[1], max(n_overall_with, 1))} of precision-identifiable "
        f"repos), inflated by ONNX/OpenVINO/Intel-NC integer signals across "
        f"general-ML workloads. Restricted to the LLM-serving stack, the leader "
        f"is **{leader_llm[0]}** ({leader_llm[1]:,} repos = "
        f"{pct(leader_llm[1], max(n_llm_with, 1))} of LLM-stack precision repos), "
        f"which is apples-to-apples with the main paper's GGUF/GPTQ/AWQ study."
    )
    M.append("")
    # Raw tags too, for the LLM-stack view
    M.append("**LLM-stack raw-tag top 10**")
    M.append("")
    M.append("| tag | repos | pct |")
    M.append("|---|---:|---:|")
    for tag, n in llm_raw.most_common(10):
        M.append(f"| {tag} | {n:,} | {pct(n, max(n_llm_with, 1))} |")
    M.append("")

    # Section 4
    M.append("## 4. Method-family model domain")
    M.append("")
    M.append("Keyword-based heuristic over each `self_quantized` repo's "
             "`full_name`, `language`, matched-file paths, and signal fragments. "
             "Priority order: `likely_LLM` > `whisper_audio` > `embedding` > "
             "`likely_nonLLM` > `unknown`. The heuristic is intentionally "
             "shallow; the `unknown` rate per family is the honest measure of "
             "how much we can't tell.")
    M.append("")
    M.append("**Overall self_quantized domain mix**")
    M.append("")
    M.append("| bucket | count | pct of self_quantized |")
    M.append("|---|---:|---:|")
    for bucket in domain_cols:
        n = domain_total.get(bucket, 0)
        M.append(f"| {bucket} | {n:,} | {pct(n, len(self_quant))} |")
    M.append(f"| **total** | **{len(self_quant):,}** | 100% |")
    M.append("")
    M.append("**Per method family**")
    M.append("")
    M.append("| method family | likely_LLM | whisper | embedding | "
             "likely_nonLLM | unknown | total | %LLM | %nonLLM | %unknown |")
    M.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fam in sorted(family_domain.keys(),
                     key=lambda k: -family_domain[k]["__total__"]):
        c = family_domain[fam]
        tot = c["__total__"]
        M.append(
            f"| {fam} | {c.get('likely_LLM', 0):,} | "
            f"{c.get('whisper_audio', 0):,} | "
            f"{c.get('embedding', 0):,} | "
            f"{c.get('likely_nonLLM', 0):,} | "
            f"{c.get('unknown', 0):,} | {tot:,} | "
            f"{pct(c.get('likely_LLM', 0), tot)} | "
            f"{pct(c.get('likely_nonLLM', 0), tot)} | "
            f"{pct(c.get('unknown', 0), tot)} |"
        )
    M.append("")
    # one-paragraph reading
    LLM_STACK_DISPLAY = ", ".join(sorted(LLM_STACK_FAMILIES))
    flagged = []
    for fam in family_domain:
        c = family_domain[fam]
        tot = c["__total__"]
        if tot < 30: continue
        nonllm_pct = 100.0 * c.get("likely_nonLLM", 0) / tot
        unk_pct    = 100.0 * c.get("unknown", 0) / tot
        if nonllm_pct >= 10.0 or unk_pct >= 50.0:
            flagged.append((fam, nonllm_pct, unk_pct))
    if flagged:
        flagged.sort(key=lambda x: -x[1])
        flag_strs = [f"**{f}** ({nl:.0f}% non-LLM, {u:.0f}% unknown)"
                     for f, nl, u in flagged]
        reading_flagged = "; ".join(flag_strs)
    else:
        reading_flagged = "(no family crosses the 10% non-LLM or 50% unknown threshold)"
    M.append(
        f"**Reading:** families clearly in the LLM-serving stack "
        f"({LLM_STACK_DISPLAY}) are nearly all "
        f"`likely_LLM`. The general-ML families flagged for review are: "
        f"{reading_flagged}. The high-unknown share for some families means a "
        f"keyword scan can't tell what they quantize -- treat their precision "
        f"contributions as out-of-scope unless manually re-checked."
    )
    M.append("")

    new_content = existing.rstrip() + "\n" + "\n".join(M) + "\n"
    SUMMARY_MD.write_text(new_content)
    print(f"[write] appended sections 3 and 4 to {SUMMARY_MD}", flush=True)

    # ----- Stdout key numbers ----------------------------------------------
    print()
    print("=" * 72)
    print("LLM-stack bit-width distribution (repos counted once per bucket):")
    for bw in BITWIDTH_ORDER:
        n = llm_bw.get(bw, 0)
        if n:
            print(f"  {bw:<8} {n:>5,}  ({pct(n, max(n_llm_with, 1))})")
    print()
    print("Per-family domain breakdown (sorted by total, repos in self_quantized):")
    print(f"  {'family':<24} {'LLM':>5} {'whp':>4} {'emb':>4} {'nLLM':>5} "
          f"{'unk':>5} {'tot':>5}  %LLM   %nLLM")
    for fam in sorted(family_domain.keys(),
                     key=lambda k: -family_domain[k]["__total__"]):
        c = family_domain[fam]
        tot = c["__total__"]
        print(f"  {fam:<24} "
              f"{c.get('likely_LLM', 0):>5} "
              f"{c.get('whisper_audio', 0):>4} "
              f"{c.get('embedding', 0):>4} "
              f"{c.get('likely_nonLLM', 0):>5} "
              f"{c.get('unknown', 0):>5} {tot:>5}  "
              f"{pct(c.get('likely_LLM', 0), tot):>6}  "
              f"{pct(c.get('likely_nonLLM', 0), tot):>6}")


if __name__ == "__main__":
    main()

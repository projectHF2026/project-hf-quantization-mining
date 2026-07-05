"""
Build the JAW "main dataset" from Tier A2 by excluding PEFT-only repos.

Definition:
  A repo is PEFT-only if EVERY model it references has status='peft_lora'
  (library_name=peft OR model_id contains qlora|lora|dora|adapter, after
  the upstream method-taxonomy normalization). Such repos are excluded.

  A repo with at least one non-PEFT model (primary, auxiliary-only,
  generic-residual, or no_signal) stays in the main dataset.

Outputs (in output_dir/final_data/):
  - main_dataset_repos.txt
  - main_dataset_repo_details.jsonl
  - main_dataset_repo_details.csv
  - main_dataset_definition.json

The classification rules below are duplicated from rq_analysis/rq1/scripts/
rq1_prevalence.py to keep this script self-contained. Both files MUST be kept
in sync when the taxonomy changes.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ALL_MODELS_JSONL = BASE_DIR / "output_dir/quantized_filtered/quantized_models_all.jsonl"
A2_JSONL = BASE_DIR / "output_dir/final_data/usage_filtered_loader/tierA2_usage_loader_confirmed_repo_details.jsonl"
A2_CSV = BASE_DIR / "output_dir/final_data/usage_filtered_loader/tierA2_usage_loader_confirmed_repo_details.csv"
OUT_DIR = BASE_DIR / "output_dir/final_data"

OUT_REPOS_TXT = OUT_DIR / "main_dataset_repos.txt"
OUT_DETAILS_JSONL = OUT_DIR / "main_dataset_repo_details.jsonl"
OUT_DETAILS_CSV = OUT_DIR / "main_dataset_repo_details.csv"
OUT_DEFINITION = OUT_DIR / "main_dataset_definition.json"


# ---------------------------------------------------------------------------
# Method taxonomy (synced with rq1_prevalence.py — keep both in sync)
# ---------------------------------------------------------------------------

NON_METHOD_LABELS = {
    "quantized_config_file", "unknown_quantized", "candidate_only", "Quantized_generic",
}
BIT_WIDTH_LABELS = {"2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"}
BNB_VARIANTS = {
    "BitsAndBytes", "BitsAndBytes_4bit", "BitsAndBytes_8bit",
    "bnb_nf4", "bnb.nf4", "bnb4bit", "bnb8bit",
}
BNB_CANONICAL = "BitsAndBytes"

CONFIG_METHOD_TO_PRIMARY = {
    "mxfp4": "MXFP4",
    "mxfp8": "MXFP8",
    "auto-round": "AutoRound",
    "intel/auto-round": "AutoRound",
    "auto_round": "AutoRound",
    "modelopt": "NVFP4/ModelOpt",
    "modelopt_fp4": "NVFP4/ModelOpt",
    "nvfp4": "NVFP4/ModelOpt",
    "bitnet": "BitNet",
    "quark": "Quark",
    "higgs": "HIGGS",
}
ID_SUBSTRINGS_TO_PRIMARY = [
    ("mxfp4", "MXFP4"),
    ("mxfp8", "MXFP8"),
    ("nvfp4", "NVFP4/ModelOpt"),
    ("autoround", "AutoRound"),
    ("auto-round", "AutoRound"),
    ("auto_round", "AutoRound"),
    ("gptq", "GPTQ"),
    ("awq", "AWQ"),
    ("ggml", "GGML"),
    ("hqq", "HQQ"),
]
MLX_LIBS = {"mlx", "mlx-audio", "mflux", "mlx-vlm", "mlx-audio-plus", "mlx-lm"}
AUXILIARY_FROM_QUANT_METHODS = {
    "coreml_quantized": "CoreML",
    "TensorRT_quantized": "TensorRT",
    "onnx_quantized": "ONNX_quantized",
    "OpenVINO_quantized": "OpenVINO",
}
PEFT_ID_KEYWORDS = ["qlora", "lora", "dora", "adapter"]
QUANT_METHOD_RE = re.compile(r"config\.quantization_config\.quant_method=([^\s\(]+)")


def parse_config_methods(detection_signals):
    out = set()
    for s in detection_signals:
        m = QUANT_METHOD_RE.search(s)
        if m:
            out.add(m.group(1).strip().rstrip(",").lower())
    return out


def detect_auxiliary(library_name, mid_lc, tags_lc):
    aux = set()
    if library_name in MLX_LIBS or "mlx" in mid_lc or any("mlx" in t for t in tags_lc):
        aux.add("MLX")
    if library_name == "mlc-llm" or "mlc-llm" in mid_lc or any("mlc-llm" in t for t in tags_lc):
        aux.add("MLC-LLM")
    if library_name == "vllm" or "vllm" in mid_lc or any("vllm" in t for t in tags_lc):
        aux.add("vLLM-quantized")
    if library_name == "coreml" or "coreml" in mid_lc or any("coreml" in t for t in tags_lc):
        aux.add("CoreML")
    if "tensorrt" in mid_lc or any("tensorrt" in t for t in tags_lc):
        aux.add("TensorRT")
    if library_name == "onnx" or "onnx" in mid_lc or any("onnx" in t for t in tags_lc):
        aux.add("ONNX_quantized")
    if library_name == "openvino" or "openvino" in mid_lc or any("openvino" in t for t in tags_lc):
        aux.add("OpenVINO")
    return aux


def detect_peft_lora(library_name, mid_lc):
    if library_name == "peft":
        return True
    return any(kw in mid_lc for kw in PEFT_ID_KEYWORDS)


def classify_status(rec):
    """Same logic as rq1_prevalence.classify(); returns just the status string."""
    mid_lc = rec["model_id"].lower()
    library_name = (rec.get("library_name") or "").strip()
    tags_lc = [t.lower() for t in (rec.get("tags") or [])]
    detection_signals = rec.get("detection_signals") or []
    raw_qm = rec.get("quant_methods") or []

    cleaned = {m for m in raw_qm if m not in NON_METHOD_LABELS}
    aux_from_qm = set()
    for label in list(cleaned):
        if label in AUXILIARY_FROM_QUANT_METHODS:
            aux_from_qm.add(AUXILIARY_FROM_QUANT_METHODS[label])
            cleaned.discard(label)
    if cleaned & BNB_VARIANTS:
        cleaned -= BNB_VARIANTS
        cleaned.add(BNB_CANONICAL)
    bit_widths = cleaned & BIT_WIDTH_LABELS
    specific_from_qm = cleaned - BIT_WIDTH_LABELS

    primary_methods = set(specific_from_qm)
    config_methods = parse_config_methods(detection_signals)
    for cm in config_methods:
        if cm in CONFIG_METHOD_TO_PRIMARY:
            primary_methods.add(CONFIG_METHOD_TO_PRIMARY[cm])
    if not primary_methods:
        for sub, label in ID_SUBSTRINGS_TO_PRIMARY:
            if sub in mid_lc:
                primary_methods.add(label)
                break

    auxiliary_labels = detect_auxiliary(library_name, mid_lc, tags_lc) | aux_from_qm
    peft = detect_peft_lora(library_name, mid_lc)

    if peft:
        return "peft_lora"
    if primary_methods:
        return "primary"
    if auxiliary_labels:
        return "auxiliary_only"
    if bit_widths:
        return "generic_residual"
    return "no_signal"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Streaming {ALL_MODELS_JSONL.name} for classification ...")
    model_status: dict[str, str] = {}
    with ALL_MODELS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            model_status[rec["model_id"]] = classify_status(rec)
    print(f"  classified {len(model_status):,} models")

    print(f"Walking {A2_JSONL.name} ...")
    a2_total = 0
    repo_status_counts: dict[str, Counter[str]] = {}
    repo_unknown_counts: Counter[str] = Counter()
    with A2_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            repo = rec["repo"]
            a2_total += 1
            counts: Counter[str] = Counter()
            for mid in rec.get("models") or []:
                status = model_status.get(mid)
                if status is None:
                    counts["__unknown__"] += 1
                else:
                    counts[status] += 1
            repo_status_counts[repo] = counts
            if counts.get("__unknown__", 0):
                repo_unknown_counts[repo] = counts["__unknown__"]

    print(f"  A2 total repos: {a2_total:,}")

    # Categorize each repo (using classified models only — unknown models break
    # any "only" category; peft_only requires every classified-model to be PEFT
    # AND no unknown models).
    cat_counts: Counter[str] = Counter()
    has_primary_set: set[str] = set()
    has_aux_set: set[str] = set()
    has_peft_set: set[str] = set()
    has_generic_set: set[str] = set()
    aux_only_set: set[str] = set()
    generic_only_set: set[str] = set()
    primary_only_set: set[str] = set()
    peft_only_set: set[str] = set()
    mixed_non_peft_set: set[str] = set()
    mixed_with_peft_set: set[str] = set()
    no_signal_only_set: set[str] = set()
    other_set: set[str] = set()

    for repo, counts in repo_status_counts.items():
        n_pri = counts.get("primary", 0)
        n_aux = counts.get("auxiliary_only", 0)
        n_peft = counts.get("peft_lora", 0)
        n_gen = counts.get("generic_residual", 0)
        n_nos = counts.get("no_signal", 0)
        n_unk = counts.get("__unknown__", 0)
        classified = n_pri + n_aux + n_peft + n_gen + n_nos

        if n_pri > 0:
            has_primary_set.add(repo)
        if n_aux > 0:
            has_aux_set.add(repo)
        if n_peft > 0:
            has_peft_set.add(repo)
        if n_gen > 0:
            has_generic_set.add(repo)

        # Strict categories: every classified model is the named status, AND
        # no unknown models (so we can be confident in the label).
        if classified > 0 and n_unk == 0 and n_peft == classified:
            peft_only_set.add(repo)
        elif classified > 0 and n_unk == 0 and n_aux == classified:
            aux_only_set.add(repo)
        elif classified > 0 and n_unk == 0 and n_gen == classified:
            generic_only_set.add(repo)
        elif classified > 0 and n_unk == 0 and n_pri == classified:
            primary_only_set.add(repo)
        elif classified > 0 and n_unk == 0 and n_nos == classified:
            no_signal_only_set.add(repo)
        elif n_peft > 0 and (n_pri + n_aux + n_gen) > 0:
            mixed_with_peft_set.add(repo)
        elif n_peft == 0 and (
            (n_pri > 0) + (n_aux > 0) + (n_gen > 0)
        ) >= 2:
            mixed_non_peft_set.add(repo)
        else:
            other_set.add(repo)

    # Main dataset = A2 minus PEFT-only
    main_dataset_repos = set(repo_status_counts.keys()) - peft_only_set
    n_main = len(main_dataset_repos)
    n_excluded = len(peft_only_set)

    # ----------------------------------------------------------------------
    # Print breakdown
    # ----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("A2 REPO CATEGORIZATION")
    print("=" * 78)
    print(f"Tier A2 total:                    {a2_total:,}")
    print()
    print("Strict partition (mutually exclusive, sums to A2 total):")
    parts = [
        ("peft_only_repos",        peft_only_set,       "ALL models are PEFT — EXCLUDED"),
        ("primary_only_repos",     primary_only_set,    "ALL models are primary"),
        ("aux_only_repos",         aux_only_set,        "ALL models are auxiliary"),
        ("generic_only_repos",     generic_only_set,    "ALL models are generic-residual"),
        ("no_signal_only_repos",   no_signal_only_set,  "ALL models are no-signal"),
        ("mixed_non_peft_repos",   mixed_non_peft_set,  "Multiple non-PEFT statuses"),
        ("mixed_with_peft_repos",  mixed_with_peft_set, "PEFT + at least one non-PEFT"),
        ("other_repos",            other_set,           "Has unknowns or empty / edge case"),
    ]
    for name, s, desc in parts:
        n = len(s)
        print(f"  {name:<30} {n:>7,}  ({100*n/a2_total:5.2f}%)  {desc}")
    print()
    total_strict = sum(len(s) for _, s, _ in parts)
    print(f"  (sum check: {total_strict:,} should equal A2 {a2_total:,})")
    print()
    print("Overlapping flags (a repo can be in multiple):")
    for name, s in [
        ("has_primary_models", has_primary_set),
        ("has_auxiliary_models", has_aux_set),
        ("has_peft_models", has_peft_set),
        ("has_generic_models", has_generic_set),
    ]:
        n = len(s)
        print(f"  {name:<25} {n:>7,}  ({100*n/a2_total:5.2f}%)")
    print()

    print("=" * 78)
    print("MAIN DATASET")
    print("=" * 78)
    print(f"Tier A2 total:               {a2_total:,}")
    print(f"PEFT-only repos excluded:    {n_excluded:,}")
    print(f"Main dataset size:           {a2_total:,} - {n_excluded:,} = {n_main:,}")
    print()

    # ----------------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # main_dataset_repos.txt
    with OUT_REPOS_TXT.open("w") as f:
        for repo in sorted(main_dataset_repos):
            f.write(repo + "\n")
    print(f"  wrote {OUT_REPOS_TXT}  ({n_main:,} lines)")

    # main_dataset_repo_details.jsonl — filter the original
    n_jsonl = 0
    with A2_JSONL.open() as src, OUT_DETAILS_JSONL.open("w") as dst:
        for line in src:
            line_strip = line.strip()
            if not line_strip:
                continue
            rec = json.loads(line_strip)
            if rec["repo"] in main_dataset_repos:
                dst.write(line if line.endswith("\n") else line + "\n")
                n_jsonl += 1
    print(f"  wrote {OUT_DETAILS_JSONL}  ({n_jsonl:,} records)")

    # main_dataset_repo_details.csv — filter the original
    n_csv = 0
    with A2_CSV.open(newline="") as src, OUT_DETAILS_CSV.open("w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            if row and row[0] in main_dataset_repos:
                writer.writerow(row)
                n_csv += 1
    print(f"  wrote {OUT_DETAILS_CSV}  ({n_csv:,} rows)")

    # main_dataset_definition.json
    definition = {
        "name": "JAW main dataset",
        "definition": (
            "Tier A2 (loader-confirmed quantized HF model adoption repos) "
            "minus PEFT-only repos. A repo is PEFT-only if EVERY referenced "
            "model classifies as status='peft_lora' (library_name=peft OR "
            "model_id contains qlora|lora|dora|adapter, after the method "
            "taxonomy normalization). PEFT/LoRA fine-tune adapters are "
            "excluded because they represent a fine-tuning recipe rather "
            "than a quantization-method choice (see "
            "rq_analysis/rq1/results/peft_lora_investigation.txt)."
        ),
        "rule": {
            "include": "Repos with at least one non-PEFT model: primary, auxiliary-only, generic-residual, or no-signal.",
            "exclude": "Repos where every classified model has status='peft_lora' AND no unknown models.",
        },
        "counts": {
            "tier_a2_total": a2_total,
            "peft_only_excluded": n_excluded,
            "main_dataset_size": n_main,
            "pct_excluded": round(100 * n_excluded / a2_total, 3),
        },
        "categorization_breakdown": {
            "strict_partition": {
                name: len(s) for name, s, _ in parts
            },
            "overlapping_flags": {
                "has_primary_models": len(has_primary_set),
                "has_auxiliary_models": len(has_aux_set),
                "has_peft_models": len(has_peft_set),
                "has_generic_models": len(has_generic_set),
            },
        },
        "input_files": {
            "tier_a2_jsonl": str(A2_JSONL),
            "tier_a2_csv": str(A2_CSV),
            "models_index_jsonl": str(ALL_MODELS_JSONL),
        },
        "output_files": {
            "main_dataset_repos_txt": str(OUT_REPOS_TXT),
            "main_dataset_repo_details_jsonl": str(OUT_DETAILS_JSONL),
            "main_dataset_repo_details_csv": str(OUT_DETAILS_CSV),
            "main_dataset_definition_json": str(OUT_DEFINITION),
        },
    }
    with OUT_DEFINITION.open("w") as f:
        json.dump(definition, f, indent=2)
    print(f"  wrote {OUT_DEFINITION}")


if __name__ == "__main__":
    main()

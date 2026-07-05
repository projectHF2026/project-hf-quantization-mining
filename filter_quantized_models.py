#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_quantized_models.py
==========================
Filters ALL quantized models from the full HuggingFace model dump (~2.7M models).

Detection Strategy (3 layers, from most to least reliable):
  Layer 1: config.quantization_config  — most authoritative signal
  Layer 2: tags / library_name          — HF-assigned metadata
  Layer 3: model ID + filenames         — name-based heuristics (lower precision)

Each model is classified with ALL matching signals so you can audit
which layer(s) triggered the match. This enables transparent reporting
in your paper's methodology section.

Quantization methods covered (comprehensive as of March 2026):
  GPTQ, AWQ, GGUF/GGML, BitsAndBytes (4-bit/8-bit), AQLM, QuIP/QuIP#,
  HQQ, EETQ, Quanto/TorchAO, SqueezeLLM, EXL2/EXL3, OpenVINO INT,
  TensorRT, FBGEMM, compressed-tensors, VPTQ, SpQR, FP8, SmoothQuant

Usage:
  python3 filter_quantized_models.py \
    --info-dir /path/to/modelsInfo \
    --output-dir /path/to/output

Output files:
  quantized_models_all.jsonl       — full records with detection metadata
  quantized_models_ids.txt         — one model ID per line
  quantized_models_summary.csv     — model_id, quant_methods, detection_layers, tags
  filtering_stats.json             — aggregate statistics for paper reporting
"""

import os
import re
import sys
import json
import csv
import glob
import argparse
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# ============================================================================
# QUANTIZATION DETECTION RULES
# ============================================================================

# Layer 1: quantization_config.quant_method values
# These are the values HF transformers recognizes in config.json
QUANT_CONFIG_METHODS = {
    "awq", "gptq", "bitsandbytes", "bitsandbytes_4bit", "bitsandbytes_8bit",
    "aqlm", "quanto", "eetq", "hqq", "squeezellm", "fbgemm_fp8",
    "compressed-tensors", "compressed_tensors", "fp8", "vptq",
    "torchao", "spqr", "quip", "exl2",
    "marlin", "gptqmodel", "omniquant",
    # Additional PTQ methods from literature (Table I survey)
    "owq", "quarot", "flatquant", "zeroquant", "rptq", "pb-llm", "llm.int8",
}

# Layer 2a: Tags that indicate quantization
# These are tags assigned by HF or model uploaders
QUANT_TAGS = {
    # Format/method tags — these are strong quantization signals
    "gptq", "awq", "gguf", "ggml", "bitsandbytes", "bnb",
    "aqlm", "quip", "quip#", "hqq", "eetq", "quanto", "torchao",
    "squeezellm", "exl2", "exl3", "exllama", "exllamav2",
    "vptq", "spqr", "fp8", "smoothquant",
    "compressed-tensors",
    # Additional PTQ methods from literature
    "owq", "omniquant", "quarot", "flatquant", "zeroquant",
    "rptq", "pb-llm", "llm.int8", "llm-int8", "llm_int8", "marlin", "gptqmodel",
    # Bit-width tags
    "4bit", "8bit", "3bit", "2bit", "5bit", "6bit",
    "4-bit", "8-bit", "3-bit", "2-bit", "5-bit", "6-bit",
    "int4", "int8", "int2", "int3",
    "nf4", "fp4",
    # General tags
    "quantized", "quantization",
    # Auto-GPTQ / Auto-AWQ specific
    "auto-gptq", "autogptq", "auto-awq", "autoawq",
}

# ---------------------------------------------------------------------------
# Auxiliary-ecosystem detection — UPSTREAM (intentionally narrower)
# ---------------------------------------------------------------------------
# This upstream detector writes seven `_quantized`-suffixed labels into a
# model's quant_methods field when an EXPORT_ONLY tag/library co-occurs with
# a quantization co-signal. It DOES NOT emit labels for MLX, vLLM-quantized,
# or MLC-LLM — those are detected by the analysis-time classifier in
# rq1_prevalence.py:detect_auxiliary(), which rescans library_name / tags /
# model_id and is AUTHORITATIVE for paper numbers (auxiliary-only = 8,538,
# the 7-row q1_ecosystem_distribution.csv table, taxonomy_complete_fixed.csv).
# The canonical 7-label auxiliary-ecosystem set for the paper is:
#   {MLX, MLC-LLM, vLLM-quantized, ONNX_quantized, OpenVINO, TensorRT, CoreML}
# defined in rq1_prevalence.py:detect_auxiliary(). The upstream and analysis
# detectors are kept asymmetric by design; recomputing auxiliary counts
# directly from this file's labels will UNDERCOUNT (43, not 8,538).
# Export-only frameworks: NOT quantized by themselves.
# Only count as quantized when combined with a quant co-signal
# (e.g., "openvino" + "int8" = quantized; "openvino" alone = just exported)
EXPORT_ONLY_TAGS = {
    "openvino", "tensorrt", "onnx", "coreml", "optimum",
    "neural-compressor", "nncf",
}

# Co-signals that promote an export-only tag to a quantization signal
EXPORT_QUANT_COSIGNALS = {
    "int4", "int8", "int2", "int3",
    "4bit", "8bit", "3bit", "2bit",
    "4-bit", "8-bit", "3-bit", "2-bit",
    "quantized", "quantization", "quant",
    "smoothquant", "fp8", "nf4", "fp4",
    "gptq", "awq",
}

# Layer 2b: library_name values that indicate quantization
QUANT_LIBRARIES = {
    "gguf", "gptq", "awq", "bitsandbytes",
    "aqlm", "quanto", "hqq", "eetq", "exl2",
    "vptq", "ctransformers",
}

# Export-only library names (same rule: need quant co-signal)
EXPORT_ONLY_LIBRARIES = {
    "openvino", "tensorrt", "onnx", "coreml", "optimum",
}

# Layer 3a: Regex patterns for model ID (org/name)
# These catch models like "TheBloke/CodeLlama-7B-GPTQ" or "user/model-4bit-128g"
QUANT_ID_PATTERNS = [
    # Method names in model ID
    re.compile(r'[-_.](?:gptq|GPTQ)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:awq|AWQ)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:gguf|GGUF)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:ggml|GGML)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:aqlm|AQLM)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:quip|QuIP)(?:#)?(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:hqq|HQQ)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:eetq|EETQ)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:exl2|EXL2|exl3|EXL3)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:squeezellm|SqueezeLLM)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:spqr|SpQR)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:vptq|VPTQ)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:fp8|FP8)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:smoothquant|SmoothQuant)(?:[-_.]|$)', re.IGNORECASE),
    # Bit-width patterns in model ID
    re.compile(r'[-_.](?:int[2-8]|INT[2-8])(?:[-_.]|$)'),
    re.compile(r'[-_.](?:[2-8]bit|[2-8]Bit|[2-8]BIT)(?:[-_.]|$)'),
    re.compile(r'[-_.](?:nf4|NF4|fp4|FP4)(?:[-_.]|$)'),
    re.compile(r'[-_.](?:w[48]a(?:16|fp16))(?:[-_.]|$)', re.IGNORECASE),  # w4a16, w8afp16
    # Quantized as explicit word
    re.compile(r'[-_.]quantized(?:[-_.]|$)', re.IGNORECASE),
    # K-quant GGUF patterns like Q4_K_M, Q5_0, etc.
    re.compile(r'[-_.]Q[2-8]_[0KS](?:_[MSL])?(?:[-_.]|$)'),
    # BnB patterns
    re.compile(r'[-_.](?:bnb|BnB)[-_.]?(?:4bit|8bit|nf4)(?:[-_.]|$)', re.IGNORECASE),
    # Group size patterns like 128g, 32g (common in GPTQ model names)
    re.compile(r'[-_.](?:128g|64g|32g)(?:[-_.]|$)', re.IGNORECASE),
    # Also g128, g64, g32 (reverse ordering seen in the wild)
    re.compile(r'[-_.]g(?:32|64|128)(?:[-_.]|$)', re.IGNORECASE),
    # group_size patterns like group-size-128, group_size_64
    re.compile(r'[-_.]group[-_]?size[-_]?(?:32|64|128)(?:[-_.]|$)', re.IGNORECASE),
    # Marlin / GPTQModel / OmniQuant (additional method names seen in the wild)
    re.compile(r'[-_.](?:marlin|Marlin)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:gptqmodel|GPTQModel)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:omniquant|OmniQuant)(?:[-_.]|$)', re.IGNORECASE),
    # Additional PTQ methods from literature survey
    re.compile(r'[-_.](?:owq|OWQ)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:quarot|QuaRot)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:flatquant|FlatQuant)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:zeroquant|ZeroQuant)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:rptq|RPTQ)(?:[-_.]|$)', re.IGNORECASE),
    re.compile(r'[-_.](?:pb-llm|PB-LLM|pbllm)(?:[-_.]|$)', re.IGNORECASE),
    # LLM.int8 — special case: dot in the middle
    re.compile(r'(?:^|[-_/\.])llm[-_.]?int8(?:$|[-_/\.])', re.IGNORECASE),
]

# Layer 3b: File extensions/names in siblings that indicate quantization
QUANT_FILE_PATTERNS = [
    re.compile(r'\.gguf$', re.IGNORECASE),
    re.compile(r'\.ggml$', re.IGNORECASE),
    re.compile(r'quantize_config\.json$', re.IGNORECASE),
    re.compile(r'quant_config\.json$', re.IGNORECASE),
    re.compile(r'quantization_config\.json$', re.IGNORECASE),
    re.compile(r'gptq_model', re.IGNORECASE),
]


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_from_config(model_data: dict) -> tuple[set, set]:
    """
    Layer 1: Check quantization_config in model config.
    Also checks alternative config paths seen in the wild.
    Returns (methods_found, detection_signals).
    """
    methods = set()
    signals = set()

    config = model_data.get("config", {}) or {}

    # Path 1: config -> quantization_config (standard HF path)
    qconfig = config.get("quantization_config", {}) or {}

    # Canonical set for flexible matching (hyphens/underscores/spaces)
    _qcm_canonical = {m.replace("_", "-") for m in QUANT_CONFIG_METHODS}

    if qconfig:
        qmethod_raw = qconfig.get("quant_method", "")
        qmethod = qmethod_raw.lower().strip()
        qmethod_key = qmethod.replace("_", "-").replace(" ", "")
        if qmethod:
            normalized = normalize_method(qmethod_key)
            if qmethod_key in _qcm_canonical:
                methods.add(normalized)
                signals.add(f"config.quantization_config.quant_method={qmethod_raw}")
            else:
                methods.add("unknown_quantized")
                signals.add(f"config.quantization_config.quant_method={qmethod_raw} (unknown method)")

        # BitsAndBytes detection via load_in_4bit / load_in_8bit
        if qconfig.get("load_in_4bit"):
            methods.add("BitsAndBytes_4bit")
            signals.add("config.quantization_config.load_in_4bit=True")
        if qconfig.get("load_in_8bit"):
            methods.add("BitsAndBytes_8bit")
            signals.add("config.quantization_config.load_in_8bit=True")

        # BnB sub-fields that confirm quantization
        if qconfig.get("bnb_4bit_quant_type"):
            methods.add("BitsAndBytes_4bit")
            signals.add(f"config.quantization_config.bnb_4bit_quant_type={qconfig['bnb_4bit_quant_type']}")
        if qconfig.get("bnb_4bit_compute_dtype"):
            methods.add("BitsAndBytes_4bit")
            signals.add("config.quantization_config.bnb_4bit_compute_dtype present")

        # Bits field without method
        bits = qconfig.get("bits")
        if bits and not qmethod:
            methods.add("unknown_quantized")
            signals.add(f"config.quantization_config.bits={bits}")

    # Path 2: top-level "quantization_config" (some API responses)
    qconfig_top = model_data.get("quantization_config", {}) or {}
    if qconfig_top and not qconfig:
        qmethod_raw2 = qconfig_top.get("quant_method", "")
        qmethod2 = qmethod_raw2.lower().strip()
        qmethod2_key = qmethod2.replace("_", "-").replace(" ", "")
        if qmethod2:
            if qmethod2_key in _qcm_canonical:
                methods.add(normalize_method(qmethod2_key))
                signals.add(f"quantization_config.quant_method={qmethod_raw2}")
            else:
                methods.add("unknown_quantized")
                signals.add(f"quantization_config.quant_method={qmethod_raw2} (unknown method)")
        if qconfig_top.get("load_in_4bit"):
            methods.add("BitsAndBytes_4bit")
            signals.add("quantization_config.load_in_4bit=True")
        if qconfig_top.get("load_in_8bit"):
            methods.add("BitsAndBytes_8bit")
            signals.add("quantization_config.load_in_8bit=True")

    # Path 3: alternative config keys seen in the wild
    for alt_key in ("quantization", "quant_config", "quantize_config"):
        alt_val = config.get(alt_key, {}) or {}
        if isinstance(alt_val, dict) and alt_val:
            qmethod = alt_val.get("quant_method", "").lower().strip()
            if qmethod:
                methods.add(normalize_method(qmethod))
                signals.add(f"config.{alt_key}.quant_method={qmethod}")
            elif alt_val.get("bits"):
                methods.add("unknown_quantized")
                signals.add(f"config.{alt_key}.bits={alt_val['bits']}")
        elif isinstance(alt_val, str) and alt_val.strip():
            # Some models store just the method name as a string
            methods.add(normalize_method(alt_val.strip()))
            signals.add(f"config.{alt_key}={alt_val.strip()}")

    # Path 4: BnB flags at config top level (outside quantization_config)
    if config.get("load_in_4bit"):
        methods.add("BitsAndBytes_4bit")
        signals.add("config.load_in_4bit=True")
    if config.get("load_in_8bit"):
        methods.add("BitsAndBytes_8bit")
        signals.add("config.load_in_8bit=True")

    # Path 5: top-level "gguf" field (present in GGUF models)
    if model_data.get("gguf"):
        methods.add("GGUF")
        signals.add("top-level gguf metadata present")

    return methods, signals


def detect_from_tags(model_data: dict) -> tuple[set, set]:
    """
    Layer 2: Check tags and library_name.
    Export-only frameworks (openvino, tensorrt, etc.) only count as
    quantized when combined with a quant co-signal.
    Returns (methods_found, detection_signals).
    """
    methods = set()
    signals = set()

    # Tags
    tags = model_data.get("tags", []) or []
    tags_lower = {t.lower().strip() for t in tags}

    # Strong quant tags
    for tag in tags_lower:
        if tag in QUANT_TAGS:
            methods.add(normalize_method(tag))
            signals.add(f"tag={tag}")

    # Export-only tags: only count if a quant co-signal is also present
    has_quant_cosignal = bool(tags_lower & EXPORT_QUANT_COSIGNALS)
    # Also check model name for co-signals
    model_id = (model_data.get("id") or model_data.get("modelId") or "").lower()
    if not has_quant_cosignal:
        for cosig in EXPORT_QUANT_COSIGNALS:
            if cosig in model_id:
                has_quant_cosignal = True
                break

    for tag in tags_lower:
        if tag in EXPORT_ONLY_TAGS:
            if has_quant_cosignal:
                methods.add(normalize_method(tag) + "_quantized")
                signals.add(f"tag={tag} (with quant co-signal)")
            # else: skip — export-only, not quantized

    # library_name
    lib = (model_data.get("library_name") or "").lower().strip()
    if lib in QUANT_LIBRARIES:
        methods.add(normalize_method(lib))
        signals.add(f"library_name={lib}")
    elif lib in EXPORT_ONLY_LIBRARIES:
        if has_quant_cosignal:
            methods.add(normalize_method(lib) + "_quantized")
            signals.add(f"library_name={lib} (with quant co-signal)")

    return methods, signals


def detect_from_name_and_files(model_data: dict) -> tuple[set, set]:
    """
    Layer 3: Check model ID patterns and file names.
    Returns (methods_found, detection_signals).
    """
    methods = set()
    signals = set()

    # Model ID
    model_id = model_data.get("id") or model_data.get("modelId") or ""
    for pattern in QUANT_ID_PATTERNS:
        match = pattern.search(model_id)
        if match:
            matched_text = match.group(0).strip("-_.")
            methods.add(normalize_method(matched_text))
            signals.add(f"model_id_pattern={matched_text}")

    # Siblings (file list)
    siblings = model_data.get("siblings", []) or []
    gguf_count = 0
    for sib in siblings:
        fname = sib.get("rfilename", "") if isinstance(sib, dict) else str(sib)
        for pattern in QUANT_FILE_PATTERNS:
            if pattern.search(fname):
                if fname.lower().endswith(".gguf"):
                    gguf_count += 1
                    if gguf_count == 1:  # only log once
                        methods.add("GGUF")
                        signals.add(f"file=*.gguf ({gguf_count}+ files)")
                elif fname.lower().endswith(".ggml"):
                    methods.add("GGML")
                    signals.add(f"file={fname}")
                else:
                    signals.add(f"file={fname}")
                    methods.add("quantized_config_file")

    # Update gguf count signal
    if gguf_count > 1:
        signals.discard(f"file=*.gguf (1+ files)")
        signals.add(f"file=*.gguf ({gguf_count} files)")

    return methods, signals


def normalize_method(raw: str) -> str:
    """Normalize quantization method names to canonical forms."""
    raw = raw.lower().strip("-_. ")

    # Map to canonical names
    mapping = {
        "gptq": "GPTQ",
        "autogptq": "GPTQ",
        "auto-gptq": "GPTQ",
        "awq": "AWQ",
        "autoawq": "AWQ",
        "auto-awq": "AWQ",
        "gguf": "GGUF",
        "ggml": "GGML",
        "bitsandbytes": "BitsAndBytes",
        "bitsandbytes_4bit": "BitsAndBytes_4bit",
        "bitsandbytes_8bit": "BitsAndBytes_8bit",
        "bitsandbytes-4bit": "BitsAndBytes_4bit",
        "bitsandbytes-8bit": "BitsAndBytes_8bit",
        "bnb": "BitsAndBytes",
        "bnb-4bit": "BitsAndBytes_4bit",
        "bnb-8bit": "BitsAndBytes_8bit",
        "bnb_4bit": "BitsAndBytes_4bit",
        "bnb_8bit": "BitsAndBytes_8bit",
        "bnb-nf4": "BitsAndBytes_4bit",
        "4bit": "4bit",
        "8bit": "8bit",
        "3bit": "3bit",
        "2bit": "2bit",
        "5bit": "5bit",
        "6bit": "6bit",
        "4-bit": "4bit",
        "8-bit": "8bit",
        "3-bit": "3bit",
        "2-bit": "2bit",
        "5-bit": "5bit",
        "6-bit": "6bit",
        "int4": "INT4",
        "int8": "INT8",
        "int2": "INT2",
        "int3": "INT3",
        "nf4": "NF4",
        "fp4": "FP4",
        "fp8": "FP8",
        "aqlm": "AQLM",
        "quip": "QuIP",
        "quip#": "QuIP#",
        "hqq": "HQQ",
        "eetq": "EETQ",
        "quanto": "Quanto",
        "torchao": "TorchAO",
        "squeezellm": "SqueezeLLM",
        "exl2": "EXL2",
        "exl3": "EXL3",
        "exllama": "ExLlama",
        "exllamav2": "ExLlamaV2",
        "openvino": "OpenVINO",
        "tensorrt": "TensorRT",
        "vptq": "VPTQ",
        "spqr": "SpQR",
        "smoothquant": "SmoothQuant",
        "compressed-tensors": "CompressedTensors",
        "compressed_tensors": "CompressedTensors",
        "fbgemm_fp8": "FBGEMM_FP8",
        "quantized": "Quantized_generic",
        "quantization": "Quantized_generic",
        "marlin": "Marlin",
        "gptqmodel": "GPTQModel",
        "omniquant": "OmniQuant",
        "owq": "OWQ",
        "quarot": "QuaRot",
        "flatquant": "FlatQuant",
        "zeroquant": "ZeroQuant",
        "llm.int8": "LLM_int8",
        "llm-int8": "LLM_int8",
        "llm_int8": "LLM_int8",
        "rptq": "RPTQ",
        "pb-llm": "PB-LLM",
        "pbllm": "PB-LLM",
    }

    result = mapping.get(raw)
    if result:
        return result

    # K-quant GGUF patterns: q4_k_m, q8_0, q5_k_s, q2_k, etc. → GGUF
    if re.match(r'^q[2-8]_[0ks](?:_[msl])?$', raw):
        return "GGUF"

    # Group-size patterns: 128g, g128, 64g, g64, 32g, g32 → GPTQ (group size is a GPTQ convention)
    if re.match(r'^(?:128g|64g|32g|g128|g64|g32)$', raw):
        return "GPTQ"

    # group_size variants
    if re.match(r'^group[-_]?size[-_]?(?:32|64|128)$', raw):
        return "GPTQ"

    # w4a16, w8afp16 patterns → generic quantized
    if re.match(r'^w[48]a(?:16|fp16)$', raw):
        return "Quantized_generic"

    return raw


# ============================================================================
# PROCESS ONE MODEL FILE
# ============================================================================

def process_model_file(filepath: str) -> Optional[dict]:
    """
    Process a single model JSON file.
    Returns a result dict if the model is quantized, None otherwise.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None

    model_id = data.get("id") or data.get("modelId") or ""
    if not model_id:
        return None

    # Run all three detection layers
    methods_l1, signals_l1 = detect_from_config(data)
    methods_l2, signals_l2 = detect_from_tags(data)
    methods_l3, signals_l3 = detect_from_name_and_files(data)

    all_methods = methods_l1 | methods_l2 | methods_l3
    all_signals = signals_l1 | signals_l2 | signals_l3

    if not all_methods:
        return None

    # Determine which layers triggered
    layers_triggered = []
    if methods_l1:
        layers_triggered.append("L1_config")
    if methods_l2:
        layers_triggered.append("L2_tags")
    if methods_l3:
        layers_triggered.append("L3_heuristic")

    # Extract useful metadata
    tags = data.get("tags", []) or []
    pipeline_tag = data.get("pipeline_tag", "")
    library_name = data.get("library_name", "")
    downloads = data.get("downloads", 0)
    likes = data.get("likes", 0)
    created_at = data.get("createdAt", "")
    last_modified = data.get("lastModified", "")
    author = data.get("author", "")

    # Determine high-confidence vs candidate
    # HC = Layer 1 (config) OR explicit quant method in any layer OR
    #      GGUF/GGML files in siblings OR quantize_config.json file
    HC_METHODS = {
        "GPTQ", "AWQ", "GGUF", "GGML", "BitsAndBytes", "BitsAndBytes_4bit",
        "BitsAndBytes_8bit", "AQLM", "QuIP", "QuIP#", "HQQ", "EETQ",
        "Quanto", "TorchAO", "SqueezeLLM", "EXL2", "EXL3", "ExLlama",
        "ExLlamaV2", "VPTQ", "SpQR", "FP8", "SmoothQuant",
        "CompressedTensors", "FBGEMM_FP8", "INT4", "INT8", "NF4", "FP4",
        "4bit", "8bit", "3bit", "2bit",
        "Marlin", "GPTQModel", "OmniQuant",
        "OWQ", "QuaRot", "FlatQuant", "ZeroQuant", "LLM_int8", "RPTQ", "PB-LLM",
    }
    # File-based L3 signals that are high-confidence
    HC_L3_METHODS = {"GGUF", "GGML", "quantized_config_file"}
    is_high_confidence = (
        bool(methods_l1)
        or bool(all_methods & HC_METHODS)
        or bool(methods_l3 & HC_L3_METHODS)
    )

    return {
        "model_id": model_id,
        "quant_methods": sorted(all_methods),
        "detection_layers": layers_triggered,
        "detection_signals": sorted(all_signals),
        "high_confidence": is_high_confidence,
        "pipeline_tag": pipeline_tag,
        "library_name": library_name,
        "tags": tags,
        "downloads": downloads,
        "likes": likes,
        "author": author,
        "created_at": created_at,
        "last_modified": last_modified,
        # Per-layer detail for auditing
        "l1_methods": sorted(methods_l1),
        "l2_methods": sorted(methods_l2),
        "l3_methods": sorted(methods_l3),
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def collect_files(info_dir: str) -> list:
    """Collect all model JSON files."""
    pattern = os.path.join(info_dir, "*.json")
    files = glob.glob(pattern)
    print(f"Found {len(files):,} model JSON files in {info_dir}")
    return files


def process_batch(filepaths: list) -> list:
    """Process a batch of files (used by ProcessPoolExecutor)."""
    results = []
    for fp in filepaths:
        result = process_model_file(fp)
        if result is not None:
            results.append(result)
    return results


def run_parallel(files: list, num_workers: int = 8, batch_size: int = 500) -> list:
    """Process all files in parallel batches."""
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    all_results = []

    print(f"Processing {len(files):,} files in {len(batches):,} batches "
          f"with {num_workers} workers...")

    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_batch, batch): i
                   for i, batch in enumerate(batches)}

        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
            done += 1

            if done % 50 == 0 or done == len(batches):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(batches) - done) / rate if rate > 0 else 0
                print(f"  Batches: {done}/{len(batches)} | "
                      f"Quantized found: {len(all_results):,} | "
                      f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nProcessing complete in {elapsed:.1f}s")
    print(f"Total quantized models found: {len(all_results):,}")

    return all_results


# ============================================================================
# OUTPUT
# ============================================================================

def write_outputs(results: list, output_dir: str, total_models: int):
    """Write all output files, split into high-confidence and candidate sets."""
    os.makedirs(output_dir, exist_ok=True)

    hc_results = [r for r in results if r.get("high_confidence")]
    candidate_only = [r for r in results if not r.get("high_confidence")]

    print(f"  High-confidence: {len(hc_results):,}")
    print(f"  Candidate-only:  {len(candidate_only):,}")
    print(f"  Total:           {len(results):,}")

    # 1. Full JSONL (all)
    jsonl_path = os.path.join(output_dir, "quantized_models_all.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Written: {jsonl_path}")

    # 2a. High-confidence IDs
    hc_ids_path = os.path.join(output_dir, "quantized_models_high_confidence_ids.txt")
    with open(hc_ids_path, "w", encoding="utf-8") as f:
        for r in sorted(hc_results, key=lambda x: x["model_id"]):
            f.write(r["model_id"] + "\n")
    print(f"  Written: {hc_ids_path}")

    # 2b. All IDs
    ids_path = os.path.join(output_dir, "quantized_models_all_ids.txt")
    with open(ids_path, "w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: x["model_id"]):
            f.write(r["model_id"] + "\n")
    print(f"  Written: {ids_path}")

    # 2c. Candidate-only IDs (for auditing)
    cand_ids_path = os.path.join(output_dir, "quantized_models_candidate_only_ids.txt")
    with open(cand_ids_path, "w", encoding="utf-8") as f:
        for r in sorted(candidate_only, key=lambda x: x["model_id"]):
            f.write(r["model_id"] + "\n")
    print(f"  Written: {cand_ids_path}")

    # 3. Summary CSV
    csv_path = os.path.join(output_dir, "quantized_models_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_id", "high_confidence", "quant_methods", "detection_layers",
            "pipeline_tag", "library_name", "downloads", "likes",
            "author", "created_at"
        ])
        for r in sorted(results, key=lambda x: x["model_id"]):
            writer.writerow([
                r["model_id"],
                r["high_confidence"],
                ";".join(r["quant_methods"]),
                ";".join(r["detection_layers"]),
                r["pipeline_tag"],
                r["library_name"],
                r["downloads"],
                r["likes"],
                r["author"],
                r["created_at"],
            ])
    print(f"  Written: {csv_path}")

    # 4. Statistics JSON
    stats = compute_stats(results, total_models)
    stats_path = os.path.join(output_dir, "filtering_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Written: {stats_path}")

    # Print summary
    print_summary(stats)


def compute_stats(results: list, total_models: int) -> dict:
    """Compute aggregate statistics."""
    method_counter = Counter()
    layer_counter = Counter()
    pipeline_counter = Counter()
    library_counter = Counter()
    author_counter = Counter()

    l1_only = 0
    l2_only = 0
    l3_only = 0
    multi_layer = 0

    for r in results:
        for m in r["quant_methods"]:
            method_counter[m] += 1
        for l in r["detection_layers"]:
            layer_counter[l] += 1
        if r["pipeline_tag"]:
            pipeline_counter[r["pipeline_tag"]] += 1
        if r["library_name"]:
            library_counter[r["library_name"]] += 1
        if r["author"]:
            author_counter[r["author"]] += 1

        layers = set(r["detection_layers"])
        if len(layers) == 1:
            if "L1_config" in layers:
                l1_only += 1
            elif "L2_tags" in layers:
                l2_only += 1
            elif "L3_heuristic" in layers:
                l3_only += 1
        else:
            multi_layer += 1

    hc_count = sum(1 for r in results if r.get("high_confidence"))
    candidate_only_count = len(results) - hc_count

    return {
        "total_models_scanned": total_models,
        "total_quantized_found": len(results),
        "high_confidence_count": hc_count,
        "candidate_only_count": candidate_only_count,
        "quantized_percentage": round(len(results) / total_models * 100, 2) if total_models > 0 else 0,
        "hc_percentage": round(hc_count / total_models * 100, 2) if total_models > 0 else 0,
        "detection_layer_breakdown": {
            "L1_config_only": l1_only,
            "L2_tags_only": l2_only,
            "L3_heuristic_only": l3_only,
            "multi_layer_confirmed": multi_layer,
        },
        "layer_trigger_counts": dict(layer_counter.most_common()),
        "quant_method_distribution": dict(method_counter.most_common(50)),
        "top_pipeline_tags": dict(pipeline_counter.most_common(20)),
        "top_libraries": dict(library_counter.most_common(20)),
        "top_authors": dict(author_counter.most_common(30)),
    }


def print_summary(stats: dict):
    """Print a human-readable summary."""
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total models scanned:    {stats['total_models_scanned']:>12,}")
    print(f"Quantized models (all):  {stats['total_quantized_found']:>12,}  ({stats['quantized_percentage']}%)")
    print(f"  High-confidence:       {stats['high_confidence_count']:>12,}  ({stats['hc_percentage']}%)")
    print(f"  Candidate-only:        {stats['candidate_only_count']:>12,}")
    print(f"\nDetection layer breakdown:")
    for k, v in stats["detection_layer_breakdown"].items():
        print(f"  {k:<30s}: {v:>8,}")
    print(f"\nTop 15 quantization methods:")
    for method, count in list(stats["quant_method_distribution"].items())[:15]:
        print(f"  {method:<25s}: {count:>8,}")
    print(f"\nTop 10 pipeline tags:")
    for tag, count in list(stats["top_pipeline_tags"].items())[:10]:
        print(f"  {tag:<30s}: {count:>8,}")
    print(f"\nTop 10 authors:")
    for author, count in list(stats["top_authors"].items())[:10]:
        print(f"  {author:<30s}: {count:>8,}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Filter quantized models from HuggingFace model dump"
    )
    parser.add_argument(
        "--info-dir",
        default="/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/HuggingFaceStudy/modelsInfo",
        help="Directory containing model JSON files"
    )
    parser.add_argument(
        "--output-dir",
        default="/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/quantized_filtered",
        help="Directory for output files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Files per batch (default: 500)"
    )

    args = parser.parse_args()

    print(f"Quantized Model Filter")
    print(f"{'='*60}")
    print(f"Info dir:   {args.info_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Workers:    {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Collect files
    files = collect_files(args.info_dir)
    if not files:
        print("ERROR: No JSON files found. Check --info-dir path.")
        sys.exit(1)

    total_models = len(files)

    # Process
    results = run_parallel(files, num_workers=args.workers, batch_size=args.batch_size)

    # Write outputs
    print(f"\nWriting output files...")
    write_outputs(results, args.output_dir, total_models)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Standalone Ollama-native + self-quantized GitHub miner.

A fully self-contained side study. Reads nothing from the rest of the
icpc-approch/ pipeline; writes only inside its own directory:

    ollama_selfquant_repos.jsonl   one JSON object per unique repo
    summary.csv                    counts by category, tool, signal, quant tag
    checkpoint.json                resumable state (completed signals + stats)
    miner.log                      progress + skips + rate-limit events

Discovery: GitHub Code Search REST API (text-match payloads, paginated to
the 1000-result cap). Detection: two editable token lists at the top --
Category A (Ollama-native) and Category B (self-quantize-and-persist).

Auth: export GITHUB_TOKEN=... (also accepts GH_TOKEN_1 for compatibility).

Re-runs are safe and resumable: completed signals are skipped, the JSONL is
rewritten from in-memory state after each signal, and partial progress
survives Ctrl-C.

Run:   python3 ollama_selfquant_miner.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# ===========================================================================
# CONFIG  --  editable knobs
# ===========================================================================

OUTPUT_DIR     = Path(__file__).resolve().parent
JSONL_OUT      = OUTPUT_DIR / "ollama_selfquant_repos.jsonl"
MISSED_JSONL   = OUTPUT_DIR / "missed_repos.jsonl"     # repos NOT in analysis set
SUMMARY_CSV    = OUTPUT_DIR / "summary.csv"
CHECKPOINT     = OUTPUT_DIR / "checkpoint.json"
LOG_FILE       = OUTPUT_DIR / "miner.log"

# ----- Offline subtraction config ----------------------------------------
# After mining, the script reads this JSONL of repos already in the
# HF-anchored analysis set and marks every mined repo with already_counted
# = true / false (case-insensitive full_name match). Missed repos (=false)
# get written to MISSED_JSONL and are the only ones fed to the metadata
# fetch. No API calls for the subtraction step itself.
ANALYSIS_SET_JSONL = Path(
    "/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/"
    "output_dir/rq_analysis/shared/results/analysis_set_repo_details.jsonl"
)
ANALYSIS_SET_FIELD = "repo"   # JSONL field that holds the GitHub full_name

GITHUB_API           = "https://api.github.com"
MAX_PAGES = 10
PER_PAGE             = 100
TOKEN_MIN_INTERVAL_S = 2.1    # Min seconds between successive uses of ONE token
                              # (code search = 30 req/min auth'd, 2s = 30/min + headroom).
                              # Aggregate throughput = N_tokens / TOKEN_MIN_INTERVAL_S
                              # req/s, e.g. 9 tokens -> ~4.3 req/s (~258 req/min).
HTTP_TIMEOUT_S       = 30
MAX_BACKOFF_S        = 600
USER_AGENT           = "ollama-selfquant-miner/1.0 (research)"
FETCH_METADATA       = True   # set False to skip the /repos/{owner}/{name} step
MAX_HTTP_ATTEMPTS    = 8

# ----- CATEGORY A : Ollama-native signals --------------------------------
# Each entry: name, tool, query (verbatim Code Search query), and an optional
# "qualifier" (e.g. "language:python") appended to the query at search time.
# Use a qualifier where the language/file is unambiguous; leave it "" otherwise.
CATEGORY_A_SIGNALS: list[dict] = [
    # Modelfile (filename + FROM directive) -- already has a filename: qualifier
    {"name": "ollama_modelfile_file",       "tool": "Ollama Modelfile",   "query": "filename:Modelfile",                       "qualifier": ""},
    {"name": "ollama_modelfile_FROM",       "tool": "Ollama Modelfile",   "query": '"FROM " filename:Modelfile',               "qualifier": ""},
    # Ollama CLI -- may appear in shell, Dockerfile, README, .ipynb; no language qualifier
    {"name": "ollama_cli_pull",             "tool": "Ollama CLI",         "query": '"ollama pull"',                            "qualifier": ""},
    {"name": "ollama_cli_run",              "tool": "Ollama CLI",         "query": '"ollama run"',                             "qualifier": ""},
    {"name": "ollama_cli_create",           "tool": "Ollama CLI",         "query": '"ollama create"',                          "qualifier": ""},
    {"name": "ollama_cli_serve",            "tool": "Ollama CLI",         "query": '"ollama serve"',                           "qualifier": ""},
    {"name": "ollama_cli_push",             "tool": "Ollama CLI",         "query": '"ollama push"',                            "qualifier": ""},
    {"name": "ollama_cli_show",             "tool": "Ollama CLI",         "query": '"ollama show"',                            "qualifier": ""},
    {"name": "ollama_cli_list",             "tool": "Ollama CLI",         "query": '"ollama list"',                            "qualifier": ""},
    # ollama-python client
    {"name": "ollama_py_import",            "tool": "ollama-python",      "query": '"import ollama"',                          "qualifier": "language:python"},
    {"name": "ollama_py_from",              "tool": "ollama-python",      "query": '"from ollama"',                            "qualifier": "language:python"},
    {"name": "ollama_py_chat",              "tool": "ollama-python",      "query": '"ollama.chat("',                           "qualifier": "language:python"},
    {"name": "ollama_py_generate",          "tool": "ollama-python",      "query": '"ollama.generate("',                       "qualifier": "language:python"},
    {"name": "ollama_py_client",            "tool": "ollama-python",      "query": '"ollama.Client("',                         "qualifier": "language:python"},
    {"name": "ollama_py_async_client",      "tool": "ollama-python",      "query": '"ollama.AsyncClient("',                    "qualifier": "language:python"},
    {"name": "ollama_py_embeddings",        "tool": "ollama-python",      "query": '"ollama.embeddings("',                     "qualifier": "language:python"},
    {"name": "ollama_py_embed",             "tool": "ollama-python",      "query": '"ollama.embed("',                          "qualifier": "language:python"},
    {"name": "ollama_py_pull",              "tool": "ollama-python",      "query": '"ollama.pull("',                           "qualifier": "language:python"},
    {"name": "ollama_py_show",              "tool": "ollama-python",      "query": '"ollama.show("',                           "qualifier": "language:python"},
    {"name": "ollama_py_list",              "tool": "ollama-python",      "query": '"ollama.list("',                           "qualifier": "language:python"},
    # ollama-js client (also TS; leave 'from ollama' / 'new Ollama(' unqualified to catch TS)
    {"name": "ollama_js_require",           "tool": "ollama-js",          "query": "\"require('ollama')\"",                    "qualifier": "language:javascript"},
    {"name": "ollama_js_import_quoted",     "tool": "ollama-js",          "query": "\"from 'ollama'\"",                        "qualifier": ""},
    {"name": "ollama_js_new",               "tool": "ollama-js",          "query": '"new Ollama("',                            "qualifier": ""},
    # LangChain Ollama integrations
    {"name": "langchain_ollama_pkg",        "tool": "LangChain Ollama",   "query": '"langchain_ollama"',                       "qualifier": "language:python"},
    {"name": "langchain_ollama_chat",       "tool": "LangChain Ollama",   "query": '"ChatOllama"',                             "qualifier": "language:python"},
    {"name": "langchain_ollama_llm",        "tool": "LangChain Ollama",   "query": '"OllamaLLM"',                              "qualifier": "language:python"},
    {"name": "langchain_ollama_emb",        "tool": "LangChain Ollama",   "query": '"OllamaEmbeddings"',                       "qualifier": "language:python"},
    {"name": "langchain_legacy_ollama",     "tool": "LangChain Ollama",   "query": '"langchain.llms import Ollama"',           "qualifier": "language:python"},
    {"name": "langchain_comm_ollama",       "tool": "LangChain Ollama",   "query": '"langchain_community.llms.ollama"',        "qualifier": "language:python"},
    # LlamaIndex Ollama
    {"name": "llamaindex_ollama_pkg",       "tool": "LlamaIndex Ollama",  "query": '"llama_index.llms.ollama"',                "qualifier": "language:python"},
    {"name": "llamaindex_ollama_import",    "tool": "LlamaIndex Ollama",  "query": '"from llama_index.llms.ollama import"',    "qualifier": "language:python"},
    # REST API to local Ollama (default port 11434) -- any language
    {"name": "ollama_rest_localhost",       "tool": "Ollama REST",        "query": '"localhost:11434"',                        "qualifier": ""},
    {"name": "ollama_rest_127",             "tool": "Ollama REST",        "query": '"127.0.0.1:11434"',                        "qualifier": ""},
    {"name": "ollama_rest_0000",            "tool": "Ollama REST",        "query": '"0.0.0.0:11434"',                          "qualifier": ""},
    {"name": "ollama_rest_api_generate",    "tool": "Ollama REST",        "query": '"/api/generate" "11434"',                  "qualifier": ""},
    {"name": "ollama_rest_api_chat",        "tool": "Ollama REST",        "query": '"/api/chat" "11434"',                      "qualifier": ""},
    {"name": "ollama_rest_api_pull",        "tool": "Ollama REST",        "query": '"/api/pull" "11434"',                      "qualifier": ""},
    {"name": "ollama_rest_api_embeddings",  "tool": "Ollama REST",        "query": '"/api/embeddings" "11434"',                "qualifier": ""},
    {"name": "ollama_rest_api_embed",       "tool": "Ollama REST",        "query": '"/api/embed" "11434"',                     "qualifier": ""},
    {"name": "ollama_rest_api_tags",        "tool": "Ollama REST",        "query": '"/api/tags" "11434"',                      "qualifier": ""},
    # Ollama registry references
    {"name": "ollama_registry_old",         "tool": "Ollama Registry",    "query": '"registry.ollama.ai"',                     "qualifier": ""},
    {"name": "ollama_registry_library",     "tool": "Ollama Registry",    "query": '"ollama.com/library"',                     "qualifier": ""},
    # Ollama Go SDK
    {"name": "ollama_go_sdk",               "tool": "Ollama Go SDK",      "query": '"github.com/ollama/ollama/api"',           "qualifier": "language:go"},
]

# ----- CATEGORY B : self-quantized signals -------------------------------
# Each query targets an explicit *quantize-and-persist* step. BitsAndBytes
# load_in_4bit at load time is intentionally excluded: it's load-time
# quantization, not an artifact-producing quantize step. `save_quantized(`
# IS included because that is the canonical persist call.
CATEGORY_B_SIGNALS: list[dict] = [
    # GPTQ family -- Python ecosystem
    {"name": "gptq_autogptq_class",         "tool": "GPTQ:AutoGPTQ",      "query": '"AutoGPTQForCausalLM"',                    "qualifier": "language:python"},
    {"name": "gptq_autogptq_pkg",           "tool": "GPTQ:AutoGPTQ",      "query": '"from auto_gptq"',                         "qualifier": "language:python"},
    {"name": "gptq_gptqmodel_class",        "tool": "GPTQ:GPTQModel",     "query": '"GPTQModel"',                              "qualifier": "language:python"},
    {"name": "gptq_gptqmodel_pkg",          "tool": "GPTQ:GPTQModel",     "query": '"from gptqmodel"',                         "qualifier": "language:python"},
    {"name": "gptq_baseconfig",             "tool": "GPTQ",               "query": '"BaseQuantizeConfig"',                     "qualifier": "language:python"},
    {"name": "gptq_quantizer",              "tool": "GPTQ",               "query": '"GPTQQuantizer"',                          "qualifier": "language:python"},
    {"name": "gptq_hf_config",              "tool": "HF GPTQ",            "query": '"GPTQConfig"',                             "qualifier": "language:python"},
    {"name": "gptq_optimum",                "tool": "Optimum GPTQ",       "query": '"from optimum.gptq"',                      "qualifier": "language:python"},
    {"name": "gptq_for_llama",              "tool": "GPTQ-for-LLaMa",     "query": '"GPTQ-for-LLaMa"',                         "qualifier": ""},
    {"name": "gptq_save_quantized",         "tool": "GPTQ",               "query": '"save_quantized("',                        "qualifier": "language:python"},
    {"name": "gptq_save_pretrained_gptq",   "tool": "save_pretrained_gptq","query": '"save_pretrained_gptq"',                  "qualifier": "language:python"},
    # AWQ family
    {"name": "awq_autoawq_class",           "tool": "AWQ:AutoAWQ",        "query": '"AutoAWQForCausalLM"',                     "qualifier": "language:python"},
    {"name": "awq_autoawq_pkg",             "tool": "AWQ:AutoAWQ",        "query": '"from awq import"',                        "qualifier": "language:python"},
    {"name": "awq_hf_config",               "tool": "HF AWQ",             "query": '"AwqConfig"',                              "qualifier": "language:python"},
    {"name": "awq_llm_awq",                 "tool": "MIT llm-awq",        "query": '"llm-awq"',                                "qualifier": ""},
    # llama.cpp / GGUF conversion + quantization (cross-language; both shell + python)
    {"name": "gguf_convert_hf_underscore",  "tool": "llama.cpp convert",  "query": '"convert_hf_to_gguf"',                     "qualifier": ""},
    {"name": "gguf_convert_hf_hyphen",      "tool": "llama.cpp convert",  "query": '"convert-hf-to-gguf"',                     "qualifier": ""},
    {"name": "gguf_convert_lora",           "tool": "llama.cpp convert",  "query": '"convert_lora_to_gguf"',                   "qualifier": ""},
    {"name": "gguf_llama_quantize_cli",     "tool": "llama.cpp quantize", "query": '"llama-quantize"',                         "qualifier": ""},
    {"name": "gguf_outtype_flag",           "tool": "llama.cpp convert",  "query": '"--outtype" gguf',                         "qualifier": ""},
    {"name": "gguf_py_import",              "tool": "gguf-py",            "query": '"from gguf import"',                       "qualifier": "language:python"},
    {"name": "gguf_writer",                 "tool": "gguf-py",            "query": '"GGUFWriter"',                             "qualifier": "language:python"},
    # HF / PyTorch native quantize + persist
    {"name": "quanto_optimum_pkg",          "tool": "optimum-quanto",     "query": '"from optimum.quanto"',                    "qualifier": "language:python"},
    {"name": "quanto_quantize_call",        "tool": "Quanto",             "query": '"quanto.quantize"',                        "qualifier": "language:python"},
    {"name": "torchao_pkg",                 "tool": "torchao",            "query": '"from torchao"',                           "qualifier": "language:python"},
    {"name": "torchao_int4_wo",             "tool": "torchao",            "query": '"int4_weight_only"',                       "qualifier": "language:python"},
    {"name": "torchao_int8_wo",             "tool": "torchao",            "query": '"int8_weight_only"',                       "qualifier": "language:python"},
    {"name": "torchao_dyn_act",             "tool": "torchao",            "query": '"int8_dynamic_activation_int8_weight"',    "qualifier": "language:python"},
    {"name": "optimum_ort_quantizer",       "tool": "Optimum ORT",        "query": '"ORTQuantizer"',                           "qualifier": "language:python"},
    # Other toolkits
    {"name": "intel_inc_pkg",               "tool": "Intel Neural Compressor", "query": '"neural_compressor"',                 "qualifier": "language:python"},
    {"name": "intel_inc_quantizer",         "tool": "Intel Neural Compressor", "query": '"INCQuantizer"',                      "qualifier": "language:python"},
    {"name": "nncf_import",                 "tool": "OpenVINO NNCF",      "query": '"import nncf"',                            "qualifier": "language:python"},
    {"name": "nncf_compress_weights",       "tool": "OpenVINO NNCF",      "query": '"nncf.compress_weights"',                  "qualifier": "language:python"},
    {"name": "exllamav2_convert",           "tool": "ExLlamaV2",          "query": '"exllamav2/convert"',                      "qualifier": ""},
    {"name": "exllamav2_pkg",               "tool": "ExLlamaV2",          "query": '"from exllamav2"',                         "qualifier": "language:python"},
    {"name": "eetq_import",                 "tool": "EETQ",               "query": '"from eetq"',                              "qualifier": "language:python"},
    {"name": "compressed_tensors_pkg",      "tool": "compressed-tensors", "query": '"from compressed_tensors"',                "qualifier": "language:python"},
    {"name": "amd_quark_pkg",               "tool": "AMD Quark",          "query": '"from quark.torch"',                       "qualifier": "language:python"},
    {"name": "onnxrt_quantize_dyn",         "tool": "ONNX Runtime",       "query": '"quantize_dynamic("',                      "qualifier": "language:python"},
    {"name": "onnxrt_quantize_static",      "tool": "ONNX Runtime",       "query": '"quantize_static("',                       "qualifier": "language:python"},
    {"name": "openvino_compress",           "tool": "OpenVINO",           "query": '"openvino" "compress_weights"',            "qualifier": "language:python"},
    {"name": "tensorrt_int8_calibrator",    "tool": "TensorRT",           "query": '"IInt8EntropyCalibrator"',                 "qualifier": ""},
    {"name": "llmcompressor_pkg",           "tool": "llm-compressor",     "query": '"llmcompressor"',                          "qualifier": "language:python"},
    {"name": "smoothquant_pkg",             "tool": "SmoothQuant",        "query": '"smoothquant"',                            "qualifier": ""},
]

ALL_SIGNALS: list[dict] = (
    [{**s, "category": "A"} for s in CATEGORY_A_SIGNALS]
  + [{**s, "category": "B"} for s in CATEGORY_B_SIGNALS]
)

# Post-extraction regexes (applied to text_match fragments)
OLLAMA_MODEL_RE = re.compile(
    r"(?:FROM\s+|ollama\s+(?:pull|run|create|push)\s+)([\w.~/-]+)(?::([\w.~/-]+))?",
    re.IGNORECASE,
)
GGUF_QUANT_RE = re.compile(r"\b(Q[0-9]+[_A-Z0-9]*|IQ[0-9]+[_A-Z0-9]*|F16|F32|BF16)\b")


# ===========================================================================
# Helpers
# ===========================================================================

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log(msg: str) -> None:
    line = f"[{utcnow_iso()}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_tokens() -> list[str]:
    """Collect every GitHub token we can find. Looks for GH_TOKEN_1..49 first
    (the multi-token convention used elsewhere in this project) and then
    GITHUB_TOKEN as a fallback. Returns the full list, deduped and stripped."""
    seen: set[str] = set()
    tokens: list[str] = []
    for i in range(1, 50):
        t = os.environ.get(f"GH_TOKEN_{i}", "").strip()
        if t and t not in seen:
            tokens.append(t); seen.add(t)
    t = os.environ.get("GITHUB_TOKEN", "").strip()
    if t and t not in seen:
        tokens.append(t); seen.add(t)
    if not tokens:
        raise RuntimeError(
            "No GitHub tokens found. export GH_TOKEN_1=... (and optionally "
            "GH_TOKEN_2..N) and/or GITHUB_TOKEN=... then re-run."
        )
    return tokens


def atomic_write(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# ===========================================================================
# Rate-limit-aware HTTP
# ===========================================================================

class Session:
    """Multi-token pool with per-token rate-limit handling.

    Each token has its own requests.Session, a last-used timestamp (enforcing
    TOKEN_MIN_INTERVAL_S between successive uses of THAT token), and a
    cool-down timestamp set on 403/429 responses. For every request we pick
    the token with the earliest ready time and sleep only if all are still
    cooling. With N tokens, sustained throughput is roughly
    N / TOKEN_MIN_INTERVAL_S req/s (e.g. 9 tokens -> ~4.3 req/s).

    A 403/429 on one token does NOT pause the others -- it just shifts that
    token's ready time forward. The picker hops to the next-soonest token."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = list(tokens)
        self.sessions: list[requests.Session] = []
        for tok in tokens:
            s = requests.Session()
            s.headers.update({
                "Accept": "application/vnd.github.v3.text-match+json",
                "Authorization": f"Bearer {tok}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": USER_AGENT,
            })
            self.sessions.append(s)
        n = len(tokens)
        self._last_used:      list[float] = [0.0] * n
        self._cooldown_until: list[float] = [0.0] * n

    def _pick_ready_idx(self) -> int:
        """Block until a token is ready, then return its index."""
        while True:
            now = time.time()
            best_idx = 0
            best_time = float("inf")
            for i in range(len(self.tokens)):
                ready_at = max(self._last_used[i] + TOKEN_MIN_INTERVAL_S,
                               self._cooldown_until[i])
                if ready_at < best_time:
                    best_time = ready_at
                    best_idx = i
            if best_time <= now:
                return best_idx
            # All tokens busy/cooling -- sleep until the next-soonest one
            # frees up, but don't oversleep in one chunk.
            time.sleep(min(best_time - now, 30.0))

    def get(self, url: str, params: dict | None = None) -> requests.Response | None:
        backoff = 30
        for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
            idx = self._pick_ready_idx()
            self._last_used[idx] = time.time()
            try:
                r = self.sessions[idx].get(url, params=params, timeout=HTTP_TIMEOUT_S)
            except requests.RequestException as e:
                log(f"  http exc tok#{idx+1} (attempt {attempt}): {e!r}; sleeping {backoff}s")
                time.sleep(min(backoff, MAX_BACKOFF_S))
                backoff = min(backoff * 2, MAX_BACKOFF_S)
                continue

            if r.status_code == 200:
                return r
            if r.status_code in (404, 451, 422):
                return r
            if r.status_code in (403, 429):
                retry_after = r.headers.get("Retry-After")
                reset = r.headers.get("X-RateLimit-Reset")
                remaining = r.headers.get("X-RateLimit-Remaining", "?")
                wait = 60.0
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        pass
                elif reset:
                    try:
                        wait = max(0.0, float(reset) - time.time()) + 5.0
                    except Exception:
                        pass
                wait = min(wait, MAX_BACKOFF_S)
                self._cooldown_until[idx] = time.time() + wait
                log(f"  tok#{idx+1} rate-limited ({r.status_code}; "
                    f"remaining={remaining}); cooling {wait:.0f}s; "
                    f"continuing on other tokens")
                # Do NOT sleep here; _pick_ready_idx will pick another
                # token that's ready.
                continue
            if r.status_code >= 500:
                log(f"  server {r.status_code} tok#{idx+1}; sleeping {backoff}s")
                time.sleep(min(backoff, MAX_BACKOFF_S))
                backoff = min(backoff * 2, MAX_BACKOFF_S)
                continue
            log(f"  unexpected {r.status_code} tok#{idx+1}: {r.text[:200]}; "
                f"sleeping {backoff}s")
            time.sleep(min(backoff, MAX_BACKOFF_S))
            backoff = min(backoff * 2, MAX_BACKOFF_S)
        return None


# ===========================================================================
# Extract model/quant from text-match fragments
# ===========================================================================

def extract_ollama_model_quant(fragment: str) -> tuple[str | None, str | None]:
    m = OLLAMA_MODEL_RE.search(fragment or "")
    if not m:
        return None, None
    return m.group(1), m.group(2)


def extract_gguf_quant(fragment: str) -> str | None:
    m = GGUF_QUANT_RE.search(fragment or "")
    return m.group(1) if m else None


# ===========================================================================
# In-memory state + checkpointing
# ===========================================================================

class State:
    def __init__(self) -> None:
        self.repos: dict[str, dict] = {}
        self.completed_signals: set[str] = set()
        self.metadata_fetched: set[str] = set()
        self.stats: Counter = Counter()

    def load(self) -> None:
        if JSONL_OUT.exists():
            with JSONL_OUT.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        self.repos[r["full_name"]] = r
                    except Exception:
                        pass
        if CHECKPOINT.exists():
            try:
                d = json.loads(CHECKPOINT.read_text())
                self.completed_signals = set(d.get("completed_signals", []))
                self.metadata_fetched = set(d.get("metadata_fetched", []))
                self.stats = Counter(d.get("stats", {}))
            except Exception as e:
                log(f"checkpoint load error: {e!r}")

    def save(self) -> None:
        body = "\n".join(json.dumps(r) for r in self.repos.values())
        atomic_write(JSONL_OUT, body + ("\n" if body else ""))
        atomic_write(CHECKPOINT, json.dumps({
            "completed_signals":  sorted(self.completed_signals),
            "metadata_fetched":   sorted(self.metadata_fetched),
            "stats":              dict(self.stats),
            "updated_at":         utcnow_iso(),
        }, indent=2))

    def add_hit(self, signal: dict, item: dict) -> None:
        repo = item.get("repository") or {}
        full_name = repo.get("full_name")
        if not full_name:
            return
        path = item.get("path", "")
        fragments: list[str] = []
        for tm in (item.get("text_matches") or []):
            frag = tm.get("fragment")
            if frag:
                fragments.append(frag)

        sig_record: dict[str, Any] = {
            "signal":       signal["name"],
            "tool":         signal["tool"],
            "category":     signal["category"],
            "matched_file": path,
            "fragments":    fragments[:3],
        }
        model_name = ollama_tag = gguf_tag = None
        for frag in fragments:
            if signal["category"] == "A":
                m, t = extract_ollama_model_quant(frag)
                if m and not model_name:
                    model_name = m
                if t and not ollama_tag:
                    ollama_tag = t
            else:
                g = extract_gguf_quant(frag)
                if g and not gguf_tag:
                    gguf_tag = g
        if model_name:
            sig_record["ollama_model_name"] = model_name
        if ollama_tag:
            sig_record["ollama_quant_tag"] = ollama_tag
        if gguf_tag:
            sig_record["gguf_quant_tag"] = gguf_tag

        if full_name not in self.repos:
            self.repos[full_name] = {
                "full_name":      full_name,
                "html_url":       repo.get("html_url", f"https://github.com/{full_name}"),
                "categories":     [],
                "signals":        [],
                "stars":          None,
                "pushed_at":      None,
                "already_counted": None,   # filled by subtract_analysis_set
                "first_seen":     utcnow_iso(),
            }
        rec = self.repos[full_name]
        if "already_counted" not in rec:
            rec["already_counted"] = None
        if signal["category"] not in rec["categories"]:
            rec["categories"].append(signal["category"])
        rec["signals"].append(sig_record)
        self.stats[f"category_{signal['category']}_hits"] += 1
        self.stats[f"signal:{signal['name']}"] += 1


# ===========================================================================
# Code-search loop + repo-metadata fetch
# ===========================================================================

def run_signal(session: Session, state: State, signal: dict) -> int:
    base_q = signal["query"]
    qualifier = signal.get("qualifier", "")
    q = f"{base_q} {qualifier}".strip() if qualifier else base_q
    cap = MAX_PAGES * PER_PAGE   # GitHub caps code-search at 1000 results
    n_hits = 0
    for page in range(1, MAX_PAGES + 1):
        params = {"q": q, "per_page": PER_PAGE, "page": page}
        r = session.get(f"{GITHUB_API}/search/code", params=params)
        if r is None or r.status_code != 200:
            sc = r.status_code if r is not None else "no-response"
            body = r.text[:200] if r is not None else ""
            log(f"  '{signal['name']}' page {page}: failed ({sc}) {body}")
            return n_hits
        data = r.json()
        items = data.get("items", [])
        n_hits += len(items)
        for it in items:
            state.add_hit(signal, it)
        # GitHub code search returns ragged/short pages: a page with fewer than
        # PER_PAGE items is NOT a reliable end-of-results signal. Continue
        # paginating while items were returned AND we haven't exhausted
        # min(total_count, 1000). Stop only when items is empty or the cap
        # is reached.
        total = int(data.get("total_count", 0) or 0)
        if not items:
            break
        if page * PER_PAGE >= min(total, cap):
            break
    return n_hits


def load_analysis_set_names(path: Path, field: str) -> set[str]:
    """Read the existing HF-anchored analysis-set JSONL and return the set of
    GitHub full_names it contains, lower-cased for case-insensitive matching.
    Pure file I/O; no API calls."""
    names: set[str] = set()
    if not path.exists():
        log(f"  analysis-set file missing: {path}")
        return names
    n_lines = n_with_field = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            v = r.get(field)
            if isinstance(v, str) and v:
                names.add(v.lower())
                n_with_field += 1
    log(f"  analysis-set: {n_lines:,} lines read, "
        f"{n_with_field:,} with non-empty '{field}', "
        f"{len(names):,} unique names")
    return names


def subtract_analysis_set(state: State) -> tuple[int, int]:
    """Mark every mined repo with already_counted = True / False (case-
    insensitive full_name match against ANALYSIS_SET_JSONL). Returns
    (n_already_counted, n_missed). Offline -- no API calls."""
    log(f"subtract: reading {ANALYSIS_SET_JSONL}")
    analysis_names = load_analysis_set_names(ANALYSIS_SET_JSONL, ANALYSIS_SET_FIELD)
    n_in = n_out = 0
    for fn, rec in state.repos.items():
        in_set = fn.lower() in analysis_names
        rec["already_counted"] = bool(in_set)
        if in_set:
            n_in += 1
        else:
            n_out += 1
    state.stats["already_counted"] = n_in
    state.stats["missed"]          = n_out
    log(f"  already_counted = {n_in:,}; missed = {n_out:,}")
    return n_in, n_out


def write_missed_jsonl(state: State) -> int:
    """Write the subset of mined repos whose already_counted is False. These
    are the repos missed by the HF-anchored pipeline -- the new corpus the
    side study is about."""
    rows = [r for r in state.repos.values() if r.get("already_counted") is False]
    body = "\n".join(json.dumps(r) for r in rows)
    atomic_write(MISSED_JSONL, body + ("\n" if body else ""))
    log(f"  wrote {len(rows):,} repos to {MISSED_JSONL}")
    return len(rows)


def fetch_metadata(session: Session, state: State, only_missed: bool = True) -> None:
    """Fetch /repos/{owner}/{name} metadata. When only_missed is True (the
    default after subtraction), this is restricted to repos with
    already_counted == False -- the repos we actually care about for the
    side study."""
    if only_missed:
        todo = [
            fn for fn, r in state.repos.items()
            if r.get("already_counted") is False and fn not in state.metadata_fetched
        ]
        log(f"metadata: {len(todo)} MISSED repos to fetch (skipping already_counted)")
    else:
        todo = [fn for fn in state.repos if fn not in state.metadata_fetched]
        log(f"metadata: {len(todo)} unique repos to fetch")
    for i, fn in enumerate(todo, 1):
        r = session.get(f"{GITHUB_API}/repos/{fn}")
        state.metadata_fetched.add(fn)
        if r is None:
            continue
        if r.status_code == 200:
            d = r.json()
            state.repos[fn]["stars"]      = d.get("stargazers_count")
            state.repos[fn]["pushed_at"]  = d.get("pushed_at")
            state.repos[fn]["created_at"] = d.get("created_at")
            state.repos[fn]["forks"]      = d.get("forks_count")
            state.repos[fn]["archived"]   = d.get("archived")
            state.repos[fn]["fork"]       = d.get("fork")
            state.repos[fn]["language"]   = d.get("language")
        else:
            state.repos[fn]["repo_status"] = r.status_code
        if i % 100 == 0:
            log(f"  metadata: {i}/{len(todo)}")
            state.save()


# ===========================================================================
# Summary CSV
# ===========================================================================

def write_summary(state: State) -> None:
    cat_counts: Counter = Counter()
    tool_counts: Counter = Counter()
    signal_counts: Counter = Counter()
    ollama_tags: Counter = Counter()
    gguf_tags: Counter = Counter()
    # Category x analysis-set membership breakdown:
    #   key = "<cat_set>:already_counted" or "<cat_set>:missed" or "<cat_set>:unknown"
    coverage_breakdown: Counter = Counter()
    for r in state.repos.values():
        cats = tuple(sorted(r.get("categories") or []))
        cat_key = ",".join(cats) or "none"
        cat_counts[cat_key] += 1
        ac = r.get("already_counted")
        if ac is True:
            tag = "already_counted"
        elif ac is False:
            tag = "missed"
        else:
            tag = "unknown"
        coverage_breakdown[f"{cat_key}:{tag}"] += 1
        seen_tools_in_repo: set[str] = set()
        seen_signals_in_repo: set[str] = set()
        for s in r["signals"]:
            seen_tools_in_repo.add(s["tool"])
            seen_signals_in_repo.add(s["signal"])
            if s.get("ollama_quant_tag"):
                ollama_tags[s["ollama_quant_tag"]] += 1
            if s.get("gguf_quant_tag"):
                gguf_tags[s["gguf_quant_tag"]] += 1
        for t in seen_tools_in_repo:
            tool_counts[t] += 1
        for sn in seen_signals_in_repo:
            signal_counts[sn] += 1

    n_already = sum(1 for r in state.repos.values() if r.get("already_counted") is True)
    n_missed  = sum(1 for r in state.repos.values() if r.get("already_counted") is False)
    n_unknown = sum(1 for r in state.repos.values() if r.get("already_counted") is None)

    with SUMMARY_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["unique_mined_repos", len(state.repos)])
        w.writerow(["already_counted_in_analysis_set", n_already])
        w.writerow(["missed_by_analysis_set",          n_missed])
        if n_unknown:
            w.writerow(["unknown_analysis_set_membership", n_unknown])
        w.writerow(["---", "---"])
        for cat, n in sorted(cat_counts.items()):
            w.writerow([f"category:{cat}", n])
        w.writerow(["---", "---"])
        for key, n in sorted(coverage_breakdown.items()):
            w.writerow([f"coverage:{key}", n])
        w.writerow(["---", "---"])
        for tool, n in tool_counts.most_common():
            w.writerow([f"repos_with_tool:{tool}", n])
        w.writerow(["---", "---"])
        for sig, n in signal_counts.most_common():
            w.writerow([f"repos_with_signal:{sig}", n])
        w.writerow(["---", "---"])
        for tag, n in ollama_tags.most_common():
            w.writerow([f"ollama_quant_tag:{tag}", n])
        w.writerow(["---", "---"])
        for tag, n in gguf_tags.most_common():
            w.writerow([f"gguf_quant_tag:{tag}", n])


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== miner start ===")
    log(f"OUTPUT_DIR = {OUTPUT_DIR}")
    log(f"signals: {len(ALL_SIGNALS)} total "
        f"({len(CATEGORY_A_SIGNALS)} category-A, {len(CATEGORY_B_SIGNALS)} category-B)")

    tokens = load_tokens()
    log(f"GH tokens loaded: {len(tokens)} "
        f"(effective ~{len(tokens) / TOKEN_MIN_INTERVAL_S:.1f} req/s "
        f"= ~{60 * len(tokens) / TOKEN_MIN_INTERVAL_S:.0f} req/min)")
    session = Session(tokens)

    state = State()
    state.load()
    log(f"resume: {len(state.completed_signals)} signals already done; "
        f"{len(state.repos)} repos in jsonl")

    try:
        for i, sig in enumerate(ALL_SIGNALS, 1):
            if sig["name"] in state.completed_signals:
                continue
            log(f"[{i:>3}/{len(ALL_SIGNALS)}] {sig['category']} "
                f"{sig['name']:<32} {sig['query']}")
            try:
                n = run_signal(session, state, sig)
                log(f"  -> {n} hits this signal; repos so far = {len(state.repos)}")
            except Exception as e:
                log(f"  signal '{sig['name']}' raised {e!r}; skipping")
                continue
            state.completed_signals.add(sig["name"])
            state.save()

        log("--- offline subtraction vs analysis-set ---")
        subtract_analysis_set(state)
        state.save()

        log("--- writing missed_repos.jsonl ---")
        write_missed_jsonl(state)

        if FETCH_METADATA:
            log("--- fetching repo metadata (missed repos only) ---")
            fetch_metadata(session, state, only_missed=True)
            state.save()
            # Refresh missed_repos.jsonl now that metadata is filled in
            write_missed_jsonl(state)

        log("--- writing summary.csv ---")
        write_summary(state)
        state.save()
        log(f"DONE. unique mined = {len(state.repos)}; "
            f"missed = {state.stats.get('missed', 0)}; "
            f"already_counted = {state.stats.get('already_counted', 0)}")
        log(f"outputs: {JSONL_OUT}, {MISSED_JSONL}, {SUMMARY_CSV}, "
            f"{CHECKPOINT}, {LOG_FILE}")

    except KeyboardInterrupt:
        log("interrupted by user; saving partial state and exiting")
        state.save()
        sys.exit(130)


if __name__ == "__main__":
    main()

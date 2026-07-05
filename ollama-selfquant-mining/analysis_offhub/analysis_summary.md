# Off-hub repo analysis

Reference date for activity buckets: 2026-06-16.
Input: `missed_repos.jsonl` (8,424 repos missed by the HF-anchored pipeline).

## 1. Adoption taxonomy & method preference

Each repo is assigned ONE adoption mode using a priority rule:

1. `self_quantized` — any Category B signal
2. `ollama_obtains_model` — any Ollama obtain/produce signal (`Modelfile FROM`, `ollama pull`, `ollama create`, `ollama push`, registry refs)
3. `ollama_backend_only` — Ollama usage only via client / REST / LangChain / LlamaIndex / `ollama run`

| mode | count | pct of all off-hub |
|---|---:|---:|
| `self_quantized` | 3,171 | 37.64% |
| `ollama_obtains_model` | 659 | 7.82% |
| `ollama_backend_only` | 4,594 | 54.53% |
| unclassified | 0 | 0.00% |
| **total** | **8,424** | 100% |

All repos were assigned to one of the three modes (0 unclassified, as expected).

**Overlap (`self_quantize AND serve via Ollama`):** `n_self_quant_also_ollama` = **29** (= 0.91% of `self_quantized`). These are repos that would have been `ollama_obtains_model` without the priority rule.

**Obtain-or-produce subset:** `self_quantized` + `ollama_obtains_model` = 3,830 repos. This is the universe for the method/precision tables below and the input for the later temporal pass (`offhub_obtain_repo_ids.txt`).

### Self-quant method-family distribution

Across `self_quantized` repos (3,171); each repo counted once per family it uses (a repo can use >1).

| method family | repos | pct of self_quantized |
|---|---:|---:|
| GPTQ | 677 | 21.35% |
| ONNXRuntime | 651 | 20.53% |
| llama.cpp/GGUF-convert | 513 | 16.18% |
| torchao | 360 | 11.35% |
| other | 289 | 9.11% |
| AWQ | 243 | 7.66% |
| OpenVINO/NNCF | 209 | 6.59% |
| quanto | 128 | 4.04% |
| Intel-NC | 94 | 2.96% |
| SmoothQuant | 77 | 2.43% |
| ExLlamaV2 | 74 | 2.33% |
| compressed-tensors | 72 | 2.27% |
| AMD-Quark | 70 | 2.21% |

### Precision distribution (where identifiable)

Computed across the obtain-or-produce subset (3,830 repos). **903** repos (23.58%) carry an identifiable precision tag. Size tags such as `7b`/`8b`/`1.5B` are NOT counted as precisions (they're model size).

| precision | repos | pct of repos w/ identifiable precision |
|---|---:|---:|
| 8bit | 553 | 61.24% |
| 4bit | 200 | 22.15% |
| F16 | 97 | 10.74% |
| Q4_K_M | 64 | 7.09% |
| FP16 | 50 | 5.54% |
| Q8_0 | 34 | 3.77% |
| FP8 | 33 | 3.65% |
| Q-other | 30 | 3.32% |
| BF16 | 26 | 2.88% |
| F32 | 22 | 2.44% |
| FP32 | 21 | 2.33% |
| Q5_K_M | 11 | 1.22% |
| Q4_0 | 10 | 1.11% |
| IQ-other | 5 | 0.55% |
| Q6_K | 4 | 0.44% |
| IQ4_XS | 3 | 0.33% |
| Q4_K_S | 3 | 0.33% |
| Q2_K | 3 | 0.33% |
| Q4_1 | 1 | 0.11% |
| Q5_1 | 1 | 0.11% |
| Q3_K_S | 1 | 0.11% |
| IQ2_XXS | 1 | 0.11% |
| IQ3_XXS | 1 | 0.11% |

## 2. Repo characteristics by adoption mode

Stars (median / mean / p90 / p99 / share at each cut), forks (median, mean), archived & fork share, and pushed-at recency (reference date 2026-06-16).

| cohort | n | stars med | stars mean | p90 | p99 | %=0 | %>=1 | %>=5 | %>=10 | forks med | forks mean | %archived | %fork | active <6mo | <12mo | older |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `all_offhub` | 8,424 | 0.00 | 138.26 | 36.70 | 2330.01 | 58.10% | 41.90% | 21.58% | 16.86% | 0.00 | 21.64 | 0.00% | 0.00% | 4,020 | 1,553 | 2,851 |
| `self_quantized` | 3,171 | 1 | 248.12 | 101.00 | 6339.80 | 48.91% | 51.09% | 30.65% | 24.88% | 0 | 36.22 | 0.00% | 0.00% | 1,499 | 495 | 1,177 |
| `ollama_obtains_model` | 659 | 1 | 167.79 | 95.20 | 2357.88 | 49.32% | 50.68% | 27.92% | 23.37% | 0 | 35.42 | 0.00% | 0.00% | 366 | 91 | 202 |
| `ollama_backend_only` | 4,594 | 0.00 | 58.20 | 11.00 | 687.99 | 65.69% | 34.31% | 14.41% | 10.38% | 0.00 | 9.60 | 0.00% | 0.00% | 2,155 | 967 | 1,472 |

### Top-15 primary languages by cohort

**`all_offhub`** (8,424 repos):

| language | count |
|---|---:|
| Python | 5,194 |
| JavaScript | 608 |
| Jupyter Notebook | 575 |
| TypeScript | 547 |
| HTML | 269 |
| C++ | 223 |
| Go | 202 |
| Unknown | 148 |
| Shell | 127 |
| Rust | 100 |
| C | 57 |
| Java | 53 |
| C# | 29 |
| CSS | 17 |
| Dart | 17 |

**`self_quantized`** (3,171 repos):

| language | count |
|---|---:|
| Python | 2,299 |
| Jupyter Notebook | 247 |
| C++ | 204 |
| Unknown | 59 |
| TypeScript | 53 |
| HTML | 44 |
| Rust | 41 |
| C | 36 |
| Go | 35 |
| JavaScript | 30 |
| Shell | 28 |
| Cuda | 10 |
| Java | 9 |
| C# | 9 |
| Makefile | 8 |

**`ollama_obtains_model`** (659 repos):

| language | count |
|---|---:|
| Python | 298 |
| TypeScript | 44 |
| Go | 43 |
| Jupyter Notebook | 41 |
| JavaScript | 41 |
| Unknown | 38 |
| Shell | 34 |
| HTML | 34 |
| Rust | 15 |
| Java | 13 |
| PowerShell | 7 |
| C++ | 4 |
| R | 4 |
| Roff | 3 |
| CSS | 3 |

**`ollama_backend_only`** (4,594 repos):

| language | count |
|---|---:|
| Python | 2,597 |
| JavaScript | 537 |
| TypeScript | 450 |
| Jupyter Notebook | 287 |
| HTML | 191 |
| Go | 124 |
| Shell | 65 |
| Unknown | 51 |
| Rust | 44 |
| Java | 31 |
| C# | 18 |
| C | 18 |
| C++ | 15 |
| Dart | 14 |
| Vue | 14 |

### Created_at by year

| cohort | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `all_offhub` | 1 | 3 | 6 | 6 | 10 | 8 | 21 | 18 | 30 | 46 | 60 | 141 | 535 | 1792 | 3215 | 2532 |
| `self_quantized` | 1 | 2 | 0 | 2 | 6 | 2 | 9 | 6 | 16 | 30 | 46 | 102 | 397 | 612 | 1026 | 914 |
| `ollama_obtains_model` | 0 | 1 | 1 | 2 | 1 | 3 | 3 | 5 | 8 | 5 | 5 | 14 | 25 | 155 | 216 | 215 |
| `ollama_backend_only` | 0 | 0 | 5 | 2 | 3 | 3 | 9 | 7 | 6 | 11 | 9 | 25 | 113 | 1025 | 1973 | 1403 |


## 3. LLM-stack-only precision

The main paper's precision study covers GGUF/GPTQ/AWQ-style **LLM-serving** quantization. The overall off-hub distribution mixes that with general-ML toolkits (ONNXRuntime, OpenVINO/NNCF, Intel-NC, AMD-Quark, torchao, quanto, SmoothQuant, TensorRT) that quantize a wide variety of models, including non-LLM. This view restricts precision evidence to signals whose method family is one of **{llama.cpp/GGUF-convert, GPTQ, AWQ, ExLlamaV2, compressed-tensors}** or to Ollama obtain-signals whose text shows a GGUF indicator (`.gguf` or a `Q*`/`IQ*` tag).

**Coverage**

| view | basis | n |
|---|---|---:|
| overall | obtain-or-produce | 3,830 |
| overall | with identifiable precision | 903 (23.58%) |
| llmstack | in LLM-stack scope | 1,552 (40.52% of obtain-or-produce) |
| llmstack | with identifiable precision | 276 (17.78% of stack scope) |

**Bit-width distribution (each repo counted once per bucket it uses)**

| bit-width | overall | overall % | llm-stack | llm-stack % |
|---|---:|---:|---:|---:|
| 2-bit | 4 | 0.44% | 3 | 1.09% |
| 3-bit | 2 | 0.22% | 2 | 0.72% |
| 4-bit | 274 | 30.34% | 130 | 47.10% |
| 5-bit | 12 | 1.33% | 9 | 3.26% |
| 6-bit | 4 | 0.44% | 2 | 0.72% |
| 8-bit | 584 | 64.67% | 48 | 17.39% |
| FP16 | 151 | 16.72% | 117 | 42.39% |
| FP32 | 42 | 4.65% | 22 | 7.97% |
| FP8 | 33 | 3.65% | 2 | 0.72% |
| other | 34 | 3.77% | 25 | 9.06% |

**Reading:** the overall view is dominated by the **8-bit** bucket (584 repos = 64.67% of precision-identifiable repos), inflated by ONNX/OpenVINO/Intel-NC integer signals across general-ML workloads. Restricted to the LLM-serving stack, the leader is **4-bit** (130 repos = 47.10% of LLM-stack precision repos), which is apples-to-apples with the main paper's GGUF/GPTQ/AWQ study.

**LLM-stack raw-tag top 10**

| tag | repos | pct |
|---|---:|---:|
| F16 | 94 | 34.06% |
| 4bit | 66 | 23.91% |
| Q4_K_M | 56 | 20.29% |
| Q8_0 | 33 | 11.96% |
| BF16 | 22 | 7.97% |
| FP16 | 22 | 7.97% |
| F32 | 21 | 7.61% |
| Q-other | 21 | 7.61% |
| 8bit | 17 | 6.16% |
| Q4_0 | 8 | 2.90% |

## 4. Method-family model domain

Keyword-based heuristic over each `self_quantized` repo's `full_name`, `language`, matched-file paths, and signal fragments. Priority order: `likely_LLM` > `whisper_audio` > `embedding` > `likely_nonLLM` > `unknown`. The heuristic is intentionally shallow; the `unknown` rate per family is the honest measure of how much we can't tell.

**Overall self_quantized domain mix**

| bucket | count | pct of self_quantized |
|---|---:|---:|
| likely_LLM | 1,932 | 60.93% |
| whisper_audio | 20 | 0.63% |
| embedding | 36 | 1.14% |
| likely_nonLLM | 167 | 5.27% |
| unknown | 1,016 | 32.04% |
| **total** | **3,171** | 100% |

**Per method family**

| method family | likely_LLM | whisper | embedding | likely_nonLLM | unknown | total | %LLM | %nonLLM | %unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPTQ | 661 | 0 | 0 | 0 | 16 | 677 | 97.64% | 0.00% | 2.36% |
| ONNXRuntime | 197 | 16 | 34 | 30 | 374 | 651 | 30.26% | 4.61% | 57.45% |
| llama.cpp/GGUF-convert | 459 | 2 | 0 | 0 | 52 | 513 | 89.47% | 0.00% | 10.14% |
| torchao | 149 | 1 | 0 | 7 | 203 | 360 | 41.39% | 1.94% | 56.39% |
| other | 120 | 0 | 0 | 49 | 120 | 289 | 41.52% | 16.96% | 41.52% |
| AWQ | 240 | 0 | 0 | 0 | 3 | 243 | 98.77% | 0.00% | 1.23% |
| OpenVINO/NNCF | 42 | 1 | 2 | 70 | 94 | 209 | 20.10% | 33.49% | 44.98% |
| quanto | 42 | 0 | 0 | 5 | 81 | 128 | 32.81% | 3.91% | 63.28% |
| Intel-NC | 35 | 0 | 0 | 8 | 51 | 94 | 37.23% | 8.51% | 54.26% |
| SmoothQuant | 62 | 0 | 0 | 1 | 14 | 77 | 80.52% | 1.30% | 18.18% |
| ExLlamaV2 | 74 | 0 | 0 | 0 | 0 | 74 | 100.00% | 0.00% | 0.00% |
| compressed-tensors | 54 | 0 | 0 | 0 | 18 | 72 | 75.00% | 0.00% | 25.00% |
| AMD-Quark | 63 | 0 | 0 | 0 | 7 | 70 | 90.00% | 0.00% | 10.00% |

**Reading:** families clearly in the LLM-serving stack (AWQ, ExLlamaV2, GPTQ, compressed-tensors, llama.cpp/GGUF-convert) are nearly all `likely_LLM`. The general-ML families flagged for review are: **OpenVINO/NNCF** (33% non-LLM, 45% unknown); **other** (17% non-LLM, 42% unknown); **Intel-NC** (9% non-LLM, 54% unknown); **ONNXRuntime** (5% non-LLM, 57% unknown); **quanto** (4% non-LLM, 63% unknown); **torchao** (2% non-LLM, 56% unknown). The high-unknown share for some families means a keyword scan can't tell what they quantize -- treat their precision contributions as out-of-scope unless manually re-checked.


## 5. LLM-stack temporal (metadata-based)

**Scope used here: EXPANDED LLM-stack** = STRICT `{AWQ, ExLlamaV2, GPTQ, compressed-tensors, llama.cpp/GGUF-convert}` + `{AMD-Quark, SmoothQuant}` + Ollama obtain-signals whose text shows a GGUF indicator (`.gguf` or `Q*`/`IQ*` tag). Temporal evidence is metadata-only (`created_at` for inception, `pushed_at` for activity). No git history was mined for this section.

LLM subset size: **1,657** of 3,830 obtain-or-produce repos (43.26%). Split: `self_quantized` = 1,603, `ollama_obtains_model` = 54.

**created_at counts by year (LLM subset)**

| year | self_quantized | ollama_obtains_model | total |
|---|---:|---:|---:|
| 2021 | 11 | 0 | 11 |
| 2022 | 18 | 0 | 18 |
| 2023 | 246 | 1 | 247 |
| 2024 | 328 | 17 | 345 |
| 2025 | 478 | 21 | 499 |
| 2026 | 487 | 14 | 501 |
| other year | 35 | 1 | 36 |
| **total** | **1,603** | **54** | **1,657** |

**pushed_at recency profile (LLM subset, reference 2026-06-16)**

| recency bucket | self_quantized | % | ollama_obtains_model | % |
|---|---:|---:|---:|---:|
| active_last_6mo | 800 | 49.91% | 23 | 42.59% |
| active_last_12mo | 208 | 12.98% | 9 | 16.67% |
| older_than_12mo | 595 | 37.12% | 22 | 40.74% |
| unknown | 0 | 0.00% | 0 | 0.00% |

**Reading:** 1,000 of 1,657 LLM-subset repos (60.35%) were created in 2025-2026, and 823 (49.67%) have a push in the last 6 months. Treat these as lower-bound adoption-rate indicators: repository creation date is a lagging signal of when a tool became popular enough to spawn a project, and the off-hub corpus is itself filtered for currently-public, non-archived, non-fork repos.

## 6. Precision sensitivity: STRICT vs EXPANDED

Recomputes the bit-width distribution under two stack definitions to show the 4-bit-leads finding is robust to including the two LLM-leaning families (SmoothQuant, AMD-Quark).

| coverage | STRICT | EXPANDED |
|---|---:|---:|
| in-scope (obtain-or-produce) | 1,552 (40.52%) | 1,657 (43.26%) |
| with identifiable precision | 276 (17.78% of scope) | 294 (17.74% of scope) |

| bit-width | STRICT count | STRICT % | EXPANDED count | EXPANDED % |
|---|---:|---:|---:|---:|
| 2-bit | 3 | 1.09% | 3 | 1.02% |
| 3-bit | 2 | 0.72% | 2 | 0.68% |
| 4-bit | 130 | 47.10% | 136 | 46.26% |
| 5-bit | 9 | 3.26% | 9 | 3.06% |
| 6-bit | 2 | 0.72% | 2 | 0.68% |
| 8-bit | 48 | 17.39% | 61 | 20.75% |
| FP16 | 117 | 42.39% | 120 | 40.82% |
| FP32 | 22 | 7.97% | 22 | 7.48% |
| FP8 | 2 | 0.72% | 10 | 3.40% |
| other | 25 | 9.06% | 26 | 8.84% |

**Reading:** under STRICT, **4-bit leads** (130 repos, 47.10% of precision-identifiable LLM-stack repos), with 4-bit at 47.10%. Under EXPANDED, **4-bit leads** (136 repos, 46.26%), with 4-bit at 46.26%. Adding SmoothQuant and AMD-Quark does not flip the leader -- the 4-bit-leads finding is stable across both stack definitions.

## 7. Method-family ranking, two views

Each `self_quantized` repo is counted once per method family it uses. View A is the full obtain-or-produce ecosystem (all families). View B restricts to the EXPANDED LLM-stack (7 families = STRICT + SmoothQuant + AMD-Quark). `%likely_LLM` is the share of each family's `self_quantized` repos that the keyword heuristic in Section 4 classifies as `likely_LLM` -- shown alongside so the LLM-scoping is transparent.

### View A — full obtain-or-produce ecosystem

| rank | method family | n repos | in STRICT | in EXPANDED | %likely_LLM (self_quant) |
|---:|---|---:|:---:|:---:|---:|
| 1 | GPTQ | 677 | ✓ | ✓ | 97.64% |
| 2 | ONNXRuntime | 651 |  |  | 30.26% |
| 3 | llama.cpp/GGUF-convert | 513 | ✓ | ✓ | 89.47% |
| 4 | torchao | 360 |  |  | 41.39% |
| 5 | other | 289 |  |  | 41.52% |
| 6 | AWQ | 243 | ✓ | ✓ | 98.77% |
| 7 | OpenVINO/NNCF | 209 |  |  | 20.10% |
| 8 | quanto | 128 |  |  | 32.81% |
| 9 | Intel-NC | 94 |  |  | 37.23% |
| 10 | SmoothQuant | 77 |  | ✓ | 80.52% |
| 11 | ExLlamaV2 | 74 | ✓ | ✓ | 100.00% |
| 12 | compressed-tensors | 72 | ✓ | ✓ | 75.00% |
| 13 | AMD-Quark | 70 |  | ✓ | 90.00% |

### View B — EXPANDED LLM-stack only

| rank | method family | n repos | in STRICT | %likely_LLM (self_quant) |
|---:|---|---:|:---:|---:|
| 1 | GPTQ | 677 | ✓ | 97.64% |
| 2 | llama.cpp/GGUF-convert | 513 | ✓ | 89.47% |
| 3 | AWQ | 243 | ✓ | 98.77% |
| 4 | SmoothQuant | 77 |  | 80.52% |
| 5 | ExLlamaV2 | 74 | ✓ | 100.00% |
| 6 | compressed-tensors | 72 | ✓ | 75.00% |
| 7 | AMD-Quark | 70 |  | 90.00% |

**Reading:** in View A the top of the ecosystem is dominated by general-ML families (ONNXRuntime, OpenVINO/NNCF, torchao, quanto) whose `%likely_LLM` is well below 50%. View B drops those entirely and shows the LLM-stack ranking: a clean ordering of GPTQ, AWQ, llama.cpp/GGUF-convert, ExLlamaV2, compressed-tensors plus the two LLM-leaning additions. This is the ranking to cite alongside the main paper.


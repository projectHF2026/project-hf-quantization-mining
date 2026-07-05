# RQ2 Phase 1 Pilot - Two-Layer Temporal Mining

> **Scope note:** Phase 1 mines git histories to test whether
> the cohort signal observed in Phase 0 reflects actual
> temporal adoption patterns. First-signal dates are lower
> bounds (git history can be rebased), and dates are restricted
> to the default branch. The pilot tests on 100 stratified
> repos before any decision to scale to 20,704.

## Inputs used

- `/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/rq_analysis/rq2/phase1_pilot/pilot_repos.csv` (7,715 bytes)
- `/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/rq_analysis/rq2/phase1_pilot/pilot_clone_log.jsonl` (28,355 bytes)
- `/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/rq_analysis/rq2/phase1_pilot/results/layer1_method_signals.jsonl` (133,310 bytes)
- `/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/rq_analysis/rq2/phase1_pilot/results/layer2_model_signals.jsonl` (157,105 bytes)

## Technical feasibility

### Clone success rate per stratum

| stratum | success | 404 | timeout | auth_required | other_error | success rate |
|---|---:|---:|---:|---:|---:|---:|
| GGUF | 19 | 1 | 0 | 0 | 0 | 95.0% |
| BitsAndBytes | 19 | 1 | 0 | 0 | 0 | 95.0% |
| GPTQ | 20 | 0 | 0 | 0 | 0 | 100.0% |
| AWQ | 19 | 0 | 0 | 0 | 1 | 95.0% |
| Other | 20 | 0 | 0 | 0 | 0 | 100.0% |
| **Overall** | **97** | 2 | 0 | 0 | 1 | **97.0%** |

Target: ≥85% per stratum. **PASS**.

### Layer 1 mining success rate per stratum

| stratum | success | other_error | success rate |
|---|---:|---:|---:|
| GGUF | 95 | 0 | 100.0% |
| BitsAndBytes | 95 | 0 | 100.0% |
| GPTQ | 100 | 0 | 100.0% |
| AWQ | 95 | 0 | 100.0% |
| Other | 100 | 0 | 100.0% |

Target: ≥80% per stratum. **PASS**.

### Layer 2 first_model_signal_method distribution

| stratum | pickaxe_exact | head_fallback | not_found_in_history | pickaxe_exact % |
|---|---:|---:|---:|---:|
| GGUF | 31 | 2 | 0 | 93.9% |
| BitsAndBytes | 51 | 5 | 0 | 91.1% |
| GPTQ | 166 | 1 | 0 | 99.4% |
| AWQ | 17 | 12 | 1 | 56.7% |
| Other | 44 | 1 | 5 | 88.0% |
| **Overall** | **309** | 21 | 6 | **92.0%** |

Target: pickaxe_exact ≥70% overall. **PASS**.

### Runtime and disk projection

- Total pilot clone seconds: 367s (6.1 min)
- Mean clone size: 273.6 MB
- Projected full-run clone wall clock (× 207.0): **21.1 hours**
- Projected full-run disk: **5365.4 GB**

Target: full-run wall clock ≤24h. **PASS**.

### Spot-check verification

Sampled 4 `pickaxe_exact` pairs per stratum from Layer 2 (seed=42), verified that the model_id appears in `git show <commit>:<matched_file>`. Result: **20/20 passed** (target ≥18/20).

| # | repo | model_id | file | commit | verify |
|---|---|---|---|---|---|
| 1 | `nrl-ai/llama-assistant` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | `llama_assistant/config.py` | `ab867101` | ✓ |
| 2 | `arnyigor/ollamaagent` | `janhq/Jan-v1-4B-GGUF` | `download_model.py` | `c5dd81b0` | ✓ |
| 3 | `AtakanTekparmak/klippa-presentation` | `NousResearch/Hermes-3-Llama-3.1-8B-GGUF` | `src/config.py` | `fd1b0629` | ✓ |
| 4 | `nrl-ai/llama-assistant` | `hugging-quants/Llama-3.2-1B-Instruct-Q4_` | `llama_assistant/config.py` | `6d632ac6` | ✓ |
| 5 | `Jaxy205/TriTueNhanTao` | `unsloth/Llama-3.2-90B-Vision-Instruct-bn` | `nb/Qwen3_VL_(8B)-Vision.ipynb` | `ad230f98` | ✓ |
| 6 | `MixNatchapol/superai_s5_thaicaption` | `unsloth/llava-v1.6-mistral-7b-hf-bnb-4bi` | `tuning34.ipynb` | `88d66fa3` | ✓ |
| 7 | `MixNatchapol/superai_s5_thaicaption` | `unsloth/llava-1.5-7b-hf-bnb-4bit` | `tuning34.ipynb` | `88d66fa3` | ✓ |
| 8 | `MixNatchapol/superai_s5_thaicaption` | `unsloth/Llama-3.2-90B-Vision-bnb-4bit` | `tuning34.ipynb` | `88d66fa3` | ✓ |
| 9 | `Irus-ls/GaLorePlus` | `Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8` | `src/llmtuner/extras/constants.py` | `5b400698` | ✓ |
| 10 | `authurlord/MELD` | `Qwen/Qwen1.5-72B-Chat-AWQ` | `src/llamafactory/extras/constants.py` | `eab50970` | ✓ |
| 11 | `Irus-ls/GaLorePlus` | `Qwen/Qwen-7B-Chat-Int8` | `src/llmtuner/extras/constants.py` | `5b400698` | ✓ |
| 12 | `Lintianqianjin/LangGFM` | `Qwen/Qwen-72B-Chat-Int4` | `training/llamafactory/src/llamafactory/e` | `c0d277c2` | ✓ |
| 13 | `VectorSpaceLab/OmniGen2` | `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` | `OmniGen2-RL/evaluation/GEdit-Bench/viesc` | `3a13017e` | ✓ |
| 14 | `GiftAngel/chatbox` | `TheBloke/U-Amethyst-20B-AWQ` | `tutorial_env/Lib/site-packages/huggingfa` | `2fbfd792` | ✓ |
| 15 | `BIOIN-401-Project-8/paper-qa-chatbot` | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | `make_docs.py` | `5a4239a1` | ✓ |
| 16 | `hasanar1f/PAINT` | `TheBloke/deepseek-llm-7B-base-GPTQ` | `transformers/examples/research_projects/` | `11048d36` | ✓ |
| 17 | `weiber2002/ACA_final` | `Qwen/Qwen3-0.6B-FP8` | `cache_infer.py` | `a13e8879` | ✓ |
| 18 | `weiber2002/ACA_final` | `Qwen/Qwen3-1.7B-FP8` | `cache_infer.py` | `c84e7f58` | ✓ |
| 19 | `mastra-ai/mastra` | `deepseek-ai/DeepSeek-V3.2-Speciale` | `packages/core/src/llm/model/provider-typ` | `e259cce3` | ✓ |
| 20 | `mastra-ai/mastra` | `nvidia/Llama-3.3-70B-Instruct-FP8` | `packages/core/src/llm/model/provider-typ` | `edee4b37` | ✓ |

## H1 — Confirmatory: GGUF temporal trend

Monthly first-method-signal counts (Layer 1, `pickaxe_exact` only):

| year | GGUF | BitsAndBytes | GPTQ | AWQ | Other | total | GGUF share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 0 | 1 | 0 | 1 | 1 | 3 | 0.0% |
| 2023 | 10 | 10 | 15 | 11 | 13 | 59 | 16.9% |
| 2024 | 21 | 24 | 13 | 21 | 17 | 96 | 21.9% |
| 2025 | 33 | 37 | 17 | 34 | 34 | 155 | 21.3% |
| 2026 | 8 | 9 | 3 | 10 | 11 | 41 | 19.5% |

- GGUF total across pilot: 72
- GGUF 2025+2026 count vs 2023 count: 41 vs 10
- GGUF share late (avg of 2025+2026) vs early (2023): 20.4% vs 16.9%
- GGUF monthly OLS slope, 2023+ window: +0.063 signals/month

**ASCII monthly trajectory (GGUF first-signal counts):**

```
  2022-12    0  
  2023-04    0  
  2023-05    2  ██
  2023-07    1  █
  2023-08    2  ██
  2023-09    0  
  2023-10    0  
  2023-11    2  ██
  2023-12    3  ███
  2024-01    1  █
  2024-02    2  ██
  2024-03    1  █
  2024-05    2  ██
  2024-06    0  
  2024-07    2  ██
  2024-08    3  ███
  2024-09    2  ██
  2024-10    2  ██
  2024-11    4  ████
  2024-12    2  ██
  2025-01    2  ██
  2025-02    2  ██
  2025-03    4  ████
  2025-04    4  ████
  2025-05    3  ███
  2025-06    5  █████
  2025-07    2  ██
  2025-08    2  ██
  2025-09    2  ██
  2025-10    1  █
  2025-11    2  ██
  2025-12    4  ████
  2026-01    2  ██
  2026-02    2  ██
  2026-03    4  ████
```

**H1 verdict: CONFIRMED**

## H2 — Mechanism: BitsAndBytes trajectory

- BnB 2025-2026 count vs 2023-2024 count: 46 vs 34
- BnB share late (avg 2025+2026) vs early (avg 2023+2024): 22.9% vs 21.0%

**H2 verdict: (c) INDETERMINATE** — pattern is mixed; pilot too small to interpret cleanly.

## H3 — Supporting: adoption lag distributions

From Layer 2 `pickaxe_exact` records with valid `hf_createdAt`:

| primary_method | n | median (days) | p25 | p75 |
|---|---:|---:|---:|---:|
| GGUF | 31 | 121.2 | 40.3 | 401.8 |
| BitsAndBytes | 51 | 181.3 | 74.3 | 393.0 |
| GPTQ | 166 | 338.8 | 163.3 | 442.2 |
| AWQ | 17 | 192.5 | 62.7 | 306.7 |
| Other | 44 | 46.7 | 18.8 | 212.5 |

Spread of medians across methods: 292.1 days (visibly distinct ≈ >100 days for short-lag adoption windows).

*H3 is supporting only — does not gate the verdict.*

## Overall verdict

**(b) PROCEED WITH CAUTION** — mixed signal — full run recommended.

Gate summary:

- Clone success per stratum ≥85%: PASS
- Layer 1 mining success per stratum ≥80%: PASS
- Layer 2 pickaxe_exact ≥70% overall: PASS (92.0%)
- Spot-check ≥18/20: PASS (20/20)
- Full-run wall clock ≤24h: PASS (21.1h projected)
- H1 (GGUF temporal trend): CONFIRMED
- H2 (BnB mechanism): c
- H3 (lag distributions): supporting only


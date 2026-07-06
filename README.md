# JQuantized Model Adoption in Open-Source Software: Prevalence, Practices, and Evolution (Replication Package)

End-to-end pipeline for the empirical study on how quantized Hugging Face (HF)
models are adopted in real-world GitHub repositories, split into a **hub-side**
channel (HF-mined models × GitHub-mined repos) and an **off-hub** channel
(self-quantized / Ollama-obtained models mined from GitHub only).

This README walks through every replication step. All commands assume the
project root:

```
$ROOT = /Your/Root/Folder/Path/
```

Adjust `$ROOT` to your machine and (once) do a search-and-replace on the
absolute paths at the top of the scripts if you clone this elsewhere.

---

## 0. Prerequisites

### Python environment
```bash
# Python 3.10+
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # requests, pandas, scipy, matplotlib
```

The core statistical / regression steps use `scipy`. Everything else is stdlib.

### API tokens
Multi-token pools are used everywhere to survive GitHub / HF rate limits.
Export before running any fetch script:

```bash
# GitHub personal access tokens (need public_repo scope). Any number 1..9.
export GH_TOKEN_1=ghp_YOUR_TOKEN_HERE
export GH_TOKEN_2=ghp_YOUR_TOKEN_HERE
# ... up to GH_TOKEN_9

# Hugging Face read token (for HF-models-scraper*.py)
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

Do **not** commit tokens. `HF-models-scraper.py` line 11–12 shows the placeholder
`hf_YOUR_TOKEN_HERE`; put your real token in the environment variable, or
uncomment and paste in the header line if running interactively.

### Output layout
Everything the pipeline writes lives under `$ROOT/output_dir/`:
```
output_dir/
├── HuggingFaceStudy/modelsInfo/        # HF model metadata (step 1)
├── quantized_filtered/                 # HC quantized set (step 2)
├── github_search_results/              # Stage-1 GitHub code search (step 3)
├── final_data/                         # Tiered usage filtering (step 4-5)
└── rq_analysis/                        # All RQ analyses (steps 6-11)
    ├── rq0/  rq1/  rq2/  rq3/  shared/
```

---

## 1. Hub side — HF model discovery

**Script:** `HF-models-scraper-modified-optimized-final.py` (latest optimized
variant; earlier ones are kept for provenance).

**What it does:** paginates `GET https://huggingface.co/api/models` with
authorization, writes one JSON file per model to
`output_dir/HuggingFaceStudy/modelsInfo/<owner>£sep£<name>.json`.

```bash
python3 HF-models-scraper-modified-optimized-final.py
```

Output: ≈2.3 M per-model JSON files (~85 GB). This is the master HF corpus.

---

## 2. Filter to quantized models

**Script:** `filter_quantized_models.py`

**What it does:** applies the three-layer detector (L1 config → L2 tags → L3
model-id heuristic) with a `high_confidence` flag. Emits:

- `output_dir/quantized_filtered/quantized_models_all.jsonl` — one row per
  detected quantized model with `quant_methods`, `detection_layers`,
  `detection_signals`, `high_confidence`, `library_name`, `tags`,
  `created_at`, `last_modified`, etc.
- `output_dir/quantized_filtered/quantized_models_summary.csv` — per-method
  counts.
- `output_dir/quantized_filtered/quantized_models_all_ids.txt`,
  `quantized_models_high_confidence_ids.txt` — text ID lists.

```bash
python3 filter_quantized_models.py
```

Expected HC-set size: **271,996** models (verified downstream by
`rq1_prevalence.py`).

---

## 3. Hub side — GitHub Code Search over each model ID

**Script:** `github_code_search.py`

**What it does:** for every HC model ID, queries GitHub Code Search
(`GET /search/code?q="<model_id>"`), persists all matched code file locations,
tracks the true `total_count` returned by the API (needed for the "capped
queries" analysis), and rotates the `GH_TOKEN_*` pool to survive rate limits.

Outputs into `output_dir/github_search_results/`:

- `search_results.jsonl` — one row per queried model with schema
  `{model_id, total_count, capped, matches[], unique_repos[], num_repos, num_files}`
- `capped_models.txt` — TSV of model_id + total_count for queries where
  `total_count > 1000` (the API cap).
- `model_to_repos.json`, `repo_to_models.json` — inverted indexes.
- `unique_repos.txt` — deduped repo full_names surfaced (input for step 4).
- `search_stats.json`, `search_counters.json` — run metadata.

```bash
python3 github_code_search.py
```

**Auditable counts** you can reproduce after this step:
- 271,996 queried models → 40,764 surfaced (≥1 hit) → 90 capped (0.22 % of
  surfaced).

---

## 4. Repo metadata (RQ0)

**Script:** `output_dir/rq_analysis/rq0/scripts/collect_repo_metadata.py`

**What it does:** for every surfaced repo full_name, calls `GET /repos/{owner}/{name}`
and persists stars, forks, primary_language, created_at, updated_at, license,
topics, contributor-count, commit-count, etc.

```bash
python3 output_dir/rq_analysis/rq0/scripts/collect_repo_metadata.py
```

Output: `output_dir/rq_analysis/rq0/results/repo_metadata.csv`.

Then run the RQ0 descriptive analysis:
```bash
python3 output_dir/rq_analysis/rq0/scripts/rq0_analysis.py
```

---

## 5. Usage-tier filtering (Tier A / B / C / D → Tier A2)

**Scripts:**
- `filter_usage_repos.py` — first pass, splits surfaced repos into
  A (real usage), B (config-only), C (mention-only), D (ambiguous).
- `filter_usage_repos_with_loader_confirmation.py` — loader-confirmation
  pass on Tier A, producing Tier **A2** (loader-confirmed usage).

```bash
python3 filter_usage_repos.py
python3 filter_usage_repos_with_loader_confirmation.py
```

Outputs land in `output_dir/final_data/usage_filtered/` and
`output_dir/final_data/usage_filtered_loader/`.

Tier A2 size: **28,258** repos (loader-confirmed).

---

## 6. Main dataset build (analysis-set definition)

**Scripts:**
- `output_dir/rq_analysis/shared/scripts/build_analysis_set.py` — builds the
  "analysis set" (Tier A2 minus PEFT-only repos and other cleaning gates).
- `build_main_dataset.py` — top-level runner producing
  `main_dataset_repo_details.jsonl` + `main_dataset_repos.txt` +
  `main_dataset_definition.json`.

```bash
python3 output_dir/rq_analysis/shared/scripts/build_analysis_set.py
python3 build_main_dataset.py
```

Outputs the canonical **22,024-repo main dataset**, saved to
`output_dir/rq_analysis/shared/results/analysis_set_repo_details.jsonl` and
`output_dir/final_data/main_dataset_repo_details.jsonl`. This is the input
denominator for every hub-side RQ.

Also builds star-cohort subsets for sensitivity analyses:
`engineered_subset_stars1_repo_details.jsonl` (4,861 repos),
`engineered_subset_stars5_repo_details.jsonl` (2,831),
`engineered_subset_stars10_repo_details.jsonl` (2,246).

---

## 7. RQ1 — Prevalence of quantization methods

**Script:** `output_dir/rq_analysis/rq1/scripts/rq1_prevalence.py`

Runs the full RQ1 classification, aggregation, and figure production for a
chosen cohort. Cohort chosen via `--input`:

```bash
python3 output_dir/rq_analysis/rq1/scripts/rq1_prevalence.py --input analysis_set
python3 output_dir/rq_analysis/rq1/scripts/rq1_prevalence.py --input engineered_subset_stars1
python3 output_dir/rq_analysis/rq1/scripts/rq1_prevalence.py --input engineered_subset_stars5
python3 output_dir/rq_analysis/rq1/scripts/rq1_prevalence.py --input engineered_subset_stars10
```

**5-bucket classification** (`primary / auxiliary_only / generic_residual /
peft_lora / no_signal`) is the AUTHORITATIVE detector — see the docstring in
`rq1_prevalence.py:detect_auxiliary()`. It's asymmetric to the upstream
`filter_quantized_models.py` and this is intentional.

Auxiliary companion scripts:
```bash
python3 output_dir/rq_analysis/rq1/scripts/compute_language_distribution.py
python3 output_dir/rq_analysis/rq1/scripts/rq1_figures.py
python3 output_dir/rq_analysis/rq1/scripts/rq1_figure_b_bubble.py
```

Language distribution over `analysis_set_repo_details.jsonl` produces the
`tab:rq1_language` body:
`output_dir/rq_analysis/rq1/results/analysis_set/rq1_language_distribution.csv`.

Optional HTML-primary-repo language-byte top-up
(`fetch_html_repo_languages.py`) writes
`html_repos_language_bytes.jsonl`; requires `GH_TOKEN_*` env vars.

---

## 8. RQ2 — Method adoption & bit-width distribution

### Phase 0: repo frame + cohort matrices
```bash
python3 output_dir/rq_analysis/rq2/phase0/phase0_analysis.py
```
Produces `repo_frame.csv` (per-repo first-adoption events with stars +
cohort flags), plus year × method count/share matrices split by
`all / stars1 / stars5 / stars10` cohorts.

### Phase 1 (full): loader mining, first-signal dates, lag
```bash
python3 output_dir/rq_analysis/rq2/phase1_full/mine_full.py
```
Clones repos in parallel, runs Layer-1 method-signal detection and
Layer-2 model×repo pickaxe / head-fallback mining. Emits
`layer1_method_signals.jsonl`, `layer2_model_signals.jsonl` (per-event lag),
`method_adoption_summary.csv`, `rq2_lag_table_updated.csv`,
and diagnostic figures.

### Bit-width distribution (hub-side, adopted 13,752 models)
```bash
python3 output_dir/rq_analysis/rq2/scripts/compute_hub_bitwidth_distribution.py
```
Reproduces `tab:rq2_bitwidth`:
`output_dir/rq_analysis/rq2/results/hub_bitwidth_distribution.csv` +
`hub_bitwidth_per_model.csv`.

---

## 9. RQ3 — Temporal analysis (this repo's sensitivity extensions)

**Hub-side scripts (added in this replication package):**
```bash
python3 output_dir/rq_analysis/rq3/scripts/rq3_1_method_over_time_sensitivity.py
python3 output_dir/rq_analysis/rq3/scripts/rq3_2_adoption_lag_sensitivity.py
python3 output_dir/rq_analysis/rq3/scripts/rq3_new_quantized_per_year.py
```

- **RQ3.1** (method × year × cohort) uses `rq2/phase0/repo_frame.csv`; produces
  per-cohort count + share CSVs + trend summary.
- **RQ3.2** (adoption lag × cohort) joins Layer-2 per-event lag to the
  cohort flags; produces per-cohort median-lag CSVs + stability summary.
- **rq3_new_quantized_per_year** counts new HC models per year (2021–2025)
  from `quantized_models_all.jsonl` and fits `count ~ year_index`.

Results in `output_dir/rq_analysis/rq3/results/`.

---

## 10. Off-hub side — Ollama & self-quantized mining

Directory: `ollama-selfquant-mining/`. Independent pipeline that mines GitHub
Code Search for `ollama create/pull` invocations and self-quantization calls.

```bash
# Step 10a: discovery
python3 ollama-selfquant-mining/ollama_selfquant_miner.py
# → ollama_selfquant_repos.jsonl (matched repos)

# Step 10b: repo-details metadata top-up
python3 ollama-selfquant-mining/metadata_topup/fetch_offhub_metadata_topup.py
# Optional topics top-up (single GET /repos per repo, resumable):
python3 ollama-selfquant-mining/metadata_topup/fetch_offhub_topics.py

# Step 10c: temporal mining (first-adoption dates)
python3 ollama-selfquant-mining/temporal_full/mine_offhub_temporal.py
python3 ollama-selfquant-mining/temporal_full/build_offhub_rq3_stats.py

# Step 10d: final-cut analysis (bit-width, tool ranking, precision sensitivity)
python3 ollama-selfquant-mining/analyze_offhub_finalcut.py
```

Outputs land under `ollama-selfquant-mining/` and its
`analysis_offhub/`, `temporal_full/`, `metadata_topup/` subdirs. Notably:
- `analysis_offhub/precision_overall_vs_llmstack.csv` — the numbers behind
  the "8-bit 64.67 % overall / 4-bit 47.10 % LLM-serving stack" table.
- `analysis_offhub/precision_sensitivity.csv` — STRICT vs EXPANDED scoping.
- `temporal_full/offhub_rq3_method_share_by_year.csv` — off-hub RQ3.

**Note on precision counting.** Off-hub precision buckets are computed
**per-repo** on a `set` of bit-width labels — a repo publishing multiple
bit-width variants (e.g. Q4_K_M + Q5_K_M + Q8_0 Modelfiles) contributes
`+1` to each bucket. Columns can and do sum above 100 %. See
`analyze_offhub_finalcut.py:178–194` and `398–413`.

---

## 11. Manual validation (annotator agreement + precision)

Sample construction:
```bash
python3 output_dir/rq_analysis/shared/validation/build_precision_sample.py
```
Draws 400 stratified samples (100 per taxonomy stratum) → annotator-facing
CSV, kappa join key, controlled-vocabulary cheat sheet.

After annotation (independent A1 / A2 columns saved into
`validation_precision_sample_annotated.csv`, then adjudicated to
`_final` columns in `validation_precision_sample_adjudicated.csv`):

```bash
python3 output_dir/rq_analysis/shared/validation/compute_validation_metrics.py       # IRA + as-is precision
python3 output_dir/rq_analysis/shared/validation/compute_validation_metrics_final.py # final adjudicated precision
```

Results in `output_dir/rq_analysis/shared/validation/results/`:
`validation_results_final.json`, `precision_by_stratum_final.csv`,
`false_positives_final.csv`, `validation_summary_final.md`, plus the
IRA-only `agreement_matrices.txt`, `disagreements.csv`.

---

## 12. PEFT-boundary sensitivity

Directory: `output_dir/rq_analysis/shared/peft_boundary_rerun/`. Rebuilds the
main dataset with a different PEFT-exclusion boundary and compares.

```bash
python3 output_dir/rq_analysis/shared/peft_boundary_rerun/scripts/build_main_dataset_boundary.py
python3 output_dir/rq_analysis/shared/peft_boundary_rerun/scripts/compare_peft_only_sets.py
```

---

## Reproducing the paper's headline numbers

| Table / Number | Script | Output file |
|---|---|---|
| 271,996 HC quantized models | `filter_quantized_models.py` | `quantized_models_all.jsonl` |
| 22,024 main-dataset repos | `build_main_dataset.py` | `main_dataset_repo_details.jsonl` |
| tab:rq1_language | `compute_language_distribution.py` | `rq1_language_distribution.csv` |
| tab:rq1_prevalence (Q1a/b) | `rq1_prevalence.py --input analysis_set` | `rq1/results/analysis_set/q1_*.csv` |
| taxonomy_complete_fixed.csv (62 labels) | `build_taxonomy_inventory.py` | `shared/results/taxonomy_complete_fixed.csv` |
| tab:rq2_bitwidth | `compute_hub_bitwidth_distribution.py` | `rq2/results/hub_bitwidth_distribution.csv` |
| RQ3.1 sensitivity | `rq3_1_method_over_time_sensitivity.py` | `rq3/results/rq3_1_counts_*.csv` |
| RQ3.2 sensitivity | `rq3_2_adoption_lag_sensitivity.py` | `rq3/results/rq3_2_lag_*.csv` |
| off-hub precision | `analyze_offhub_finalcut.py` | `analysis_offhub/precision_overall_vs_llmstack.csv` |
| validation precision (final) | `compute_validation_metrics_final.py` | `validation/results/validation_summary_final.md` |

---

## Notes on rerunning from scratch

1. **API budget.** Step 1 (HF scrape) is ~2.3 M requests. Step 3 (GitHub Code
   Search) is ~272 k queries with cap. Step 4/5 (repo metadata) is up to
   ~40 k `GET /repos` calls. Provide 5–9 tokens and expect ~24 hours end to
   end.
2. **Deterministic parts.** Every script downstream of step 3 is deterministic
   given the mined data and `SEED=42` (used by validation-sample construction
   and any random-draw step). The pipeline is idempotent — rerunning a stage
   overwrites its outputs.
3. **Tokens sanitized.** All previously-committed HF and GitHub tokens have
   been replaced with `hf_YOUR_TOKEN_HERE` and `ghp_YOUR_TOKEN_HERE`
   placeholders. Supply your own via env vars before running any fetch stage.

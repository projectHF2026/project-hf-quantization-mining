# RQ2 Phase 0 - Repository Cohort Analysis

> **Scope note:** Phase 0 evaluates repository cohorts rather
> than adoption events. Observed differences motivate, but do
> not establish, temporal adoption dynamics; these require
> git-history mining in subsequent phases. Creation year is
> used here as a proxy for repository vintage, not adoption
> time. Any phrasing in this document should describe what
> cohorts of repositories use, not when methods were adopted.

## Inputs used

- `/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/rq_analysis/shared/results/analysis_set_repos.txt` (578,468 bytes) — analysis_set_repos.txt
- `/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/rq_analysis/shared/results/analysis_set_repo_details.jsonl` (39,170,464 bytes) — analysis_set_repo_details.jsonl
- `/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/quantized_filtered/quantized_models_all.jsonl` (214,687,235 bytes) — quantized_models_all.jsonl

## Coverage summary

- Total repos in analysis set: **22,024**
- Excluded from Phase 0: **1,320** (1,320 no primary method, 0 no created_at, 0 year out of range)
- Frame rows: **20,704**

Distribution of `created_at_year`:

```
  2009         1  █
  2010         1  █
  2012         2  █
  2013         7  █
  2014        13  █
  2015        22  █
  2016        24  █
  2017        28  █
  2018        38  █
  2019        54  █
  2020       100  █
  2021       129  █
  2022       245  █
  2023     2,133  ██████
  2024     5,947  ███████████████████
  2025     9,246  ██████████████████████████████
  2026     2,714  ████████
```

Distribution of `method_bucket`:

```
  AWQ              1,555  █████
  BitsAndBytes     3,874  ████████████
  GGUF             8,971  ██████████████████████████████
  GPTQ             2,679  ████████
  Other            3,625  ████████████
```

Stars-threshold subset counts (compared to RQ1 Table 6 expected values):

| subset | observed | expected (Table 6) |
|---|---:|---:|
| is_stars1 | 4,536 | 4,861 |
| is_stars5 | 2,636 | 2,831 |
| is_stars10 | 2,091 | 2,246 |

## Matrix: counts, all repos

| year | GGUF | BitsAndBytes | GPTQ | AWQ | Other | Total |
|---|---|---|---|---|---|---|
| 2009 | 0 | 0 | 0 | 0 | 1 | 1 |
| 2010 | 1 | 0 | 0 | 0 | 0 | 1 |
| 2012 | 1 | 0 | 0 | 0 | 1 | 2 |
| 2013 | 4 | 0 | 0 | 0 | 3 | 7 |
| 2014 | 7 | 4 | 0 | 1 | 1 | 13 |
| 2015 | 13 | 6 | 0 | 1 | 2 | 22 |
| 2016 | 19 | 4 | 0 | 0 | 1 | 24 |
| 2017 | 13 | 5 | 4 | 4 | 2 | 28 |
| 2018 | 21 | 8 | 3 | 1 | 5 | 38 |
| 2019 | 26 | 18 | 1 | 3 | 6 | 54 |
| 2020 | 41 | 32 | 7 | 4 | 16 | 100 |
| 2021 | 60 | 26 | 21 | 8 | 14 | 129 |
| 2022 | 91 | 54 | 15 | 9 | 76 | 245 |
| 2023 | 674 | 256 | 577 | 162 | 464 | 2,133 |
| 2024 | 2,371 | 1,244 | 937 | 534 | 861 | 5,947 |
| 2025 | 4,049 | 1,886 | 971 | 682 | 1,658 | 9,246 |
| 2026 | 1,580 | 331 | 143 | 146 | 514 | 2,714 |

## Matrix: within-cohort shares, all repos

| year | GGUF | BitsAndBytes | GPTQ | AWQ | Other | Total |
|---|---|---|---|---|---|---|
| 2009 | 0 | 0 | 0 | 0 | 1 | 1 |
| 2010 | 1 | 0 | 0 | 0 | 0 | 1 |
| 2012 | 0.50 | 0 | 0 | 0 | 0.50 | 1 |
| 2013 | 0.57 | 0 | 0 | 0 | 0.43 | 1 |
| 2014 | 0.54 | 0.31 | 0 | 0.08 | 0.08 | 1 |
| 2015 | 0.59 | 0.27 | 0 | 0.05 | 0.09 | 1 |
| 2016 | 0.79 | 0.17 | 0 | 0 | 0.04 | 1 |
| 2017 | 0.46 | 0.18 | 0.14 | 0.14 | 0.07 | 1 |
| 2018 | 0.55 | 0.21 | 0.08 | 0.03 | 0.13 | 1 |
| 2019 | 0.48 | 0.33 | 0.02 | 0.06 | 0.11 | 1 |
| 2020 | 0.41 | 0.32 | 0.07 | 0.04 | 0.16 | 1 |
| 2021 | 0.47 | 0.20 | 0.16 | 0.06 | 0.11 | 1 |
| 2022 | 0.37 | 0.22 | 0.06 | 0.04 | 0.31 | 1 |
| 2023 | 0.32 | 0.12 | 0.27 | 0.08 | 0.22 | 1 |
| 2024 | 0.40 | 0.21 | 0.16 | 0.09 | 0.14 | 1 |
| 2025 | 0.44 | 0.20 | 0.10 | 0.07 | 0.18 | 1 |
| 2026 | 0.58 | 0.12 | 0.05 | 0.05 | 0.19 | 1 |

## Matrix: within-method shares, all repos

| year | GGUF | BitsAndBytes | GPTQ | AWQ | Other |
|---|---|---|---|---|---|
| 2009 | 0 | 0 | 0 | 0 | 0.00 |
| 2010 | 0.00 | 0 | 0 | 0 | 0 |
| 2012 | 0.00 | 0 | 0 | 0 | 0.00 |
| 2013 | 0.00 | 0 | 0 | 0 | 0.00 |
| 2014 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 2015 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 2016 | 0.00 | 0.00 | 0 | 0 | 0.00 |
| 2017 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2018 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2019 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2020 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| 2021 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| 2022 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| 2023 | 0.08 | 0.07 | 0.22 | 0.10 | 0.13 |
| 2024 | 0.26 | 0.32 | 0.35 | 0.34 | 0.24 |
| 2025 | 0.45 | 0.49 | 0.36 | 0.44 | 0.46 |
| 2026 | 0.18 | 0.09 | 0.05 | 0.09 | 0.14 |
| Total | 1 | 1 | 1 | 1 | 1 |

## Matrix: within-cohort shares, stars>=1 subset

| year | GGUF | BitsAndBytes | GPTQ | AWQ | Other | Total |
|---|---|---|---|---|---|---|
| 2009 | 0 | 0 | 0 | 0 | 1 | 1 |
| 2010 | 1 | 0 | 0 | 0 | 0 | 1 |
| 2012 | 1 | 0 | 0 | 0 | 0 | 1 |
| 2013 | 0.40 | 0 | 0 | 0 | 0.60 | 1 |
| 2014 | 0.67 | 0.11 | 0 | 0.11 | 0.11 | 1 |
| 2015 | 0.92 | 0 | 0 | 0.08 | 0 | 1 |
| 2016 | 0.89 | 0.06 | 0 | 0 | 0.06 | 1 |
| 2017 | 0.57 | 0.07 | 0.14 | 0.21 | 0 | 1 |
| 2018 | 0.65 | 0.09 | 0.09 | 0 | 0.17 | 1 |
| 2019 | 0.58 | 0.12 | 0 | 0.12 | 0.17 | 1 |
| 2020 | 0.47 | 0.12 | 0.12 | 0.05 | 0.26 | 1 |
| 2021 | 0.54 | 0.07 | 0.12 | 0.09 | 0.18 | 1 |
| 2022 | 0.44 | 0.07 | 0.10 | 0.06 | 0.34 | 1 |
| 2023 | 0.41 | 0.08 | 0.20 | 0.08 | 0.23 | 1 |
| 2024 | 0.40 | 0.17 | 0.16 | 0.12 | 0.15 | 1 |
| 2025 | 0.41 | 0.14 | 0.13 | 0.09 | 0.22 | 1 |
| 2026 | 0.64 | 0.07 | 0.04 | 0.06 | 0.19 | 1 |

## Matrix: within-cohort shares, stars>=5 subset

| year | GGUF | BitsAndBytes | GPTQ | AWQ | Other | Total |
|---|---|---|---|---|---|---|
| 2009 | 0 | 0 | 0 | 0 | 1 | 1 |
| 2010 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2012 | 1 | 0 | 0 | 0 | 0 | 1 |
| 2013 | 0.50 | 0 | 0 | 0 | 0.50 | 1 |
| 2014 | 0.40 | 0.20 | 0 | 0.20 | 0.20 | 1 |
| 2015 | 0.89 | 0 | 0 | 0.11 | 0 | 1 |
| 2016 | 0.83 | 0.08 | 0 | 0 | 0.08 | 1 |
| 2017 | 0.55 | 0.09 | 0.18 | 0.18 | 0 | 1 |
| 2018 | 0.65 | 0.10 | 0.10 | 0 | 0.15 | 1 |
| 2019 | 0.57 | 0.10 | 0 | 0.14 | 0.19 | 1 |
| 2020 | 0.47 | 0.08 | 0.14 | 0.03 | 0.28 | 1 |
| 2021 | 0.51 | 0.07 | 0.17 | 0.10 | 0.15 | 1 |
| 2022 | 0.43 | 0.09 | 0.10 | 0.09 | 0.30 | 1 |
| 2023 | 0.43 | 0.09 | 0.15 | 0.09 | 0.23 | 1 |
| 2024 | 0.40 | 0.13 | 0.15 | 0.15 | 0.18 | 1 |
| 2025 | 0.36 | 0.12 | 0.17 | 0.11 | 0.24 | 1 |
| 2026 | 0.61 | 0.05 | 0.07 | 0.07 | 0.20 | 1 |

## Matrix: within-cohort shares, stars>=10 subset

| year | GGUF | BitsAndBytes | GPTQ | AWQ | Other | Total |
|---|---|---|---|---|---|---|
| 2009 | 0 | 0 | 0 | 0 | 1 | 1 |
| 2010 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2012 | 1 | 0 | 0 | 0 | 0 | 1 |
| 2013 | 0.50 | 0 | 0 | 0 | 0.50 | 1 |
| 2014 | 0.40 | 0.20 | 0 | 0.20 | 0.20 | 1 |
| 2015 | 0.88 | 0 | 0 | 0.12 | 0 | 1 |
| 2016 | 0.83 | 0.08 | 0 | 0 | 0.08 | 1 |
| 2017 | 0.44 | 0.11 | 0.22 | 0.22 | 0 | 1 |
| 2018 | 0.63 | 0.11 | 0.11 | 0 | 0.16 | 1 |
| 2019 | 0.60 | 0 | 0 | 0.13 | 0.27 | 1 |
| 2020 | 0.47 | 0.08 | 0.14 | 0.03 | 0.28 | 1 |
| 2021 | 0.48 | 0.06 | 0.15 | 0.12 | 0.18 | 1 |
| 2022 | 0.45 | 0.05 | 0.12 | 0.07 | 0.31 | 1 |
| 2023 | 0.44 | 0.08 | 0.15 | 0.08 | 0.24 | 1 |
| 2024 | 0.40 | 0.12 | 0.14 | 0.14 | 0.20 | 1 |
| 2025 | 0.33 | 0.11 | 0.19 | 0.11 | 0.26 | 1 |
| 2026 | 0.55 | 0.05 | 0.09 | 0.09 | 0.23 | 1 |

## Robustness across stars thresholds

For each named method, we compare its within-cohort share trajectory across the full set and the three engineered subsets. A trajectory is the sequence of within-cohort shares for qualifying years (cohort size ≥ 50 repos). The signal is **robust** if the share moves in the same direction (sign of Δ) across all four views.

| method | view | earliest year | earliest share | latest year | latest share | Δ pp |
|---|---|---:|---:|---:|---:|---:|
| GGUF | all | 2019 | 48.1% | 2026 | 58.2% | +10.1 |
| GGUF | stars≥1 | 2021 | 53.6% | 2026 | 64.0% | +10.5 |
| GGUF | stars≥5 | 2022 | 42.9% | 2026 | 60.8% | +18.0 |
| GGUF | stars≥10 | 2022 | 44.8% | 2026 | 54.5% | +9.7 |
| BitsAndBytes | all | 2019 | 33.3% | 2026 | 12.2% | -21.1 |
| BitsAndBytes | stars≥1 | 2021 | 7.1% | 2026 | 6.9% | -0.2 |
| BitsAndBytes | stars≥5 | 2022 | 8.6% | 2026 | 4.6% | -4.0 |
| BitsAndBytes | stars≥10 | 2022 | 5.2% | 2026 | 4.5% | -0.6 |
| GPTQ | all | 2019 | 1.9% | 2026 | 5.3% | +3.4 |
| GPTQ | stars≥1 | 2021 | 12.5% | 2026 | 4.3% | -8.2 |
| GPTQ | stars≥5 | 2022 | 10.0% | 2026 | 7.1% | -2.9 |
| GPTQ | stars≥10 | 2022 | 12.1% | 2026 | 8.5% | -3.5 |
| AWQ | all | 2019 | 5.6% | 2026 | 5.4% | -0.2 |
| AWQ | stars≥1 | 2021 | 8.9% | 2026 | 5.9% | -3.0 |
| AWQ | stars≥5 | 2022 | 8.6% | 2026 | 7.1% | -1.5 |
| AWQ | stars≥10 | 2022 | 6.9% | 2026 | 9.1% | +2.2 |

Per-method robustness verdict (signs of Δ across four views):

- **GGUF**: signs = + + + + → stable across all thresholds
- **BitsAndBytes**: signs = − − − − → stable across all thresholds
- **GPTQ**: signs = + − − − → direction changes across thresholds
- **AWQ**: signs = − − − + → direction changes across thresholds

*Reminder: stability of direction is a positive signal; direction changes across thresholds indicate the cohort-displacement pattern is concentrated in one quality tier — note this in Phase 1 design.*

## Observations

All observations below describe what cohorts of repositories use, not when methods were adopted.

- **GGUF**: within-cohort shares across qualifying years: 2019=48.1%, 2020=41.0%, 2021=46.5%, 2022=37.1%, 2023=31.6%, 2024=39.9%, 2025=43.8%, 2026=58.2%. Peak cohort year = **2026** (58.2%). Δ between earliest (2019) and latest (2026) qualifying cohorts: **+10.1 pp**.
  - GGUF's within-cohort share **crosses 50%** in cohort year(s): 2026.
  - Peak-to-latest Δ for GGUF: +0.0 pp (from 2026 to 2026).
- **BitsAndBytes**: within-cohort shares across qualifying years: 2019=33.3%, 2020=32.0%, 2021=20.2%, 2022=22.0%, 2023=12.0%, 2024=20.9%, 2025=20.4%, 2026=12.2%. Peak cohort year = **2019** (33.3%). Δ between earliest (2019) and latest (2026) qualifying cohorts: **-21.1 pp**.
  - BitsAndBytes's share **drops by >50%** from peak (33.3%) to latest cohort (12.2%).
  - Peak-to-latest Δ for BitsAndBytes: -21.1 pp (from 2019 to 2026).
- **GPTQ**: within-cohort shares across qualifying years: 2019=1.9%, 2020=7.0%, 2021=16.3%, 2022=6.1%, 2023=27.1%, 2024=15.8%, 2025=10.5%, 2026=5.3%. Peak cohort year = **2023** (27.1%). Δ between earliest (2019) and latest (2026) qualifying cohorts: **+3.4 pp**.
  - GPTQ's share **drops by >50%** from peak (27.1%) to latest cohort (5.3%).
  - Peak-to-latest Δ for GPTQ: -21.8 pp (from 2023 to 2026).
- **AWQ**: within-cohort shares across qualifying years: 2019=5.6%, 2020=4.0%, 2021=6.2%, 2022=3.7%, 2023=7.6%, 2024=9.0%, 2025=7.4%, 2026=5.4%. Peak cohort year = **2024** (9.0%). Δ between earliest (2019) and latest (2026) qualifying cohorts: **-0.2 pp**.
  - Peak-to-latest Δ for AWQ: -3.6 pp (from 2024 to 2026).

## Verdict on RQ2 mining

**Verdict: (a) STRONG COHORT SIGNAL**

Candidate finding sentence (Phase 0 alone supports this; Phase 1 mining is required to convert it to an adoption claim):

> Repositories created in 2026 are less likely to use BitsAndBytes as their primary quantization method than repositories created in 2019 (33.3% → 12.2% within-cohort share, Δ = -21.1 pp).

The signal direction is robust: the same sign of cohort displacement for BitsAndBytes holds across all three stars-threshold subsets (stars≥1, stars≥5, stars≥10).



"""
RQ2 Phase 0 — Repository Cohort Analysis (no cloning, no git mining).

Stage 1: build per-repo frame with dominant primary method + creation-year cohort.
Stage 2: build 12 matrices (3 views × 4 subsets: all / stars1 / stars5 / stars10).
Stage 3: write phase0_diagnostics.md.
Stage 4: write optional 100% stacked-bar PNG (only if matplotlib available).

All percentages frame REPOSITORY COHORTS, not adoption events. Creation year is
a vintage proxy, NOT an adoption-time proxy. The verdict gates whether the
git-mining pilot should proceed.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

BASE_DIR = Path("/scratch/oldhome/user/projects/JAW/scripts/icpc-approch")
ANALYSIS_REPOS_TXT = (
    BASE_DIR / "output_dir/rq_analysis/shared/results/analysis_set_repos.txt"
)
ANALYSIS_JSONL = (
    BASE_DIR / "output_dir/rq_analysis/shared/results/analysis_set_repo_details.jsonl"
)
ALL_MODELS_JSONL = BASE_DIR / "output_dir/quantized_filtered/quantized_models_all.jsonl"

OUT_DIR = BASE_DIR / "output_dir/rq_analysis/rq2/phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_CSV = OUT_DIR / "repo_frame.csv"
DIAGNOSTICS_MD = OUT_DIR / "phase0_diagnostics.md"
PLOT_PNG = OUT_DIR / "phase0_cohort_stacked_bars.png"

METHOD_ORDER = ["GGUF", "BitsAndBytes", "GPTQ", "AWQ", "Other"]
NAMED_METHODS = ["GGUF", "BitsAndBytes", "GPTQ", "AWQ"]
COHORT_MIN_REPOS = 50  # year cohorts smaller than this are flagged as sparse

# Expected sensitivity-analysis counts (from RQ1 Table 6 / shared/results/
# analysis_set_definition.json) for sanity-check.
EXPECTED_STARS_COUNTS = {
    "is_stars1":  4861,
    "is_stars5":  2831,
    "is_stars10": 2246,
}

# ---------------------------------------------------------------------------
# Method taxonomy (mirrored from rq1_prevalence.py / sampling_script.py)
# ---------------------------------------------------------------------------

NON_METHOD_LABELS = {
    "quantized_config_file", "unknown_quantized", "candidate_only", "Quantized_generic",
}
BIT_WIDTH_LABELS = {"2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"}
BNB_VARIANTS = {
    "BitsAndBytes", "BitsAndBytes_4bit", "BitsAndBytes_8bit",
    "bnb_nf4", "bnb.nf4", "bnb4bit", "bnb8bit",
}
AUXILIARY_FROM_QUANT_METHODS = {
    "coreml_quantized": "CoreML",
    "TensorRT_quantized": "TensorRT",
    "onnx_quantized": "ONNX_quantized",
    "OpenVINO_quantized": "OpenVINO",
}
CONFIG_METHOD_TO_PRIMARY = {
    "mxfp4": "MXFP4", "mxfp8": "MXFP8",
    "auto-round": "AutoRound", "intel/auto-round": "AutoRound", "auto_round": "AutoRound",
    "modelopt": "NVFP4/ModelOpt", "modelopt_fp4": "NVFP4/ModelOpt", "nvfp4": "NVFP4/ModelOpt",
    "bitnet": "BitNet", "quark": "Quark", "higgs": "HIGGS",
}
ID_SUBSTRINGS_TO_PRIMARY = [
    ("mxfp4", "MXFP4"), ("mxfp8", "MXFP8"), ("nvfp4", "NVFP4/ModelOpt"),
    ("autoround", "AutoRound"), ("auto-round", "AutoRound"), ("auto_round", "AutoRound"),
    ("gptq", "GPTQ"), ("awq", "AWQ"), ("ggml", "GGML"), ("hqq", "HQQ"),
]
MLX_LIBS = {"mlx", "mlx-audio", "mflux", "mlx-vlm", "mlx-audio-plus", "mlx-lm"}
PEFT_ID_KEYWORDS = ["qlora", "lora", "dora", "adapter"]
QUANT_METHOD_RE = re.compile(r"config\.quantization_config\.quant_method=([^\s\(]+)")


def primary_methods_of(rec: dict) -> set[str]:
    mid_lc = rec["model_id"].lower()
    library_name = (rec.get("library_name") or "").strip()
    tags_lc = [t.lower() for t in (rec.get("tags") or [])]
    raw_qm = rec.get("quant_methods") or []
    signals = rec.get("detection_signals") or []

    if library_name == "peft" or any(kw in mid_lc for kw in PEFT_ID_KEYWORDS):
        return set()

    cleaned = {m for m in raw_qm if m not in NON_METHOD_LABELS}
    cleaned = {m for m in cleaned if m not in AUXILIARY_FROM_QUANT_METHODS}
    if cleaned & BNB_VARIANTS:
        cleaned -= BNB_VARIANTS
        cleaned.add("BitsAndBytes")

    bit_widths = cleaned & BIT_WIDTH_LABELS
    specific = cleaned - BIT_WIDTH_LABELS
    primary = set(specific)

    for s in signals:
        m = QUANT_METHOD_RE.search(s)
        if not m:
            continue
        cm = m.group(1).strip().rstrip(",").lower()
        if cm in CONFIG_METHOD_TO_PRIMARY:
            primary.add(CONFIG_METHOD_TO_PRIMARY[cm])

    if not primary:
        for sub, label in ID_SUBSTRINGS_TO_PRIMARY:
            if sub in mid_lc:
                primary.add(label)
                break

    has_aux = (
        library_name in MLX_LIBS or "mlx" in mid_lc or any("mlx" in t for t in tags_lc)
        or library_name == "mlc-llm" or "mlc-llm" in mid_lc or any("mlc-llm" in t for t in tags_lc)
        or library_name == "vllm" or "vllm" in mid_lc or any("vllm" in t for t in tags_lc)
        or library_name == "coreml" or "coreml" in mid_lc or any("coreml" in t for t in tags_lc)
        or "tensorrt" in mid_lc or any("tensorrt" in t for t in tags_lc)
        or library_name == "onnx" or "onnx" in mid_lc or any("onnx" in t for t in tags_lc)
        or library_name == "openvino" or "openvino" in mid_lc or any("openvino" in t for t in tags_lc)
    )
    if not primary and has_aux:
        return set()

    if not primary and bit_widths:
        primary = {f"Generic {b.replace('bit', '-bit')}" for b in bit_widths}
    return primary


def dominant_method(counts: Counter) -> str | None:
    if not counts:
        return None
    max_c = max(counts.values())
    top = sorted(m for m, c in counts.items() if c == max_c)
    return top[0]


def bucket_for(dominant: str) -> str:
    if dominant in ("GGUF", "BitsAndBytes", "GPTQ", "AWQ"):
        return dominant
    return "Other"


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------


def df_to_md(df: pd.DataFrame, float_fmt: str | None = "{:.2f}",
             int_fmt: str = "{:,}", index_label: str = "") -> str:
    """Render a DataFrame as a markdown table."""
    cols = [index_label] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer()):
                cells.append(int_fmt.format(int(v)))
            elif float_fmt is not None:
                cells.append(float_fmt.format(float(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def text_histogram(counts: dict, label_width: int = 6) -> str:
    if not counts:
        return "(empty)"
    max_c = max(counts.values())
    width = 30
    lines = []
    for k in sorted(counts):
        n = counts[k]
        bar = "█" * max(1, int(width * n / max_c)) if n > 0 else ""
        lines.append(f"  {str(k):<{label_width}} {n:>7,}  {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------


def build_matrices(df_sub: pd.DataFrame, year_order: list[int]) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Return (counts, cohort_share, method_share) with years × method_bucket."""
    counts = pd.crosstab(df_sub["created_at_year"], df_sub["method_bucket"])
    counts = counts.reindex(columns=METHOD_ORDER, fill_value=0)
    counts = counts.reindex(index=year_order, fill_value=0)
    counts.index.name = "year"
    counts.columns.name = None

    row_sums = counts.sum(axis=1)
    col_sums = counts.sum(axis=0)
    cohort_share = counts.div(row_sums.replace(0, pd.NA), axis=0).fillna(0.0).astype(float)
    method_share = counts.div(col_sums.replace(0, pd.NA), axis=1).fillna(0.0).astype(float)
    return counts, cohort_share, method_share


def save_matrices(counts: pd.DataFrame, cohort_share: pd.DataFrame,
                  method_share: pd.DataFrame, suffix: str) -> None:
    counts.to_csv(OUT_DIR / f"matrix_counts{suffix}.csv")
    cohort_share.round(4).to_csv(OUT_DIR / f"matrix_cohort_share{suffix}.csv")
    method_share.round(4).to_csv(OUT_DIR / f"matrix_method_share{suffix}.csv")


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def cohort_share_trajectory(
    cohort_share: pd.DataFrame,
    counts: pd.DataFrame,
    method: str,
    min_cohort_size: int = COHORT_MIN_REPOS,
) -> dict:
    """Trajectory of within-cohort share for one method across qualifying years."""
    qualifying = counts.sum(axis=1) >= min_cohort_size
    years = [y for y in cohort_share.index if qualifying.get(y, False)]
    if not years:
        return {"years": [], "shares": [], "earliest_share": None,
                "latest_share": None, "delta_pp": None, "peak_year": None,
                "peak_share": None}
    shares = [float(cohort_share.loc[y, method]) for y in years]
    earliest = shares[0]
    latest = shares[-1]
    peak_idx = max(range(len(shares)), key=lambda i: shares[i])
    return {
        "years": years,
        "shares": shares,
        "earliest_year": years[0],
        "latest_year": years[-1],
        "earliest_share": earliest,
        "latest_share": latest,
        "delta_pp": (latest - earliest) * 100,
        "peak_year": years[peak_idx],
        "peak_share": shares[peak_idx],
    }


def determine_verdict(
    all_summary: dict,
    s1_summary: dict,
    s5_summary: dict,
    s10_summary: dict,
) -> tuple[str, str, str]:
    """Return (letter, finding_sentence, robustness_statement)."""
    deltas_full = {
        m: abs(all_summary[m]["delta_pp"]) for m in NAMED_METHODS
        if all_summary[m].get("delta_pp") is not None
    }
    if not deltas_full:
        return ("c",
                "No named-method cohort years had sufficient repos for a valid delta.",
                "")
    max_delta = max(deltas_full.values())
    max_method = max(deltas_full, key=deltas_full.get)
    full_traj = all_summary[max_method]

    # Direction (rising vs falling) of largest-delta method in the full set
    signed_delta = full_summary_delta = full_traj["delta_pp"]

    # Robustness check: does each subset show the same DIRECTION for max_method?
    same_direction_subsets = []
    for label, summary in [("stars≥1", s1_summary), ("stars≥5", s5_summary),
                           ("stars≥10", s10_summary)]:
        d = summary.get(max_method, {}).get("delta_pp")
        if d is None:
            continue
        if (d >= 0) == (signed_delta >= 0):
            same_direction_subsets.append(label)
    robust = len(same_direction_subsets) == 3

    # Verdict thresholds (per spec)
    if max_delta >= 20:
        verdict = "a"
        direction = "more likely" if signed_delta > 0 else "less likely"
        finding = (
            f"Repositories created in {full_traj['latest_year']} are "
            f"{direction} to use {max_method} as their primary quantization "
            f"method than repositories created in {full_traj['earliest_year']} "
            f"({full_traj['earliest_share']*100:.1f}% → "
            f"{full_traj['latest_share']*100:.1f}% within-cohort share, "
            f"Δ = {signed_delta:+.1f} pp)."
        )
        if robust:
            robustness = (
                f"The signal direction is robust: the same sign of cohort "
                f"displacement for {max_method} holds across all three "
                f"stars-threshold subsets (stars≥1, stars≥5, stars≥10)."
            )
        else:
            robustness = (
                f"The signal direction is NOT robust across all stars-threshold "
                f"subsets — same direction confirmed only in: "
                f"{', '.join(same_direction_subsets) if same_direction_subsets else 'none'}. "
                f"Worth flagging in Phase 1 design."
            )
    elif max_delta >= 10:
        verdict = "b"
        finding = ""
        robustness = ""
    else:
        verdict = "c"
        finding = ""
        robustness = ""
    return verdict, finding, robustness


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log_lines: list[str] = []

    def logp(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    logp("=== RQ2 Phase 0 — Repository Cohort Analysis ===")
    logp()
    logp(f"Inputs:")
    logp(f"  analysis_set_repos.txt        — {ANALYSIS_REPOS_TXT}  "
         f"({ANALYSIS_REPOS_TXT.stat().st_size:,} bytes)")
    logp(f"  analysis_set_repo_details.jsonl — {ANALYSIS_JSONL}  "
         f"({ANALYSIS_JSONL.stat().st_size:,} bytes)")
    logp(f"  quantized_models_all.jsonl    — {ALL_MODELS_JSONL}  "
         f"({ALL_MODELS_JSONL.stat().st_size:,} bytes)")
    logp()

    # ----------------------------------------------------------
    # Stage 1: classify models, then build per-repo frame
    # ----------------------------------------------------------
    logp(f"Streaming {ALL_MODELS_JSONL.name} ...")
    model_primary: dict[str, set[str]] = {}
    with ALL_MODELS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            primary = primary_methods_of(rec)
            if primary:
                model_primary[rec["model_id"]] = primary
    logp(f"  {len(model_primary):,} models have ≥1 primary method")

    logp(f"Walking {ANALYSIS_JSONL.name} ...")
    rows: list[dict] = []
    n_total = 0
    n_excluded_no_primary = 0
    n_excluded_no_date = 0
    n_excluded_bad_year = 0
    bad_year_examples: list[str] = []
    with ANALYSIS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1
            repo = rec["repo"]

            counts: Counter[str] = Counter()
            for mid in rec.get("models") or []:
                for m in model_primary.get(mid, set()):
                    counts[m] += 1

            dom = dominant_method(counts)
            if dom is None:
                n_excluded_no_primary += 1
                continue

            created_at = rec.get("created_at", "")
            if not created_at:
                n_excluded_no_date += 1
                continue
            try:
                year = int(created_at[:4])
            except (ValueError, IndexError):
                n_excluded_no_date += 1
                continue
            if year < 2008 or year > 2026:
                n_excluded_bad_year += 1
                if len(bad_year_examples) < 5:
                    bad_year_examples.append(f"{repo}: {created_at}")
                continue

            try:
                stars = int(rec.get("stars") or 0)
            except (TypeError, ValueError):
                stars = 0

            # Engineered-subset gate (matches build_analysis_set.py /
            # analysis_set_definition.json engineered_subset_stars*N* rule):
            # disabled=False AND is_template=False AND commits>1 AND
            # contributors>1 AND stars >= N. Stars threshold is applied
            # at flag time below. status_error is NOT in this JSONL
            # (already filtered out at analysis_set construction), so
            # it is implicitly satisfied.
            commits = rec.get("num_commits_default_branch")
            contribs = rec.get("num_contributors_api_approx")
            passes_non_stars_engineered = (
                not bool(rec.get("disabled"))
                and not bool(rec.get("is_template"))
                and commits is not None and commits > 1
                and contribs is not None and contribs > 1
            )

            n_models = len(rec.get("models") or [])
            rows.append({
                "repo_id": repo,
                "created_at_year": year,
                "created_at_date": created_at,
                "primary_method": dom,
                "method_bucket": bucket_for(dom),
                "n_referenced_models": n_models,
                "stars": stars,
                # is_starsN flags match RQ1 Table 6 engineered subsets:
                # engineered filter (commits>1, contributors>1, not disabled,
                # not template) AND stars >= N.
                "is_stars1":  passes_non_stars_engineered and stars >= 1,
                "is_stars5":  passes_non_stars_engineered and stars >= 5,
                "is_stars10": passes_non_stars_engineered and stars >= 10,
            })

    df = pd.DataFrame(rows)
    df.to_csv(FRAME_CSV, index=False)

    logp(f"  rows in frame      : {len(df):,}")
    logp(f"  excluded (no primary method): {n_excluded_no_primary:,}")
    logp(f"  excluded (no created_at)    : {n_excluded_no_date:,}")
    logp(f"  excluded (year out of range): {n_excluded_bad_year:,}")
    if bad_year_examples:
        for ex in bad_year_examples:
            logp(f"    — {ex}")
    logp(f"  total analysis-set repos walked: {n_total:,}")
    logp()

    # Sanity checks
    year_counts = df["created_at_year"].value_counts().to_dict()
    bucket_counts = df["method_bucket"].value_counts().to_dict()
    stars_counts = {
        "is_stars1":  int(df["is_stars1"].sum()),
        "is_stars5":  int(df["is_stars5"].sum()),
        "is_stars10": int(df["is_stars10"].sum()),
    }

    logp("Distribution of created_at_year:")
    logp(text_histogram(year_counts))
    logp()
    logp("Distribution of method_bucket:")
    logp(text_histogram(bucket_counts, label_width=14))
    logp()
    logp("Stars-threshold subset counts (compare to RQ1 Table 6):")
    for k, expected in EXPECTED_STARS_COUNTS.items():
        observed = stars_counts[k]
        pct_diff = abs(observed - expected) / expected * 100
        flag = ""
        if pct_diff > 5:
            flag = f"  ⚠ off by {pct_diff:.1f}%"
        logp(f"  {k:<12} observed={observed:>5,}   expected={expected:>5,}{flag}")
    logp()

    year_order = sorted(df["created_at_year"].unique())

    # ----------------------------------------------------------
    # Stage 2: matrices for all + 3 stars-threshold subsets
    # ----------------------------------------------------------
    counts_all, cohort_share_all, method_share_all = build_matrices(df, year_order)
    counts_s1, cohort_share_s1, method_share_s1 = build_matrices(df[df["is_stars1"]], year_order)
    counts_s5, cohort_share_s5, method_share_s5 = build_matrices(df[df["is_stars5"]], year_order)
    counts_s10, cohort_share_s10, method_share_s10 = build_matrices(df[df["is_stars10"]], year_order)

    save_matrices(counts_all, cohort_share_all, method_share_all, "")
    save_matrices(counts_s1, cohort_share_s1, method_share_s1, "_stars1")
    save_matrices(counts_s5, cohort_share_s5, method_share_s5, "_stars5")
    save_matrices(counts_s10, cohort_share_s10, method_share_s10, "_stars10")

    # Trajectories for verdict
    def summary_of(counts, cohort_share):
        return {
            m: cohort_share_trajectory(cohort_share, counts, m)
            for m in NAMED_METHODS
        }
    all_summary = summary_of(counts_all, cohort_share_all)
    s1_summary = summary_of(counts_s1, cohort_share_s1)
    s5_summary = summary_of(counts_s5, cohort_share_s5)
    s10_summary = summary_of(counts_s10, cohort_share_s10)

    verdict, finding, robustness = determine_verdict(
        all_summary, s1_summary, s5_summary, s10_summary
    )

    # ----------------------------------------------------------
    # Stage 3: diagnostics markdown
    # ----------------------------------------------------------
    md: list[str] = []
    md.append("# RQ2 Phase 0 - Repository Cohort Analysis")
    md.append("")
    md.append("> **Scope note:** Phase 0 evaluates repository cohorts rather")
    md.append("> than adoption events. Observed differences motivate, but do")
    md.append("> not establish, temporal adoption dynamics; these require")
    md.append("> git-history mining in subsequent phases. Creation year is")
    md.append("> used here as a proxy for repository vintage, not adoption")
    md.append("> time. Any phrasing in this document should describe what")
    md.append("> cohorts of repositories use, not when methods were adopted.")
    md.append("")

    md.append("## Inputs used")
    md.append("")
    for label, p in [
        ("analysis_set_repos.txt", ANALYSIS_REPOS_TXT),
        ("analysis_set_repo_details.jsonl", ANALYSIS_JSONL),
        ("quantized_models_all.jsonl", ALL_MODELS_JSONL),
    ]:
        md.append(f"- `{p}` ({p.stat().st_size:,} bytes) — {label}")
    md.append("")

    md.append("## Coverage summary")
    md.append("")
    md.append(f"- Total repos in analysis set: **{n_total:,}**")
    md.append(f"- Excluded from Phase 0: **{n_excluded_no_primary + n_excluded_no_date + n_excluded_bad_year:,}** "
              f"({n_excluded_no_primary:,} no primary method, "
              f"{n_excluded_no_date:,} no created_at, "
              f"{n_excluded_bad_year:,} year out of range)")
    md.append(f"- Frame rows: **{len(df):,}**")
    md.append("")
    md.append("Distribution of `created_at_year`:")
    md.append("")
    md.append("```")
    md.append(text_histogram(year_counts))
    md.append("```")
    md.append("")
    md.append("Distribution of `method_bucket`:")
    md.append("")
    md.append("```")
    md.append(text_histogram(bucket_counts, label_width=14))
    md.append("```")
    md.append("")
    md.append("Stars-threshold subset counts (compared to RQ1 Table 6 expected values):")
    md.append("")
    md.append("| subset | observed | expected (Table 6) |")
    md.append("|---|---:|---:|")
    for k, expected in EXPECTED_STARS_COUNTS.items():
        md.append(f"| {k} | {stars_counts[k]:,} | {expected:,} |")
    md.append("")

    md.append("## Matrix: counts, all repos")
    md.append("")
    md.append(df_to_md(counts_all.assign(Total=counts_all.sum(axis=1)),
                       float_fmt=None, index_label="year"))
    md.append("")
    md.append("## Matrix: within-cohort shares, all repos")
    md.append("")
    md.append(df_to_md(cohort_share_all.round(4).assign(
        Total=cohort_share_all.sum(axis=1).round(4)), index_label="year"))
    md.append("")
    md.append("## Matrix: within-method shares, all repos")
    md.append("")
    ms_with_total = method_share_all.round(4).copy()
    col_totals = pd.DataFrame([method_share_all.sum(axis=0).round(4)], index=["Total"])
    ms_with_total = pd.concat([ms_with_total, col_totals])
    md.append(df_to_md(ms_with_total, index_label="year"))
    md.append("")

    md.append("## Matrix: within-cohort shares, stars>=1 subset")
    md.append("")
    md.append(df_to_md(cohort_share_s1.round(4).assign(
        Total=cohort_share_s1.sum(axis=1).round(4)), index_label="year"))
    md.append("")
    md.append("## Matrix: within-cohort shares, stars>=5 subset")
    md.append("")
    md.append(df_to_md(cohort_share_s5.round(4).assign(
        Total=cohort_share_s5.sum(axis=1).round(4)), index_label="year"))
    md.append("")
    md.append("## Matrix: within-cohort shares, stars>=10 subset")
    md.append("")
    md.append(df_to_md(cohort_share_s10.round(4).assign(
        Total=cohort_share_s10.sum(axis=1).round(4)), index_label="year"))
    md.append("")

    # Robustness section
    md.append("## Robustness across stars thresholds")
    md.append("")
    md.append(
        "For each named method, we compare its within-cohort share trajectory "
        "across the full set and the three engineered subsets. A trajectory "
        "is the sequence of within-cohort shares for qualifying years "
        f"(cohort size ≥ {COHORT_MIN_REPOS} repos). The signal is **robust** "
        "if the share moves in the same direction (sign of Δ) across all four "
        "views."
    )
    md.append("")
    md.append("| method | view | earliest year | earliest share | latest year | latest share | Δ pp |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for m in NAMED_METHODS:
        for view_name, summary in [
            ("all",      all_summary),
            ("stars≥1",  s1_summary),
            ("stars≥5",  s5_summary),
            ("stars≥10", s10_summary),
        ]:
            t = summary[m]
            if t.get("delta_pp") is None:
                md.append(f"| {m} | {view_name} | — | — | — | — | — |")
            else:
                md.append(
                    f"| {m} | {view_name} | "
                    f"{t['earliest_year']} | {t['earliest_share']*100:.1f}% | "
                    f"{t['latest_year']} | {t['latest_share']*100:.1f}% | "
                    f"{t['delta_pp']:+.1f} |"
                )
    md.append("")
    md.append("Per-method robustness verdict (signs of Δ across four views):")
    md.append("")
    for m in NAMED_METHODS:
        signs = []
        for label, summary in [("all", all_summary), ("stars≥1", s1_summary),
                                ("stars≥5", s5_summary), ("stars≥10", s10_summary)]:
            d = summary[m].get("delta_pp")
            signs.append(("+" if d is not None and d > 0
                          else "−" if d is not None and d < 0
                          else "·"))
        same = (len(set(s for s in signs if s != "·")) <= 1)
        tag = "stable across all thresholds" if same else "direction changes across thresholds"
        md.append(f"- **{m}**: signs = {' '.join(signs)} → {tag}")
    md.append("")
    md.append(
        "*Reminder: stability of direction is a positive signal; direction "
        "changes across thresholds indicate the cohort-displacement pattern "
        "is concentrated in one quality tier — note this in Phase 1 design.*"
    )
    md.append("")

    md.append("## Observations")
    md.append("")
    md.append(
        "All observations below describe what cohorts of repositories use, "
        "not when methods were adopted."
    )
    md.append("")
    for m in NAMED_METHODS:
        t = all_summary[m]
        if not t["years"]:
            md.append(f"- **{m}**: no qualifying cohort years (cohort size "
                      f"≥ {COHORT_MIN_REPOS}); skipping.")
            continue
        per_year_shares = ", ".join(
            f"{y}={s*100:.1f}%" for y, s in zip(t["years"], t["shares"])
        )
        md.append(
            f"- **{m}**: within-cohort shares across qualifying years: "
            f"{per_year_shares}. Peak cohort year = "
            f"**{t['peak_year']}** ({t['peak_share']*100:.1f}%). "
            f"Δ between earliest ({t['earliest_year']}) and latest "
            f"({t['latest_year']}) qualifying cohorts: "
            f"**{t['delta_pp']:+.1f} pp**."
        )
        # Does the share cross 50% in any cohort year?
        crosses_50 = [y for y, s in zip(t["years"], t["shares"]) if s >= 0.50]
        if crosses_50:
            md.append(
                f"  - {m}'s within-cohort share **crosses 50%** in cohort "
                f"year(s): {', '.join(map(str, crosses_50))}."
            )
        # Drop by more than half from peak?
        if t["peak_share"] > 0 and (t["shares"][-1] / t["peak_share"]) < 0.5 \
                and t["peak_year"] != t["latest_year"]:
            md.append(
                f"  - {m}'s share **drops by >50%** from peak "
                f"({t['peak_share']*100:.1f}%) to latest cohort "
                f"({t['shares'][-1]*100:.1f}%)."
            )
        # Peak-to-latest delta
        peak_to_latest = (t["shares"][-1] - t["peak_share"]) * 100
        md.append(
            f"  - Peak-to-latest Δ for {m}: {peak_to_latest:+.1f} pp "
            f"(from {t['peak_year']} to {t['latest_year']})."
        )
    md.append("")

    md.append("## Verdict on RQ2 mining")
    md.append("")
    verdict_label = {
        "a": "STRONG COHORT SIGNAL",
        "b": "MODERATE COHORT SIGNAL",
        "c": "WEAK COHORT SIGNAL",
    }[verdict]
    md.append(f"**Verdict: ({verdict}) {verdict_label}**")
    md.append("")
    if verdict == "a":
        md.append("Candidate finding sentence (Phase 0 alone supports this; "
                  "Phase 1 mining is required to convert it to an adoption "
                  "claim):")
        md.append("")
        md.append(f"> {finding}")
        md.append("")
        md.append(robustness)
        md.append("")
    elif verdict == "b":
        max_method = max(
            (m for m in NAMED_METHODS if all_summary[m].get("delta_pp") is not None),
            key=lambda m: abs(all_summary[m]["delta_pp"]),
        )
        d = all_summary[max_method]["delta_pp"]
        md.append(
            f"The largest within-cohort share change for a named method "
            f"in the full set is {abs(d):.1f} pp ({max_method}, "
            f"{all_summary[max_method]['earliest_year']} → "
            f"{all_summary[max_method]['latest_year']}). This is between "
            "the 10 pp moderate-floor and the 20 pp strong-floor; Phase 0 "
            "alone is not enough to motivate a finding, but the pilot is "
            "warranted to confirm with git-history dates rather than repo "
            "creation dates."
        )
        md.append("")
    else:
        md.append(
            "Method shares vary only minimally across cohorts "
            "(<10 pp differences for the named methods on which a delta could "
            "be computed). The pilot may not produce a finding worth the "
            "mining effort. user should reconsider RQ2's payoff before "
            "scaling git-history mining."
        )
        md.append("")
    md.append("")

    DIAGNOSTICS_MD.write_text("\n".join(md) + "\n")

    # ----------------------------------------------------------
    # Stage 4: optional 100% stacked bar plot
    # ----------------------------------------------------------
    plot_made = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        bottom = pd.Series(0.0, index=cohort_share_all.index)
        colors = {
            "GGUF": "#3a6f9a", "BitsAndBytes": "#c97954", "GPTQ": "#6a9a3a",
            "AWQ": "#9a5a3a", "Other": "#8c8c8c",
        }
        for m in METHOD_ORDER:
            vals = cohort_share_all[m] * 100
            ax.bar(cohort_share_all.index.astype(str), vals, bottom=bottom,
                   label=m, color=colors[m], edgecolor="white", linewidth=0.4)
            bottom = bottom + vals
        ax.set_ylim(0, 100)
        ax.set_xlabel("Repository creation year (cohort)")
        ax.set_ylabel("Within-cohort share (%)")
        ax.set_title("Method composition by repository cohort")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_made = True
    except Exception as e:
        logp(f"(skipped optional plot: {type(e).__name__}: {e})")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    logp()
    logp("=" * 60)
    logp(f"repo_frame.csv         : {FRAME_CSV} ({len(df):,} rows)")
    logp(f"phase0_diagnostics.md  : {DIAGNOSTICS_MD}")
    if plot_made:
        logp(f"phase0_cohort_stacked_bars.png : {PLOT_PNG}")
    logp()
    logp(f"VERDICT: ({verdict}) {verdict_label}")
    if verdict == "a":
        logp(f"  Finding: {finding}")
        logp(f"  Robustness: {robustness}")
    elif verdict == "b":
        logp("  Moderate signal — pilot needed to confirm.")
    else:
        logp("  Weak signal — reconsider RQ2 payoff.")


if __name__ == "__main__":
    main()

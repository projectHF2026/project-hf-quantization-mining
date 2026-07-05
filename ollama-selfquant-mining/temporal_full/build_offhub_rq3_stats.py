"""Off-hub RQ3 temporal stats (paper RQ3 = server-side RQ2).

Pure computation. Read offhub_temporal_first_adoption_clean.csv, run the
same statistical tests the hub uses in
output_dir/rq_analysis/rq2/phase1_full/scripts/rq2_phase1_diagnostics.py
(verbatim ports: linregress_ci, mann_kendall_simple, bootstrap_median_ci,
chi2_contingency + Cramer's V).

Outputs (in temporal_full/):
  offhub_rq3_method_share_by_year.csv     — count + share matrices
  offhub_rq3_trend_stats.txt              — chi2, V, OLS, MK, growth
  offhub_rq3_lag.csv                      — matched-subset rows
  offhub_rq3_lag_summary.txt              — lag stats + zero-lag note
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


ROOT       = Path(__file__).resolve().parent
INPUT      = ROOT / "offhub_temporal_first_adoption_clean.csv"
HF_MODELS  = Path(
    "/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/"
    "output_dir/HuggingFaceStudy/modelsInfo"
)

OUT_SHARE  = ROOT / "offhub_rq3_method_share_by_year.csv"
OUT_TREND  = ROOT / "offhub_rq3_trend_stats.txt"
OUT_LAG    = ROOT / "offhub_rq3_lag.csv"
OUT_LAGSUM = ROOT / "offhub_rq3_lag_summary.txt"

YEARS              = [2022, 2023, 2024, 2025, 2026]
PARTIAL_YEAR       = 2026
DOMINANT_FAMILIES  = ["llama.cpp/GGUF-convert", "GPTQ", "AWQ", "Ollama-obtain"]
LOWER_BOUND_NOTE = (
    "Lower-bound framing: first-commit dates from default-branch git history "
    "can be re-based or squashed; first-use dates therefore lower-bound true "
    "adoption. Default-branch-only mining means feature-branch first-uses are "
    "not seen."
)


# ---------------------------------------------------------------------------
# Statistical functions — VERBATIM PORTS from
# output_dir/rq_analysis/rq2/phase1_full/scripts/rq2_phase1_diagnostics.py
# ---------------------------------------------------------------------------

def linregress_ci(xs, ys, conf: float = 0.95):
    if len(xs) < 3:
        return None
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    res = stats.linregress(xs, ys)
    n = len(xs)
    t = stats.t.ppf(1 - (1 - conf) / 2, n - 2)
    lo = res.slope - t * res.stderr
    hi = res.slope + t * res.stderr
    return {
        "slope":     float(res.slope),
        "intercept": float(res.intercept),
        "lo":        float(lo),
        "hi":        float(hi),
        "p":         float(res.pvalue),
        "stderr":    float(res.stderr),
    }


def mann_kendall_simple(series):
    if len(series) < 3:
        return None, None
    res = stats.kendalltau(np.arange(len(series)), series)
    return float(res.statistic), float(res.pvalue)


def bootstrap_median_ci(data, n_iter: int = 10_000, conf: float = 0.95,
                        seed: int = 42):
    if not data:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, dtype=float)
    meds = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sample = rng.choice(arr, size=len(arr), replace=True)
        meds[i] = np.median(sample)
    a = (1 - conf) / 2
    return {
        "median": float(np.median(arr)),
        "lo":     float(np.quantile(meds, a)),
        "hi":     float(np.quantile(meds, 1 - a)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_dt(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def model_file_path(model_id: str) -> Path:
    return HF_MODELS / (model_id.replace("/", "£sep£") + ".json")


# Strip Ollama-style :tag suffixes and known URL prefixes
TAG_RE = re.compile(r":[^/]+$")
URL_PREFIX_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co|"
    r"ollama\.com/library|registry\.ollama\.ai)/"
)
# Characters that mark the raw value as code/template garbage rather than a
# real model identifier (placeholders, function calls, file paths, etc.)
GARBAGE_CHARS_RE = re.compile(r"[\s<>`\"'(){}\[\]\\@$]")
# Identifier shape: must be HF-like  org/name where each part is
# alphanumeric + . _ -
VALID_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def normalise_pulled_id(raw: str) -> tuple[str, str]:
    """Return (normalised_hf_candidate, classification).

    classification ∈ {
      'hf_candidate':    looks like org/name; try HF lookup
      'ollama_native':   no slash; e.g. 'llama3.1:8b' — Ollama registry only,
                         won't resolve to HF
      'garbage':         contains code placeholders / paths / quotes / etc.
      'empty':           empty/None
    }"""
    if not raw:
        return "", "empty"
    s = raw.strip()
    s = URL_PREFIX_RE.sub("", s)
    s = TAG_RE.sub("", s)   # strip :tag
    s = s.strip("/")
    s = s.rstrip(",;)\"'")
    if not s:
        return "", "empty"
    # Drop file-path-style values (e.g. './model.gguf')
    if s.startswith(".") or s.startswith("/"):
        return s, "garbage"
    if GARBAGE_CHARS_RE.search(s):
        return s, "garbage"
    if "/" not in s:
        return s, "ollama_native"
    parts = s.split("/")
    if (len(parts) >= 2 and parts[0] and parts[1]
            and VALID_PART_RE.match(parts[0]) and VALID_PART_RE.match(parts[1])):
        return f"{parts[0]}/{parts[1]}", "hf_candidate"
    return s, "garbage"


def lookup_hf_created_at(hf_id: str) -> str:
    """Return the model's createdAt ISO string from the raw HF JSON dump, or ""
    if the file isn't present."""
    p = model_file_path(hf_id)
    if not p.exists():
        return ""
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return ""
    # HF API field is 'createdAt'; the scrape preserves it
    return str(raw.get("createdAt") or "").strip()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_rows():
    if not INPUT.exists():
        sys.exit(f"ERROR: {INPUT} not found")
    rows = []
    with INPUT.open() as f:
        for r in csv.DictReader(f):
            dt = parse_dt(r["first_commit_date"])
            if dt is None:
                continue
            r["_dt"] = dt
            r["_year"] = dt.year
            # Use UTC for month key to keep deterministic ordering
            r["_ym"] = (dt.year, dt.month)
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Part 1 — method-by-year matrix + chi2/V + per-family OLS+MK
# ---------------------------------------------------------------------------

def part1(rows):
    families = sorted({r["method_family"] for r in rows})
    # year × family counts (years restricted to 2022..2026)
    year_fam_counts: dict[int, Counter] = {y: Counter() for y in YEARS}
    for r in rows:
        y = r["_year"]
        if y in year_fam_counts:
            year_fam_counts[y][r["method_family"]] += 1
    year_totals = {y: sum(year_fam_counts[y].values()) for y in YEARS}

    # Write counts + shares CSV
    with OUT_SHARE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "year", "is_partial"] + families + ["year_total"])
        for y in YEARS:
            row = ["counts", y, "yes" if y == PARTIAL_YEAR else "no"]
            for fam in families:
                row.append(year_fam_counts[y].get(fam, 0))
            row.append(year_totals[y])
            w.writerow(row)
        w.writerow([])
        w.writerow(["section", "year", "is_partial"] + families + ["year_total"])
        for y in YEARS:
            row = ["shares_pct", y, "yes" if y == PARTIAL_YEAR else "no"]
            tot = year_totals[y]
            for fam in families:
                cnt = year_fam_counts[y].get(fam, 0)
                row.append(f"{100*cnt/tot:.2f}" if tot else "0.00")
            row.append(tot)
            w.writerow(row)

    # Aggregate growth per year
    growth = {y: year_totals[y] for y in YEARS}

    # chi-square independence + Cramer's V
    table = np.array(
        [[year_fam_counts[y].get(fam, 0) for fam in families] for y in YEARS],
        dtype=float,
    )
    chi2_res = None
    if table.size and table.sum() > 0 and table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, chi2_p, dof, _ = stats.chi2_contingency(table)
        n_total = table.sum()
        min_dim = min(table.shape) - 1
        cramers_v = (
            float(np.sqrt(chi2 / (n_total * min_dim))) if min_dim > 0 else None
        )
        chi2_res = {
            "chi2": float(chi2), "p": float(chi2_p), "dof": int(dof),
            "cramers_v": cramers_v, "n_total": int(n_total),
        }

    # Monthly series per dominant family
    # First derive a global month sequence covering the data
    all_months = sorted({r["_ym"] for r in rows
                          if 2022 <= r["_ym"][0] <= 2026})
    fam_monthly = {fam: Counter() for fam in DOMINANT_FAMILIES}
    for r in rows:
        if r["_ym"] in all_months:
            fam = r["method_family"]
            if fam in fam_monthly:
                fam_monthly[fam][r["_ym"]] += 1

    per_family = {}
    for fam in DOMINANT_FAMILIES:
        counts = [fam_monthly[fam].get(m, 0) for m in all_months]
        xs = list(range(len(counts)))
        lr = linregress_ci(xs, counts)
        # Share series: monthly count divided by total adoptions that month
        total_monthly = Counter()
        for r in rows:
            if r["_ym"] in all_months:
                total_monthly[r["_ym"]] += 1
        share_series = [
            (fam_monthly[fam].get(m, 0) / total_monthly[m]) if total_monthly[m] else 0.0
            for m in all_months
        ]
        mk_tau, mk_p = mann_kendall_simple(share_series)
        per_family[fam] = {
            "n_months":         len(counts),
            "monthly_counts":   counts,
            "monthly_shares":   share_series,
            "lr":               lr,
            "mk_tau":           mk_tau,
            "mk_p":             mk_p,
            "n_total_events":   sum(counts),
        }

    return {
        "families":          families,
        "year_fam_counts":   year_fam_counts,
        "year_totals":       year_totals,
        "all_months":        all_months,
        "growth":            growth,
        "chi2":              chi2_res,
        "per_family":        per_family,
    }


def write_trend_txt(p1):
    L = []
    L.append("Off-hub RQ3.1 — method adoption over time")
    L.append("Methodology mirrored from:")
    L.append("  output_dir/rq_analysis/rq2/phase1_full/scripts/rq2_phase1_diagnostics.py")
    L.append(f"Lower-bound framing: {LOWER_BOUND_NOTE}")
    L.append("")
    L.append("=" * 70)
    L.append("Aggregate growth — total first-signal events per year")
    L.append("=" * 70)
    for y in YEARS:
        tag = " (PARTIAL)" if y == PARTIAL_YEAR else ""
        L.append(f"  {y}: {p1['growth'][y]:>5,}{tag}")
    L.append("")
    L.append("=" * 70)
    L.append("Chi-square test of independence: method_family × year")
    L.append("=" * 70)
    c = p1["chi2"]
    if c is None:
        L.append("  insufficient table dimensions")
    else:
        L.append(f"  chi2       = {c['chi2']:.3f}")
        L.append(f"  dof        = {c['dof']}")
        L.append(f"  p-value    = {c['p']:.4e}")
        L.append(f"  Cramer's V = {c['cramers_v']:.4f}")
        L.append(f"  n_total    = {c['n_total']:,}")
        verdict = "REJECTED (families and years not independent)" if c["p"] < 0.05 \
            else "NOT REJECTED"
        L.append(f"  H0 (independence): {verdict}")
    L.append("")
    L.append("=" * 70)
    L.append("Per dominant-family monthly trend tests")
    L.append("  - OLS slope + 95% CI on monthly count series")
    L.append("  - Mann-Kendall (tau, p) on monthly share series")
    L.append("=" * 70)
    for fam in DOMINANT_FAMILIES:
        f = p1["per_family"][fam]
        L.append(f"\n  family: {fam}")
        L.append(f"    n_events_in_window = {f['n_total_events']:,}")
        L.append(f"    n_months           = {f['n_months']}")
        if f["lr"] is None:
            L.append(f"    OLS slope: insufficient months")
        else:
            lr = f["lr"]
            sig = "+" if lr["slope"] > 0 else "−"
            ci_strict = "strictly positive" if lr["lo"] > 0 \
                else "strictly negative" if lr["hi"] < 0 \
                else "overlaps zero"
            L.append(f"    OLS monthly-count slope = {lr['slope']:+.4f} "
                     f"events/month (95% CI [{lr['lo']:+.4f}, {lr['hi']:+.4f}]; "
                     f"p={lr['p']:.4e}; CI {ci_strict})")
        if f["mk_tau"] is None:
            L.append(f"    Mann-Kendall on share series: insufficient months")
        else:
            mk_sig = "significant" if f["mk_p"] < 0.05 else "not significant"
            direction = ("rising" if f["mk_tau"] > 0
                         else "falling" if f["mk_tau"] < 0 else "flat")
            L.append(f"    Mann-Kendall on share series: tau = {f['mk_tau']:+.4f}, "
                     f"p = {f['mk_p']:.4e}  ({direction}, {mk_sig})")
    L.append("")
    OUT_TREND.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Part 2 — adoption lag for ollama_obtains_model with pulled_model_id
# ---------------------------------------------------------------------------

def part2(rows):
    obtain_rows = [r for r in rows
                   if r["mode"] == "ollama_obtains_model"
                   and (r.get("pulled_model_id") or "").strip()]
    self_q_count = sum(1 for r in rows if r["mode"] == "self_quantized")

    n_total_obtain = sum(1 for r in rows if r["mode"] == "ollama_obtains_model")

    # Classify each pulled_model_id, attempt HF lookup
    matched = []          # list of dicts: repo_id, raw, hf_id, model_created, first, lag_days
    unmatched_hf      = 0
    unmatched_ollama  = 0
    unmatched_empty   = 0
    unmatched_garbage = 0
    n_negative        = 0

    for r in obtain_rows:
        raw = r["pulled_model_id"].strip()
        hf_id, cls = normalise_pulled_id(raw)
        if cls == "empty":
            unmatched_empty += 1
            continue
        if cls == "garbage":
            unmatched_garbage += 1
            continue
        if cls == "ollama_native":
            unmatched_ollama += 1
            continue
        created = lookup_hf_created_at(hf_id)
        if not created:
            unmatched_hf += 1
            continue
        mdt = parse_dt(created)
        if mdt is None:
            unmatched_hf += 1
            continue
        first_dt = r["_dt"]
        lag_days = (first_dt - mdt).total_seconds() / 86400.0
        if lag_days < 0:
            n_negative += 1
        matched.append({
            "repo_id":          r["repo_id"],
            "raw_pulled_id":    raw,
            "hf_model_id":      hf_id,
            "model_created":    mdt.isoformat(timespec="seconds"),
            "first_commit_date": first_dt.isoformat(timespec="seconds"),
            "lag_days":         lag_days,
        })

    # Stats
    lags = [m["lag_days"] for m in matched]
    arr = np.asarray(lags) if lags else np.array([])
    if arr.size:
        ci = bootstrap_median_ci(lags, n_iter=10_000, seed=42)
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        med = ci["median"]
        ci_lo, ci_hi = ci["lo"], ci["hi"]
        mean = float(arr.mean())
        mx = float(arr.max())
        mn = float(arr.min())
    else:
        med = ci_lo = ci_hi = p25 = p75 = mean = mx = mn = None

    # Write CSV
    with OUT_LAG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "repo_id", "raw_pulled_id", "hf_model_id",
            "model_created", "first_commit_date", "lag_days"])
        w.writeheader()
        for m in matched:
            w.writerow({**m, "lag_days": f"{m['lag_days']:.4f}"})

    # Lag summary text
    L = []
    L.append("Off-hub RQ3.2 — adoption lag (ollama_obtains_model subset)")
    L.append("Methodology mirrored from:")
    L.append("  output_dir/rq_analysis/rq2/phase1_full/scripts/rq2_phase1_diagnostics.py")
    L.append(f"Lower-bound framing: {LOWER_BOUND_NOTE}")
    L.append("")
    L.append(f"ollama_obtains_model rows total:                  {n_total_obtain:,}")
    L.append(f"  with non-empty pulled_model_id:                 {len(obtain_rows):,}")
    L.append(f"  matched to HF metadata (hf_candidate + found):  {len(matched):,}")
    L.append(f"Unmatched breakdown:")
    L.append(f"  Ollama-native (no '/' in pulled_id):            {unmatched_ollama:,}")
    L.append(f"  HF-shaped but not in HuggingFaceStudy dump:     {unmatched_hf:,}")
    L.append(f"  garbage (code placeholders, file paths, etc.):  {unmatched_garbage:,}")
    L.append(f"  empty after URL/tag normalization:              {unmatched_empty:,}")
    L.append("")
    L.append(f"Self-quantized cohort (zero acquisition lag by construction):")
    L.append(f"  n = {self_q_count:,} repos produce their quantized model in-repo;")
    L.append(f"  these are NOT included in the lag distribution.")
    L.append("")
    L.append("=" * 70)
    L.append("Lag statistics on the matched subset (days)")
    L.append("  median + bootstrap 95% CI (10,000 resamples, seed=42)")
    L.append("=" * 70)
    if med is None:
        L.append("  no matched rows; no statistics.")
    else:
        L.append(f"  n               = {len(lags):,}")
        L.append(f"  negative lags   = {n_negative:,}  "
                 f"(commit BEFORE HF model creation; flagged, not dropped)")
        L.append(f"  median (days)   = {med:.1f}")
        L.append(f"  95% CI median   = [{ci_lo:.1f}, {ci_hi:.1f}]")
        L.append(f"  Q1 (p25)        = {p25:.1f}")
        L.append(f"  Q3 (p75)        = {p75:.1f}")
        L.append(f"  IQR             = {p75 - p25:.1f}")
        L.append(f"  mean            = {mean:.1f}")
        L.append(f"  min / max       = {mn:.1f} / {mx:.1f}")
    OUT_LAGSUM.write_text("\n".join(L) + "\n")

    return {
        "n_obtain_total":   n_total_obtain,
        "n_with_pulled":    len(obtain_rows),
        "n_matched":        len(matched),
        "n_unmatched_ollama_native": unmatched_ollama,
        "n_unmatched_hf_shaped_not_found": unmatched_hf,
        "n_unmatched_garbage":  unmatched_garbage,
        "n_unmatched_empty": unmatched_empty,
        "n_negative_lag":   n_negative,
        "median":           med,
        "ci_lo":            ci_lo,
        "ci_hi":            ci_hi,
        "p25":              p25,
        "p75":              p75,
        "self_quant_count": self_q_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = load_rows()
    print(f"Loaded {len(rows):,} rows from {INPUT.name}")
    print(f"Template script: "
          f"output_dir/rq_analysis/rq2/phase1_full/scripts/rq2_phase1_diagnostics.py")
    print(f"  (functions reused verbatim: linregress_ci, mann_kendall_simple, "
          f"bootstrap_median_ci, chi2_contingency + Cramer's V)")
    print()

    p1 = part1(rows)
    write_trend_txt(p1)
    print(f"Wrote {OUT_SHARE.name}")
    print(f"Wrote {OUT_TREND.name}")

    p2 = part2(rows)
    print(f"Wrote {OUT_LAG.name}")
    print(f"Wrote {OUT_LAGSUM.name}")

    # Headline summary print
    print()
    print("=" * 78)
    print("HEADLINE — Off-hub RQ3 (paper)  /  RQ2 (server-side)")
    print("=" * 78)
    print()
    print("PART 1: method adoption over time")
    print(f"  Aggregate growth (total first-signals per year):")
    for y in YEARS:
        tag = " [PARTIAL]" if y == PARTIAL_YEAR else ""
        print(f"    {y}: {p1['growth'][y]:>5,}{tag}")
    c = p1["chi2"]
    if c:
        v = "REJECTED" if c["p"] < 0.05 else "NOT REJECTED"
        print(f"  Chi-square method × year independence: chi2 = {c['chi2']:.1f}, "
              f"p = {c['p']:.2e}, Cramer's V = {c['cramers_v']:.3f}  [H0 {v}]")
    print()
    print(f"  Dominant-family monthly trend direction + significance:")
    for fam in DOMINANT_FAMILIES:
        f = p1["per_family"][fam]
        if f["lr"] is None or f["mk_tau"] is None:
            print(f"    {fam:<26} (insufficient months)")
            continue
        lr = f["lr"]
        ci_strict = "+CI" if lr["lo"] > 0 \
            else "-CI" if lr["hi"] < 0 else "CI~0"
        mk_sig = "*" if f["mk_p"] < 0.05 else ""
        direction = ("↑" if f["mk_tau"] > 0
                     else "↓" if f["mk_tau"] < 0 else "→")
        print(f"    {fam:<26} OLS slope {lr['slope']:+.3f} "
              f"events/mo ({ci_strict}, p={lr['p']:.2e})  "
              f"MK tau {f['mk_tau']:+.3f}{mk_sig} {direction}")
    print()
    print("PART 2: adoption lag (ollama_obtains_model with HF-resolvable pulled_id)")
    print(f"  ollama_obtains_model rows:               {p2['n_obtain_total']:,}")
    print(f"  with non-empty pulled_model_id:          {p2['n_with_pulled']:,}")
    print(f"  matched to HF metadata:                  {p2['n_matched']:,}")
    print(f"  unmatched - Ollama-registry-native names: {p2['n_unmatched_ollama_native']:,}")
    print(f"  unmatched - HF-shaped, not in dump:       {p2['n_unmatched_hf_shaped_not_found']:,}")
    print(f"  unmatched - garbage capture (paths/etc.): {p2['n_unmatched_garbage']:,}")
    if p2["median"] is not None:
        print(f"  median lag (days):                       {p2['median']:.1f}")
        print(f"  95% CI on median:                        "
              f"[{p2['ci_lo']:.1f}, {p2['ci_hi']:.1f}]")
        print(f"  IQR (Q1–Q3):                             "
              f"[{p2['p25']:.1f}, {p2['p75']:.1f}]")
        print(f"  negative-lag rows (flagged, kept):       {p2['n_negative_lag']:,}")
    print(f"  self_quantized cohort:                   {p2['self_quant_count']:,} "
          f"repos — zero acquisition lag by construction (excluded).")


if __name__ == "__main__":
    main()

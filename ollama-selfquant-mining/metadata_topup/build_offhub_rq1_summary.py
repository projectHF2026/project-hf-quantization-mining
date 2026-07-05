"""Off-hub RQ1 summary (paper RQ1 = server-side rq0).

Pure computation. Read offhub_repo_details.jsonl, write offhub_rq1_summary.json.
Mirrors output_dir/rq_analysis/rq0/scripts/rq0_analysis.py exactly: identical
bucket cuts, identical summary stats, identical missing-value convention.

Outputs in ollama-selfquant-mining/metadata_topup/:
  offhub_rq1_summary.json                 — structured like rq0_summary.json
                                            with top-level overall + per-mode
                                            blocks (mode = self_quantized /
                                            ollama_obtains_model /
                                            ollama_backend_only)
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT       = Path(__file__).resolve().parent
INPUT      = ROOT / "offhub_repo_details.jsonl"
OUTPUT     = ROOT / "offhub_rq1_summary.json"

HUB_SUMMARY = Path(
    "/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/"
    "output_dir/rq_analysis/rq0/results/rq0_summary.json"
)

INF = float("inf")

STARS_BUCKETS = [
    ("0",       0,   0),
    ("1",       1,   1),
    ("2-4",     2,   4),
    ("5-9",     5,   9),
    ("10-49",   10,  49),
    ("50-499",  50,  499),
    (">=500",   500, INF),
]
CONTRIB_BUCKETS = [
    ("1",    1,  1),
    ("2-3",  2,  3),
    ("4-9",  4,  9),
    (">=10", 10, INF),
]
COMMITS_BUCKETS = [
    ("1",       1,   1),
    ("2-10",    2,   10),
    ("11-100",  11,  100),
    (">=101",   101, INF),
]

REFERENCE_DATE = datetime(2026, 6, 20, tzinfo=timezone.utc)
MODE_ORDER = ["self_quantized", "ollama_obtains_model", "ollama_backend_only"]

CONVENTION_NOTE = (
    "n_missing = repos where the signal value is null OR falls outside the "
    "specified bucket cuts. For stars, only nulls are counted (buckets cover "
    "0..inf). For contributors and commits, the spec buckets start at 1, so "
    "values of 0 (representing API-empty / anonymous-only contributors or an "
    "empty default branch) are treated as missing. Summary statistics and "
    "bucket percentages are computed only on the non-missing in-range subset. "
    "The check bucket_sum + n_missing == cohort_total holds for every signal."
)


# ---------------------------------------------------------------------------
# Hub compute_signal logic, ported verbatim (numpy-based percentiles)
# ---------------------------------------------------------------------------

def assign_bucket(value, buckets):
    if value is None:
        return None
    for label, lo, hi in buckets:
        if lo <= value <= hi:
            return label
    return None


def compute_signal(values, buckets):
    in_range = []
    n_null = 0
    n_out_of_range = 0
    bucket_counts = OrderedDict((lbl, 0) for lbl, _, _ in buckets)
    for v in values:
        if v is None:
            n_null += 1
            continue
        b = assign_bucket(v, buckets)
        if b is None:
            n_out_of_range += 1
            continue
        in_range.append(v)
        bucket_counts[b] += 1
    n_missing = n_null + n_out_of_range
    n = len(in_range)
    if n > 0:
        arr = np.array(in_range)
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        stats = {
            "n":               n,
            "n_missing":       n_missing,
            "n_null":          n_null,
            "n_out_of_range_zero": n_out_of_range,
            "min":             int(arr.min()),
            "median":          float(np.median(arr)),
            "mean":            float(arr.mean()),
            "max":             int(arr.max()),
            "Q1":              q1,
            "Q3":              q3,
            "IQR":             q3 - q1,
            "p90":             float(np.percentile(arr, 90)),
            "p95":             float(np.percentile(arr, 95)),
            "p99":             float(np.percentile(arr, 99)),
        }
    else:
        stats = {
            "n": 0, "n_missing": n_missing,
            "n_null": n_null, "n_out_of_range_zero": n_out_of_range,
            "min": None, "median": None, "mean": None, "max": None,
            "Q1": None, "Q3": None, "IQR": None,
            "p90": None, "p95": None, "p99": None,
        }
    bucket_rows = []
    for lbl, lo, hi in buckets:
        cnt = bucket_counts[lbl]
        pct_val = (100.0 * cnt / n) if n > 0 else 0.0
        upper = "inf" if hi == INF else hi
        bucket_rows.append({
            "bucket_label":       lbl,
            "lower":              lo,
            "upper":              upper,
            "count":              cnt,
            "pct_of_non_missing": f"{pct_val:.2f}%",
            "_pct_value":         pct_val,
        })
    return {"stats": stats, "buckets": bucket_rows}


def block_for_cohort(records, field, buckets):
    vals = [r.get(field) for r in records]
    res = compute_signal(vals, buckets)
    # Invariant check: bucket_sum + n_missing == cohort_total
    bucket_sum = sum(b["count"] for b in res["buckets"])
    cohort_total = len(records)
    invariant_ok = (bucket_sum + res["stats"]["n_missing"] == cohort_total)
    return {
        "field_source": field,
        "summary_stats": res["stats"],
        "buckets":       [
            {k: v for k, v in b.items() if k != "_pct_value"}
            for b in res["buckets"]
        ],
        "invariant":     {
            "bucket_sum":   bucket_sum,
            "n_missing":    res["stats"]["n_missing"],
            "cohort_total": cohort_total,
            "ok":           invariant_ok,
        },
    }


# ---------------------------------------------------------------------------
# Side-study extras (forks, archived, fork, language, push-recency).
# Preserved so the summary is a superset of repo_characteristics.csv.
# ---------------------------------------------------------------------------

def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def percentile(data, p):
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def side_study_extras(records):
    n = len(records)
    if n == 0:
        return {"n": 0}
    forks = [int(r.get("forks") or 0) for r in records]
    n_archived = sum(1 for r in records if r.get("archived") is True)
    n_fork     = sum(1 for r in records if r.get("fork") is True)
    languages  = Counter((r.get("language") or "Unknown") for r in records)

    created_years = Counter()
    for r in records:
        dt = parse_iso(r.get("created_at"))
        if dt:
            created_years[dt.year] += 1
    pushed = {"active_last_6mo": 0, "active_last_12mo": 0,
              "older_than_12mo": 0, "unknown": 0}
    for r in records:
        dt = parse_iso(r.get("pushed_at"))
        if dt is None:
            pushed["unknown"] += 1
            continue
        days = (REFERENCE_DATE - dt).days
        if days <= 183:
            pushed["active_last_6mo"] += 1
        elif days <= 365:
            pushed["active_last_12mo"] += 1
        else:
            pushed["older_than_12mo"] += 1

    return {
        "forks": {
            "median": float(statistics.median(forks)),
            "mean":   float(statistics.mean(forks)),
            "p90":    percentile(forks, 90),
            "p99":    percentile(forks, 99),
        },
        "archived_pct":  100.0 * n_archived / n,
        "fork_pct":      100.0 * n_fork / n,
        "language_top15": languages.most_common(15),
        "created_years":  dict(sorted(created_years.items())),
        "pushed_at_recency": pushed,
        "reference_date_for_push": REFERENCE_DATE.date().isoformat(),
    }


# ---------------------------------------------------------------------------
# Load + reachability accounting
# ---------------------------------------------------------------------------

def load_records():
    if not INPUT.exists():
        sys.exit(f"ERROR: {INPUT} not found")
    records = []
    with INPUT.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def reachability_breakdown(records):
    n = len(records)
    n_repo_ok      = sum(1 for r in records if r.get("repo_status") == "ok")
    n_404          = sum(1 for r in records if r.get("repo_status") == "404_not_found")
    n_403          = sum(1 for r in records if r.get("repo_status") == "403_forbidden")
    # Repos that hit a 301 redirect AND succeeded after following:
    n_redirect_ok  = sum(1 for r in records
                         if r.get("repo_status") == "ok"
                         and r.get("full_name_resolved")
                         and r.get("full_name_resolved") != r.get("full_name"))
    n_other_repo   = n - n_repo_ok - n_404 - n_403

    # Contributors-missing under the hub convention: null OR zero
    n_contrib_null = sum(1 for r in records
                         if r.get("num_contributors_api_approx") is None)
    n_contrib_zero = sum(1 for r in records
                         if r.get("num_contributors_api_approx") == 0)
    n_contrib_missing = n_contrib_null + n_contrib_zero

    # Commits-missing: null only (hub convention; 0 counted as out-of-range)
    n_commits_null = sum(1 for r in records
                         if r.get("num_commits_default_branch") is None)
    n_commits_zero = sum(1 for r in records
                         if r.get("num_commits_default_branch") == 0)
    n_commits_missing_null_only = n_commits_null
    n_commits_missing_incl_zero = n_commits_null + n_commits_zero

    return {
        "total":                 n,
        "repo_status_ok":        n_repo_ok,
        "repo_status_404":       n_404,
        "repo_status_403":       n_403,
        "repo_status_redirect_ok": n_redirect_ok,
        "repo_status_other":     n_other_repo,
        "contributors_null":     n_contrib_null,
        "contributors_zero":     n_contrib_zero,
        "contributors_missing_hub_convention": n_contrib_missing,
        "commits_null":          n_commits_null,
        "commits_zero":          n_commits_zero,
        "commits_missing_null_only": n_commits_missing_null_only,
        "commits_missing_incl_zero": n_commits_missing_incl_zero,
    }


# ---------------------------------------------------------------------------
# Per-cohort summary
# ---------------------------------------------------------------------------

def build_cohort_block(records, label):
    return {
        "cohort":              label,
        "cohort_total_count":  len(records),
        "convention_note":     CONVENTION_NOTE,
        "stars":         block_for_cohort(records, "stars",                       STARS_BUCKETS),
        "contributors":  block_for_cohort(records, "num_contributors_api_approx", CONTRIB_BUCKETS),
        "commits":       block_for_cohort(records, "num_commits_default_branch",  COMMITS_BUCKETS),
        "side_study_extras": side_study_extras(records),
    }


# ---------------------------------------------------------------------------
# Hub side-by-side comparison printer
# ---------------------------------------------------------------------------

def comparison_table(overall_block, hub):
    h_stars   = hub["stars"]["summary_stats"]
    h_contrib = hub["contributors"]["summary_stats"]
    h_commits = hub["commits"]["summary_stats"]
    o_stars   = overall_block["stars"]["summary_stats"]
    o_contrib = overall_block["contributors"]["summary_stats"]
    o_commits = overall_block["commits"]["summary_stats"]

    def bucket_pct(blk, label):
        for b in blk["buckets"]:
            if b["bucket_label"] == label:
                return b["pct_of_non_missing"]
        return "n/a"

    # Headline buckets per signal
    h_zero_star = bucket_pct(hub["stars"], "0")
    o_zero_star = bucket_pct(overall_block["stars"], "0")
    h_solo      = bucket_pct(hub["contributors"], "1")
    o_solo      = bucket_pct(overall_block["contributors"], "1")
    h_commits_2_10 = bucket_pct(hub["commits"], "2-10")
    o_commits_2_10 = bucket_pct(overall_block["commits"], "2-10")

    print()
    print("=" * 90)
    print("HUB-vs-OFFHUB comparison")
    print(f"  HUB    cohort = analysis_set ({hub['analysis_set_total_count']:,} repos)")
    print(f"  OFFHUB cohort = all off-hub  ({overall_block['cohort_total_count']:,} repos)")
    print("=" * 90)
    print(f"{'signal':<14}{'metric':<22}{'HUB':>14}{'OFFHUB':>14}{'  delta':>10}")
    print("-" * 90)

    def row(sig, metric, h, o, fmt="{:>14.2f}"):
        try:
            d = o - h
            print(f"{sig:<14}{metric:<22}{fmt.format(h):>14}{fmt.format(o):>14}{d:>+10.2f}")
        except Exception:
            print(f"{sig:<14}{metric:<22}{str(h):>14}{str(o):>14}{'?':>10}")

    row("stars",        "median",      h_stars["median"],   o_stars["median"])
    row("stars",        "mean",        h_stars["mean"],     o_stars["mean"])
    row("stars",        "p90",         h_stars["p90"],      o_stars["p90"])
    print(f"{'stars':<14}{'% in 0-bucket':<22}{h_zero_star:>14}{o_zero_star:>14}{'':>10}")
    row("contributors", "median",      h_contrib["median"], o_contrib["median"])
    row("contributors", "mean",        h_contrib["mean"],   o_contrib["mean"])
    row("contributors", "p90",         h_contrib["p90"],    o_contrib["p90"])
    print(f"{'contributors':<14}{'% solo (=1)':<22}{h_solo:>14}{o_solo:>14}{'':>10}")
    row("commits",      "median",      h_commits["median"], o_commits["median"])
    row("commits",      "mean",        h_commits["mean"],   o_commits["mean"])
    row("commits",      "p90",         h_commits["p90"],    o_commits["p90"])
    print(f"{'commits':<14}{'% in 2-10 bucket':<22}{h_commits_2_10:>14}{o_commits_2_10:>14}{'':>10}")

    print()
    print(f"Coverage:")
    print(f"  HUB    n_stars / n_contributors / n_commits  = "
          f"{h_stars['n']:,} / {h_contrib['n']:,} / {h_commits['n']:,}")
    print(f"  OFFHUB n_stars / n_contributors / n_commits  = "
          f"{o_stars['n']:,} / {o_contrib['n']:,} / {o_commits['n']:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    records = load_records()
    if not records:
        sys.exit("ERROR: no records loaded")
    print(f"Loaded {len(records):,} records from {INPUT.name}")

    hub = json.loads(HUB_SUMMARY.read_text())

    # Reachability accounting
    reach = reachability_breakdown(records)

    # OVERALL cohort
    overall = build_cohort_block(records, "overall")

    # Per-mode cohorts
    by_mode = {m: [r for r in records if r.get("mode") == m] for m in MODE_ORDER}
    other_modes = {r.get("mode") for r in records} - set(MODE_ORDER) - {None}
    if "unclassified" in other_modes:
        by_mode["unclassified"] = [r for r in records if r.get("mode") == "unclassified"]
    per_mode_blocks = {m: build_cohort_block(by_mode[m], m) for m in by_mode}

    # ---- Invariant prints ----
    print()
    print("=" * 78)
    print(f"Invariant check: bucket_sum + n_missing == cohort_total per signal")
    print("=" * 78)
    def print_invariants(label, block):
        print(f"  cohort = {label}  (n={block['cohort_total_count']:,})")
        for sig in ("stars", "contributors", "commits"):
            inv = block[sig]["invariant"]
            verdict = "OK" if inv["ok"] else "FAIL"
            print(f"    {sig:<14} bucket_sum={inv['bucket_sum']:>5,} + "
                  f"n_missing={inv['n_missing']:>4,} = "
                  f"{inv['bucket_sum']+inv['n_missing']:>5,}  "
                  f"vs cohort_total={inv['cohort_total']:>5,}  [{verdict}]")
    print_invariants("overall", overall)
    for m in by_mode:
        print_invariants(m, per_mode_blocks[m])

    # ---- Reachability print ----
    print()
    print("=" * 78)
    print("Reachability + missing-value breakdown")
    print("=" * 78)
    print(f"  total records:            {reach['total']:,}")
    print(f"  /repos status = ok:       {reach['repo_status_ok']:,}")
    print(f"      of which 301-redirect-followed: {reach['repo_status_redirect_ok']:,}")
    print(f"  /repos status = 404:      {reach['repo_status_404']:,}")
    print(f"  /repos status = 403:      {reach['repo_status_403']:,}")
    print(f"  /repos status = other:    {reach['repo_status_other']:,}")
    print()
    print(f"  contributors null:        {reach['contributors_null']:,}")
    print(f"  contributors == 0:        {reach['contributors_zero']:,}")
    print(f"  contributors-missing (hub convention: null OR 0): "
          f"{reach['contributors_missing_hub_convention']:,}")
    print()
    print(f"  commits null:             {reach['commits_null']:,}")
    print(f"  commits == 0:             {reach['commits_zero']:,}")
    print(f"  commits-missing (hub convention: null only): "
          f"{reach['commits_missing_null_only']:,}")
    print(f"  commits-missing (incl. 0): {reach['commits_missing_incl_zero']:,}")

    # ---- Build final JSON output ----
    out = {
        "schema":              "offhub_rq1_summary/v1",
        "computed_at_utc":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file":          str(INPUT),
        "hub_template":        str(HUB_SUMMARY),
        "naming_note":         "paper RQ1 = server-side rq0; structure mirrors rq0_summary.json",
        "convention_note":     CONVENTION_NOTE,
        "bucket_definitions": {
            "stars":         [{"label": l, "lower": lo,
                               "upper": ("inf" if hi == INF else hi)}
                              for l, lo, hi in STARS_BUCKETS],
            "contributors":  [{"label": l, "lower": lo,
                               "upper": ("inf" if hi == INF else hi)}
                              for l, lo, hi in CONTRIB_BUCKETS],
            "commits":       [{"label": l, "lower": lo,
                               "upper": ("inf" if hi == INF else hi)}
                              for l, lo, hi in COMMITS_BUCKETS],
        },
        "reachability_summary": reach,
        "overall":              overall,
        "by_mode":              per_mode_blocks,
    }
    OUTPUT.write_text(json.dumps(out, indent=2))
    print()
    print(f"Wrote: {OUTPUT}")

    # ---- Hub-vs-offhub comparison ----
    comparison_table(overall, hub)


if __name__ == "__main__":
    main()

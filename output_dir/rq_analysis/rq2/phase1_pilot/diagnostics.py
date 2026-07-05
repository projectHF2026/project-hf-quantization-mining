"""
Phase 1 pilot diagnostics — reads the three JSONL outputs from mine_pilot.py
and produces pilot_diagnostics.md plus two optional plots.
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path("/scratch/oldhome/user/projects/JAW/scripts/icpc-approch")
PILOT_DIR = BASE / "output_dir/rq_analysis/rq2/phase1_pilot"
PILOT_CSV = PILOT_DIR / "pilot_repos.csv"
CLONE_LOG = PILOT_DIR / "pilot_clone_log.jsonl"
LAYER1_JSONL = PILOT_DIR / "results/layer1_method_signals.jsonl"
LAYER2_JSONL = PILOT_DIR / "results/layer2_model_signals.jsonl"
DIAGNOSTICS_MD = PILOT_DIR / "pilot_diagnostics.md"
PLOT_A_PNG = PILOT_DIR / "layer1_monthly_method_signals.png"
PLOT_B_PNG = PILOT_DIR / "layer2_adoption_lag_distribution.png"

STRATA = ["GGUF", "BitsAndBytes", "GPTQ", "AWQ", "Other"]
METHODS = ["GGUF", "BitsAndBytes", "GPTQ", "AWQ", "Other"]

# Technical-gate targets
TARGET_CLONE_RATE = 0.85
TARGET_MINING_RATE = 0.80
TARGET_PICKAXE_EXACT_OVERALL = 0.70
TARGET_SPOT_CHECK_PASSES = 18

# Full-run scale-up factor (analysis set has 20,704 repos with primary method)
SCALE_FACTOR = 20704 / 100


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def fmt_pct(n: int, d: int) -> str:
    return f"{(100*n/d):.1f}%" if d else "n/a"


def main() -> None:
    pilot = pd.read_csv(PILOT_CSV)
    clones = load_jsonl(CLONE_LOG)
    layer1 = load_jsonl(LAYER1_JSONL)
    layer2 = load_jsonl(LAYER2_JSONL)

    print(f"Loaded: {len(clones)} clone records, {len(layer1)} L1, {len(layer2)} L2",
          flush=True)

    # ----- Clone success per stratum -----
    clone_by_stratum: dict[str, Counter] = defaultdict(Counter)
    clone_seconds_total = 0.0
    clone_size_total = 0
    for c in clones:
        clone_by_stratum[c["stratum"]][c["clone_status"]] += 1
        clone_seconds_total += c.get("clone_seconds") or 0
        clone_size_total += c.get("clone_size_bytes") or 0

    # ----- Layer 1 mining status per stratum -----
    l1_by_stratum_status: dict[str, Counter] = defaultdict(Counter)
    l1_signal_method_by_stratum: dict[str, Counter] = defaultdict(Counter)
    for r in layer1:
        l1_by_stratum_status[r["stratum"]][r["mining_status"]] += 1
        l1_signal_method_by_stratum[r["stratum"]][r.get("first_method_signal_method", "")] += 1

    # ----- Layer 2 method distribution per stratum -----
    l2_by_stratum_method: dict[str, Counter] = defaultdict(Counter)
    for r in layer2:
        l2_by_stratum_method[r["stratum"]][r.get("first_model_signal_method", "")] += 1

    l2_overall_method = Counter(r.get("first_model_signal_method", "") for r in layer2)
    pickaxe_pct = 100 * l2_overall_method.get("pickaxe_exact", 0) / max(1, len(layer2))

    # ----- Runtime / disk projection -----
    layer1_seconds = 0.0  # not directly tracked; approximate from log if needed
    layer2_seconds = 0.0
    proj_clone_hours = clone_seconds_total * SCALE_FACTOR / 3600
    proj_disk_gb = clone_size_total * SCALE_FACTOR / (1024**3)

    # ----- Spot-check: 4 pickaxe_exact pairs per stratum from Layer 2 -----
    spot_check_rng = random.Random(42)
    spot_pairs: list[dict] = []
    for s in STRATA:
        candidates = [r for r in layer2 if r["stratum"] == s
                      and r.get("first_model_signal_method") == "pickaxe_exact"
                      and r.get("first_model_signal_commit")]
        if not candidates:
            continue
        chosen = spot_check_rng.sample(candidates, k=min(4, len(candidates)))
        for r in chosen:
            spot_pairs.append(r)

    # Verify each: check that the model_id is in the file content at the commit
    def verify_pair(rec) -> tuple[bool, str]:
        repo_id = rec["repo_id"]
        commit = rec["first_model_signal_commit"]
        file_path = rec["matched_file"]
        model_id = rec["model_id"]
        safe = repo_id.replace("/", "__")
        clone = PILOT_DIR / "clones" / safe
        if not clone.exists():
            return False, "clone missing"
        try:
            proc = subprocess.run(
                ["git", "-C", str(clone), "show", f"{commit}:{file_path}"],
                capture_output=True, text=True, timeout=30, errors="replace",
            )
            if proc.returncode != 0:
                return False, f"git show rc={proc.returncode}"
            if model_id in proc.stdout:
                return True, "ok"
            return False, "model_id not in file content"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    spot_results = []
    for rec in spot_pairs:
        ok, note = verify_pair(rec)
        spot_results.append((rec, ok, note))
    spot_pass = sum(1 for _, ok, _ in spot_results if ok)

    # ----- H1: GGUF temporal trend (monthly) -----
    monthly_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in layer1:
        if r.get("first_method_signal_method") != "pickaxe_exact":
            continue
        date_str = r.get("first_method_signal_date", "")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            continue
        if dt.year < 2022 or (dt.year == 2026 and dt.month > 6):
            continue
        month = dt.strftime("%Y-%m")
        monthly_counts[month][r["method"]] += 1

    months_sorted = sorted(monthly_counts.keys())

    # Yearly aggregates
    yearly_method: dict[int, Counter] = defaultdict(Counter)
    for month_key, methods_counter in monthly_counts.items():
        year = int(month_key[:4])
        for m, n in methods_counter.items():
            yearly_method[year][m] += n
    yearly_totals = {y: sum(yearly_method[y].values()) for y in yearly_method}
    gguf_yearly = {y: yearly_method[y].get("GGUF", 0) for y in sorted(yearly_method)}
    gguf_share = {y: (gguf_yearly[y] / yearly_totals[y]) if yearly_totals[y] else 0
                  for y in sorted(yearly_method)}

    # Slope of GGUF monthly counts 2023-2026 (simple ordinary least squares)
    gguf_monthly = [monthly_counts[m].get("GGUF", 0) for m in months_sorted
                    if m >= "2023-01"]
    if len(gguf_monthly) >= 3:
        xs = list(range(len(gguf_monthly)))
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(gguf_monthly) / n
        num = sum((xs[i] - mean_x) * (gguf_monthly[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        gguf_slope = num / den if den else 0.0
    else:
        gguf_slope = 0.0

    # H1 verdict logic
    has_data = len(gguf_yearly) >= 2
    gguf_late_total = sum(gguf_yearly.get(y, 0) for y in (2025, 2026))
    gguf_early_total = sum(gguf_yearly.get(y, 0) for y in (2023,))
    gguf_late_share = (gguf_share.get(2025, 0) + gguf_share.get(2026, 0)) / 2
    gguf_early_share = gguf_share.get(2023, 0)
    n_gguf_total = sum(gguf_yearly.values())

    h1_confirmed = (
        has_data
        and gguf_late_total > gguf_early_total
        and gguf_late_share > gguf_early_share
        and gguf_slope > 0
    )
    h1_inconclusive = n_gguf_total < 15
    if h1_inconclusive and not h1_confirmed:
        h1_verdict = "INCONCLUSIVE"
    elif h1_confirmed:
        h1_verdict = "CONFIRMED"
    elif n_gguf_total > 0 and (gguf_late_total <= gguf_early_total or gguf_slope <= 0):
        h1_verdict = "REJECTED"
    else:
        h1_verdict = "INCONCLUSIVE"

    # ----- H2: BitsAndBytes trajectory -----
    bnb_yearly = {y: yearly_method[y].get("BitsAndBytes", 0)
                  for y in sorted(yearly_method)}
    bnb_share = {y: (bnb_yearly[y] / yearly_totals[y]) if yearly_totals[y] else 0
                 for y in sorted(yearly_method)}
    bnb_late_total = sum(bnb_yearly.get(y, 0) for y in (2025, 2026))
    bnb_early_total = sum(bnb_yearly.get(y, 0) for y in (2023, 2024))
    bnb_late_share = (bnb_share.get(2025, 0) + bnb_share.get(2026, 0)) / 2
    bnb_early_share = (bnb_share.get(2023, 0) + bnb_share.get(2024, 0)) / 2
    n_bnb_total = sum(bnb_yearly.values())

    if n_bnb_total < 10:
        h2_verdict = "c"
        h2_reason = ("pilot N too small to distinguish declining adoption "
                     "from stable-but-diluted")
    elif bnb_late_total < 0.7 * bnb_early_total and bnb_late_share <= bnb_early_share:
        h2_verdict = "a"
        h2_reason = ("monthly BnB first-signals in 2025-2026 are substantially "
                     "lower than 2023-2024")
    elif bnb_late_share < bnb_early_share and bnb_late_total >= 0.7 * bnb_early_total:
        h2_verdict = "b"
        h2_reason = ("BnB monthly counts roughly stable, but its share shrinks "
                     "while GGUF grows")
    else:
        h2_verdict = "c"
        h2_reason = "pattern is mixed; pilot too small to interpret cleanly"

    # ----- H3: adoption lag distributions -----
    lag_by_method: dict[str, list[float]] = defaultdict(list)
    for r in layer2:
        if r.get("first_model_signal_method") != "pickaxe_exact":
            continue
        lag = r.get("adoption_lag_days")
        if lag is None:
            continue
        m = r.get("primary_method", "")
        if m in {"GGUF", "BitsAndBytes", "GPTQ", "AWQ"}:
            lag_by_method[m].append(lag)
        else:
            lag_by_method["Other"].append(lag)

    h3_table = []
    for m in METHODS:
        lags = lag_by_method.get(m, [])
        if not lags:
            h3_table.append({"method": m, "n": 0, "median": None,
                             "p25": None, "p75": None})
            continue
        srt = sorted(lags)
        def pctile(p):
            if not srt: return None
            idx = max(0, min(len(srt)-1, int(round(p * (len(srt)-1)))))
            return srt[idx]
        h3_table.append({
            "method": m,
            "n": len(srt),
            "median": statistics.median(srt),
            "p25": pctile(0.25),
            "p75": pctile(0.75),
        })

    # ----- Technical-gate check -----
    clone_success = sum(1 for c in clones if c["clone_status"] == "success")
    clone_success_rate = clone_success / max(1, len(clones))
    per_stratum_clone_pass = all(
        clone_by_stratum[s].get("success", 0) / max(1, sum(clone_by_stratum[s].values()))
        >= TARGET_CLONE_RATE
        for s in STRATA if sum(clone_by_stratum[s].values()) > 0
    )

    l1_success_total = sum(1 for r in layer1 if r["mining_status"] == "success")
    l1_per_repo_count = len(set(r["repo_id"] for r in layer1))
    per_stratum_l1_pass = all(
        sum(l1_by_stratum_status[s].get(k, 0) for k in ("success",))
        / max(1, sum(l1_by_stratum_status[s].values())) >= TARGET_MINING_RATE
        for s in STRATA if sum(l1_by_stratum_status[s].values()) > 0
    )
    pickaxe_pass = pickaxe_pct >= TARGET_PICKAXE_EXACT_OVERALL * 100

    spot_check_pass = spot_pass >= TARGET_SPOT_CHECK_PASSES
    runtime_pass = proj_clone_hours <= 24

    tech_pass = (per_stratum_clone_pass and per_stratum_l1_pass and
                 pickaxe_pass and spot_check_pass and runtime_pass)

    # ----- Overall verdict -----
    if not tech_pass:
        overall = "c"
        overall_reason = "technical gate failed"
    elif h1_verdict == "REJECTED":
        overall = "c"
        overall_reason = "H1 rejected — Phase 0 cohort signal does not match git-history adoption timing"
    elif h1_verdict == "CONFIRMED" and h2_verdict in ("a", "b"):
        overall = "a"
        overall_reason = "tech ok; H1 confirmed; H2 interpretable"
    elif h1_verdict == "INCONCLUSIVE" and tech_pass:
        overall = "b"
        overall_reason = "tech ok; H1 directionally right but pilot too small for confident verdict"
    else:
        overall = "b"
        overall_reason = "mixed signal — full run recommended"

    # =============================================================
    # Build markdown
    # =============================================================
    L: list[str] = []
    L.append("# RQ2 Phase 1 Pilot - Two-Layer Temporal Mining")
    L.append("")
    L.append("> **Scope note:** Phase 1 mines git histories to test whether")
    L.append("> the cohort signal observed in Phase 0 reflects actual")
    L.append("> temporal adoption patterns. First-signal dates are lower")
    L.append("> bounds (git history can be rebased), and dates are restricted")
    L.append("> to the default branch. The pilot tests on 100 stratified")
    L.append("> repos before any decision to scale to 20,704.")
    L.append("")
    L.append("## Inputs used")
    L.append("")
    for p in (PILOT_CSV, CLONE_LOG, LAYER1_JSONL, LAYER2_JSONL):
        size = p.stat().st_size if p.exists() else 0
        L.append(f"- `{p}` ({size:,} bytes)")
    L.append("")

    L.append("## Technical feasibility")
    L.append("")
    L.append("### Clone success rate per stratum")
    L.append("")
    L.append("| stratum | success | 404 | timeout | auth_required | other_error | success rate |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in STRATA:
        c = clone_by_stratum[s]
        tot = sum(c.values())
        L.append(f"| {s} | {c.get('success',0)} | {c.get('404',0)} | "
                 f"{c.get('timeout',0)} | {c.get('auth_required',0)} | "
                 f"{c.get('other_error',0)} | {fmt_pct(c.get('success',0), tot)} |")
    L.append(f"| **Overall** | **{clone_success}** | "
             f"{sum(c.get('404',0) for c in clone_by_stratum.values())} | "
             f"{sum(c.get('timeout',0) for c in clone_by_stratum.values())} | "
             f"{sum(c.get('auth_required',0) for c in clone_by_stratum.values())} | "
             f"{sum(c.get('other_error',0) for c in clone_by_stratum.values())} | "
             f"**{fmt_pct(clone_success, len(clones))}** |")
    L.append("")
    L.append(f"Target: ≥{int(TARGET_CLONE_RATE*100)}% per stratum. "
             f"{'**PASS**' if per_stratum_clone_pass else '**FAIL**'}.")
    L.append("")

    L.append("### Layer 1 mining success rate per stratum")
    L.append("")
    L.append("| stratum | success | other_error | success rate |")
    L.append("|---|---:|---:|---:|")
    for s in STRATA:
        c = l1_by_stratum_status[s]
        tot = sum(c.values())
        L.append(f"| {s} | {c.get('success',0)} | "
                 f"{c.get('other_error',0)} | {fmt_pct(c.get('success',0), tot)} |")
    L.append(f"")
    L.append(f"Target: ≥{int(TARGET_MINING_RATE*100)}% per stratum. "
             f"{'**PASS**' if per_stratum_l1_pass else '**FAIL**'}.")
    L.append("")

    L.append("### Layer 2 first_model_signal_method distribution")
    L.append("")
    L.append("| stratum | pickaxe_exact | head_fallback | not_found_in_history | pickaxe_exact % |")
    L.append("|---|---:|---:|---:|---:|")
    for s in STRATA:
        c = l2_by_stratum_method[s]
        tot = sum(c.values())
        L.append(f"| {s} | {c.get('pickaxe_exact',0)} | "
                 f"{c.get('head_fallback',0)} | {c.get('not_found_in_history',0)} | "
                 f"{fmt_pct(c.get('pickaxe_exact',0), tot)} |")
    L.append(f"| **Overall** | **{l2_overall_method.get('pickaxe_exact',0)}** | "
             f"{l2_overall_method.get('head_fallback',0)} | "
             f"{l2_overall_method.get('not_found_in_history',0)} | "
             f"**{pickaxe_pct:.1f}%** |")
    L.append("")
    L.append(f"Target: pickaxe_exact ≥{int(TARGET_PICKAXE_EXACT_OVERALL*100)}% overall. "
             f"{'**PASS**' if pickaxe_pass else '**FAIL**'}.")
    L.append("")

    L.append("### Runtime and disk projection")
    L.append("")
    L.append(f"- Total pilot clone seconds: {clone_seconds_total:.0f}s "
             f"({clone_seconds_total/60:.1f} min)")
    L.append(f"- Mean clone size: "
             f"{clone_size_total/max(1,clone_success)/(1024*1024):.1f} MB")
    L.append(f"- Projected full-run clone wall clock (× {SCALE_FACTOR:.1f}): "
             f"**{proj_clone_hours:.1f} hours**")
    L.append(f"- Projected full-run disk: **{proj_disk_gb:.1f} GB**")
    L.append(f"")
    L.append(f"Target: full-run wall clock ≤24h. "
             f"{'**PASS**' if runtime_pass else '**FAIL**'}.")
    L.append("")

    L.append("### Spot-check verification")
    L.append("")
    L.append(f"Sampled 4 `pickaxe_exact` pairs per stratum from Layer 2 (seed=42), "
             f"verified that the model_id appears in `git show "
             f"<commit>:<matched_file>`. Result: **{spot_pass}/{len(spot_results)} passed** "
             f"(target ≥{TARGET_SPOT_CHECK_PASSES}/20).")
    L.append("")
    L.append("| # | repo | model_id | file | commit | verify |")
    L.append("|---|---|---|---|---|---|")
    for i, (rec, ok, note) in enumerate(spot_results, 1):
        ok_str = "✓" if ok else f"✗ ({note})"
        L.append(f"| {i} | `{rec['repo_id']}` | `{rec['model_id'][:40]}` | "
                 f"`{rec['matched_file'][:40]}` | `{rec['first_model_signal_commit'][:8]}` | {ok_str} |")
    L.append("")

    # H1
    L.append("## H1 — Confirmatory: GGUF temporal trend")
    L.append("")
    L.append("Monthly first-method-signal counts (Layer 1, `pickaxe_exact` only):")
    L.append("")
    L.append("| year | GGUF | BitsAndBytes | GPTQ | AWQ | Other | total | GGUF share |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for y in sorted(yearly_method):
        ym = yearly_method[y]
        tot = yearly_totals[y]
        L.append(f"| {y} | {ym.get('GGUF',0)} | {ym.get('BitsAndBytes',0)} | "
                 f"{ym.get('GPTQ',0)} | {ym.get('AWQ',0)} | {ym.get('Other',0)} | "
                 f"{tot} | {fmt_pct(ym.get('GGUF',0), tot)} |")
    L.append("")
    L.append(f"- GGUF total across pilot: {n_gguf_total}")
    L.append(f"- GGUF 2025+2026 count vs 2023 count: {gguf_late_total} vs {gguf_early_total}")
    L.append(f"- GGUF share late (avg of 2025+2026) vs early (2023): "
             f"{gguf_late_share*100:.1f}% vs {gguf_early_share*100:.1f}%")
    L.append(f"- GGUF monthly OLS slope, 2023+ window: {gguf_slope:+.3f} signals/month")
    L.append("")
    L.append("**ASCII monthly trajectory (GGUF first-signal counts):**")
    L.append("")
    L.append("```")
    if months_sorted:
        max_v = max(monthly_counts[m].get("GGUF", 0) for m in months_sorted)
        for m in months_sorted:
            v = monthly_counts[m].get("GGUF", 0)
            bar = "█" * v if max_v <= 30 else "█" * max(1, int(30 * v / max_v)) if v else ""
            L.append(f"  {m}  {v:>3}  {bar}")
    L.append("```")
    L.append("")
    L.append(f"**H1 verdict: {h1_verdict}**")
    L.append("")

    # H2
    L.append("## H2 — Mechanism: BitsAndBytes trajectory")
    L.append("")
    L.append(f"- BnB 2025-2026 count vs 2023-2024 count: {bnb_late_total} vs {bnb_early_total}")
    L.append(f"- BnB share late (avg 2025+2026) vs early (avg 2023+2024): "
             f"{bnb_late_share*100:.1f}% vs {bnb_early_share*100:.1f}%")
    L.append("")
    h2_label = {"a": "(a) DECLINING ADOPTION",
                "b": "(b) STABLE-BUT-DILUTED",
                "c": "(c) INDETERMINATE"}[h2_verdict]
    L.append(f"**H2 verdict: {h2_label}** — {h2_reason}.")
    L.append("")

    # H3
    L.append("## H3 — Supporting: adoption lag distributions")
    L.append("")
    L.append("From Layer 2 `pickaxe_exact` records with valid `hf_createdAt`:")
    L.append("")
    L.append("| primary_method | n | median (days) | p25 | p75 |")
    L.append("|---|---:|---:|---:|---:|")
    for row in h3_table:
        med = "n/a" if row["median"] is None else f"{row['median']:.1f}"
        p25 = "n/a" if row["p25"] is None else f"{row['p25']:.1f}"
        p75 = "n/a" if row["p75"] is None else f"{row['p75']:.1f}"
        L.append(f"| {row['method']} | {row['n']} | {med} | {p25} | {p75} |")
    L.append("")
    # Distinctness check: are median ranges overlapping?
    medians = [row["median"] for row in h3_table if row["median"] is not None]
    if len(medians) >= 2:
        med_range = max(medians) - min(medians)
        L.append(f"Spread of medians across methods: {med_range:.1f} days "
                 "(visibly distinct ≈ >100 days for short-lag adoption windows).")
    L.append("")
    L.append("*H3 is supporting only — does not gate the verdict.*")
    L.append("")

    # Overall verdict
    L.append("## Overall verdict")
    L.append("")
    overall_label = {"a": "(a) PROCEED TO FULL RUN",
                     "b": "(b) PROCEED WITH CAUTION",
                     "c": "(c) STOP AND RECONSIDER"}[overall]
    L.append(f"**{overall_label}** — {overall_reason}.")
    L.append("")
    L.append("Gate summary:")
    L.append("")
    L.append(f"- Clone success per stratum ≥85%: "
             f"{'PASS' if per_stratum_clone_pass else 'FAIL'}")
    L.append(f"- Layer 1 mining success per stratum ≥80%: "
             f"{'PASS' if per_stratum_l1_pass else 'FAIL'}")
    L.append(f"- Layer 2 pickaxe_exact ≥70% overall: "
             f"{'PASS' if pickaxe_pass else 'FAIL'} ({pickaxe_pct:.1f}%)")
    L.append(f"- Spot-check ≥18/20: "
             f"{'PASS' if spot_check_pass else 'FAIL'} ({spot_pass}/{len(spot_results)})")
    L.append(f"- Full-run wall clock ≤24h: "
             f"{'PASS' if runtime_pass else 'FAIL'} ({proj_clone_hours:.1f}h projected)")
    L.append(f"- H1 (GGUF temporal trend): {h1_verdict}")
    L.append(f"- H2 (BnB mechanism): {h2_verdict}")
    L.append(f"- H3 (lag distributions): supporting only")
    L.append("")

    DIAGNOSTICS_MD.write_text("\n".join(L) + "\n")
    print(f"Wrote {DIAGNOSTICS_MD}", flush=True)

    # =============================================================
    # Optional plots
    # =============================================================
    plot_made_a = plot_made_b = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if months_sorted:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            x_labels = months_sorted
            x = list(range(len(months_sorted)))
            bottom = [0.0] * len(months_sorted)
            colors = {"GGUF": "#3a6f9a", "BitsAndBytes": "#c97954",
                      "GPTQ": "#6a9a3a", "AWQ": "#9a5a3a", "Other": "#8c8c8c"}
            for m in METHODS:
                vals = [monthly_counts[mo].get(m, 0) for mo in months_sorted]
                ax.bar(x, vals, bottom=bottom, color=colors[m], label=m,
                       edgecolor="white", linewidth=0.4)
                bottom = [bottom[i] + vals[i] for i in range(len(vals))]
            tick_step = max(1, len(months_sorted) // 12)
            ax.set_xticks(x[::tick_step])
            ax.set_xticklabels([months_sorted[i] for i in x[::tick_step]],
                               rotation=45, ha="right")
            ax.set_xlabel("Month (first-method-signal date)")
            ax.set_ylabel("New first-method signals (pickaxe_exact)")
            ax.set_title("Method first-signal counts by month (pilot)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.tight_layout()
            fig.savefig(PLOT_A_PNG, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plot_made_a = True

        # Plot B: lag boxplot
        boxes = []
        labels = []
        for m in METHODS:
            lags = lag_by_method.get(m, [])
            if lags:
                boxes.append(lags)
                labels.append(f"{m}\n(n={len(lags)})")
        if boxes:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(boxes, labels=labels, showfliers=True,
                       boxprops=dict(linewidth=0.8),
                       medianprops=dict(linewidth=1.2, color="#3a6f9a"),
                       whiskerprops=dict(linewidth=0.8))
            ax.set_ylabel("Adoption lag (days)")
            ax.set_title("Adoption lag distribution by method (pilot sample)")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.tight_layout()
            fig.savefig(PLOT_B_PNG, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plot_made_b = True
    except Exception as e:
        print(f"plot skipped: {type(e).__name__}: {e}", flush=True)

    # =============================================================
    # Final stdout summary
    # =============================================================
    print()
    print("=" * 70)
    print(f"L1 records  : {len(layer1):,}  ({LAYER1_JSONL})")
    print(f"L2 records  : {len(layer2):,}  ({LAYER2_JSONL})")
    print(f"Diagnostics : {DIAGNOSTICS_MD}")
    if plot_made_a:
        print(f"Plot A      : {PLOT_A_PNG}")
    if plot_made_b:
        print(f"Plot B      : {PLOT_B_PNG}")
    print()
    print(f"Technical gate: {'PASS' if tech_pass else 'FAIL'}")
    print(f"H1 verdict    : {h1_verdict}")
    print(f"H2 verdict    : ({h2_verdict})")
    print(f"H3 lag medians:")
    for row in h3_table:
        if row["median"] is not None:
            print(f"  {row['method']:<14}  n={row['n']:>4}  median={row['median']:>8.1f} d")
        else:
            print(f"  {row['method']:<14}  n={row['n']:>4}  median=n/a")
    print(f"OVERALL VERDICT: ({overall}) — {overall_reason}")


if __name__ == "__main__":
    main()

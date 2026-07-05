"""
Exploratory: how many analysis-set repos satisfy a stricter engineered-project
conjunction?

Criteria (all three must hold; missing/null fails conservatively):
  stars >= 10
  num_commits_default_branch >= 100
  num_contributors_api_approx >= 10

Reads:
  output_dir/rq_analysis/shared/results/analysis_set_repo_details.jsonl

Writes:
  ./strict_threshold_repos.txt
  ./strict_threshold_summary.json
  ./strict_threshold_diagnostic.txt

Exploratory only — the canonical RQ1 engineered subsets (stars1, stars5,
stars10) remain the authoritative results in the paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path("/scratch/oldhome/user/projects/JAW/scripts/icpc-approch")
INPUT_JSONL = (
    BASE_DIR / "output_dir/rq_analysis/shared/results/analysis_set_repo_details.jsonl"
)
OUT_DIR = (
    BASE_DIR / "output_dir/rq_analysis/exploratory/strict_engineered_threshold"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPOS = OUT_DIR / "strict_threshold_repos.txt"
OUT_SUMMARY = OUT_DIR / "strict_threshold_summary.json"
OUT_DIAGNOSTIC = OUT_DIR / "strict_threshold_diagnostic.txt"

THRESHOLDS = {
    "stars_min": 10,
    "commits_min": 100,
    "contributors_min": 10,
}


def passes_int(value, threshold: int) -> bool:
    """True iff value is a non-null integer >= threshold."""
    if value is None:
        return False
    try:
        return int(value) >= threshold
    except (TypeError, ValueError):
        return False


def fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "0.00%"
    return f"{100 * num / den:.2f}%"


def main() -> None:
    if not INPUT_JSONL.exists():
        sys.stderr.write(f"ERROR: missing input {INPUT_JSONL}\n")
        sys.exit(2)

    total = 0
    a_repos: set[str] = set()  # stars >= 10
    b_repos: set[str] = set()  # commits >= 100
    c_repos: set[str] = set()  # contributors >= 10

    with INPUT_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            repo = rec["repo"]
            if passes_int(rec.get("stars"), THRESHOLDS["stars_min"]):
                a_repos.add(repo)
            if passes_int(rec.get("num_commits_default_branch"), THRESHOLDS["commits_min"]):
                b_repos.add(repo)
            if passes_int(rec.get("num_contributors_api_approx"), THRESHOLDS["contributors_min"]):
                c_repos.add(repo)

    ab = a_repos & b_repos
    ac = a_repos & c_repos
    bc = b_repos & c_repos
    abc = a_repos & b_repos & c_repos

    counts = {
        "analysis_set_total": total,
        "stars_ge_10": len(a_repos),
        "commits_ge_100": len(b_repos),
        "contributors_ge_10": len(c_repos),
        "stars_AND_commits": len(ab),
        "stars_AND_contributors": len(ac),
        "commits_AND_contributors": len(bc),
        "stars_AND_commits_AND_contributors": len(abc),
    }
    pcts = {k: fmt_pct(v, total) for k, v in counts.items() if k != "analysis_set_total"}
    pcts["analysis_set_total"] = "100.00%"

    # ----- write outputs -----
    with OUT_REPOS.open("w") as f:
        for repo in sorted(abc):
            f.write(repo + "\n")

    summary = {
        "description": (
            "Exploratory strict engineered-project threshold: how many "
            "analysis-set repos satisfy ALL of stars >= 10, commits >= 100, "
            "and contributors >= 10? Missing/null values for any signal "
            "fail that criterion (conservative exclusion)."
        ),
        "thresholds": THRESHOLDS,
        "input_file": str(INPUT_JSONL),
        "counts": counts,
        "percentages_of_analysis_set": pcts,
        "note": (
            "This is exploratory only. The canonical RQ1 engineered subsets "
            "(stars1, stars5, stars10) remain the authoritative results in "
            "the paper."
        ),
    }
    with OUT_SUMMARY.open("w") as f:
        json.dump(summary, f, indent=2)

    # human-readable
    lines = [
        "=" * 72,
        "STRICT ENGINEERED THRESHOLD — EXPLORATORY DIAGNOSTIC",
        "=" * 72,
        "",
        f"Input    : {INPUT_JSONL}",
        f"Total N  : {total:,} analysis-set repositories",
        "",
        "Thresholds (all three must hold; missing/null fails conservatively):",
        f"  stars                     >= {THRESHOLDS['stars_min']}",
        f"  num_commits_default_branch >= {THRESHOLDS['commits_min']}",
        f"  num_contributors_api_approx >= {THRESHOLDS['contributors_min']}",
        "",
        "Individual-criterion counts:",
        f"  stars        >= 10       : {len(a_repos):>6,}  ({fmt_pct(len(a_repos), total)})",
        f"  commits      >= 100      : {len(b_repos):>6,}  ({fmt_pct(len(b_repos), total)})",
        f"  contributors >= 10       : {len(c_repos):>6,}  ({fmt_pct(len(c_repos), total)})",
        "",
        "Pairwise-intersection counts:",
        f"  stars >= 10  AND commits >= 100    : {len(ab):>6,}  ({fmt_pct(len(ab), total)})",
        f"  stars >= 10  AND contributors >= 10: {len(ac):>6,}  ({fmt_pct(len(ac), total)})",
        f"  commits >= 100 AND contributors>=10: {len(bc):>6,}  ({fmt_pct(len(bc), total)})",
        "",
        "Three-way conjunction (the strict cohort):",
        f"  ALL three                : {len(abc):>6,}  ({fmt_pct(len(abc), total)})",
        "",
        "Note: exploratory only. The canonical RQ1 engineered subsets",
        "(stars1, stars5, stars10) remain the authoritative results in the",
        "paper.",
    ]
    diagnostic = "\n".join(lines) + "\n"
    OUT_DIAGNOSTIC.write_text(diagnostic)

    print(diagnostic)
    print(f"Wrote:")
    print(f"  {OUT_REPOS}  ({len(abc):,} repos)")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_DIAGNOSTIC}")


if __name__ == "__main__":
    main()

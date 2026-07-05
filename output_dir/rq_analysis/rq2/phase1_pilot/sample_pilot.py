"""
Phase 1 pilot sampler.

Stratified random sample of 100 repos from repo_frame.csv (20 per method
bucket: GGUF, BitsAndBytes, GPTQ, AWQ, Other). Seed = 42.

Writes:
  pilot_repos.csv  — sample with the columns needed downstream
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/scratch/oldhome/user/projects/JAW/scripts/icpc-approch")
FRAME_CSV = BASE_DIR / "output_dir/rq_analysis/rq2/phase0/repo_frame.csv"
OUT_CSV = BASE_DIR / "output_dir/rq_analysis/rq2/phase1_pilot/pilot_repos.csv"

PER_STRATUM = 20
STRATA = ["GGUF", "BitsAndBytes", "GPTQ", "AWQ", "Other"]
SEED = 42


def main() -> None:
    df = pd.read_csv(FRAME_CSV)
    rng = random.Random(SEED)
    picks: list[dict] = []
    for s in STRATA:
        pool = df[df["method_bucket"] == s].sort_values("repo_id").to_dict("records")
        if len(pool) < PER_STRATUM:
            print(f"WARNING: stratum {s} has {len(pool)} < {PER_STRATUM}; using all.")
            sample = pool
        else:
            sample = rng.sample(pool, PER_STRATUM)
            sample.sort(key=lambda r: r["repo_id"])
        for entry in sample:
            entry["stratum"] = s
        picks.extend(sample)

    out_cols = [
        "repo_id", "stratum", "primary_method", "method_bucket",
        "created_at_year", "created_at_date", "n_referenced_models",
        "stars",
    ]
    pd.DataFrame(picks)[out_cols].to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(picks)} rows)")
    print()
    print("Per-stratum sampled counts:")
    for s in STRATA:
        n = sum(1 for p in picks if p["stratum"] == s)
        pool_n = (df["method_bucket"] == s).sum()
        print(f"  {s:<14} {n:>3}  (pool: {pool_n:,})")


if __name__ == "__main__":
    main()

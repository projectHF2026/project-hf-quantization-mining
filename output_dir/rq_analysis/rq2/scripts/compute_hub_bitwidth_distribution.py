"""RQ2: bit-width distribution of adopted hub models (reproducible).

INPUT (read-only):
  output_dir/rq_analysis/shared/results/analysis_set_repo_details.jsonl
    — defines the 22,024 analysis-set repos; collect the union of every
      `models[*]` -> the 13,752 distinct ADOPTED hub model_ids.
  output_dir/quantized_filtered/quantized_models_all.jsonl
    — per-model metadata: quant_methods, tags, detection_signals, model_id.

RECOVERY (per model, priority a > b > c, first hit wins):
  (a) explicit categorical label found in `quant_methods` or `tags`,
      mapped via LABEL_TO_BIT below (covers 2bit/3bit/4bit/.../8bit,
      INT2..INT8, NF4, FP4, FP8, fbgemm-fp8, MXFP4/8, NVFP4, BNB-Nbit
      variants, BitsAndBytes_Nbit, GPTQ tags carrying a bit suffix).
  (b) `config.quantization_config.bits=N` parsed from `detection_signals`.
  (c) model_id substring/regex: GGUF Q-tags (Q2/Q3/Q4/Q5/Q6/Q7/Q8 and IQ2/IQ3/
      IQ4/IQ5/IQ6), -bnb-Nbit, -Nbit-, -Nbit_, nf4, fp4, fp8, fp16, int4, int8.

OUTPUT (two new files):
  output_dir/rq_analysis/rq2/results/hub_bitwidth_distribution.csv
    bit_width, models, pct
  output_dir/rq_analysis/rq2/results/hub_bitwidth_per_model.csv
    model_id, bit_width, primary_method, recovery_path, signals

After computing, the script prints the table and a cell-by-cell diff vs the
reference values from the paper (4-bit 4,386 / 64.75%, 8-bit 1,313 / 19.38%,
FP8 544 / 8.03%, 2-bit 232 / 3.42%, 6-bit 127 / 1.87%, 3-bit 93 / 1.37%,
5-bit 78 / 1.15%, FP16 1 / 0.01%). Any non-zero delta is flagged as FINDING.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/scratch/oldhome/user/projects/JAW/scripts/icpc-approch")
REPOS = ROOT / "output_dir/rq_analysis/shared/results/analysis_set_repo_details.jsonl"
MODELS = ROOT / "output_dir/quantized_filtered/quantized_models_all.jsonl"

OUT_DIR     = ROOT / "output_dir/rq_analysis/rq2/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIST    = OUT_DIR / "hub_bitwidth_distribution.csv"
OUT_PERMODEL = OUT_DIR / "hub_bitwidth_per_model.csv"


# ---------------------------------------------------------------------------
# Categorical-label → bit-width bucket
# ---------------------------------------------------------------------------
# Case-folded keys; we match against quant_methods entries and tag tokens.
# Order of priority within path (a) is determined by LABEL_PRIORITY below.

# Categorical labels (matched case-sensitively against quant_methods entries)
# mapped to an INTEGER bit-width or to the FP8 bucket. Tags are NOT scanned.
INT_BIT_FROM_LABEL = {
    "2bit": 2, "3bit": 3, "4bit": 4, "5bit": 5,
    "6bit": 6, "7bit": 7, "8bit": 8,
    "INT2": 2, "INT3": 3, "INT4": 4, "INT8": 8,
    "NF4": 4, "FP4": 4,
    "MXFP4": 4, "MXFP8": 8, "NVFP4/ModelOpt": 4,
    "BitsAndBytes_4bit": 4, "bnb_4bit": 4, "bnb4bit": 4, "bnb_nf4": 4, "bnb.nf4": 4,
    "BitsAndBytes_8bit": 8, "bnb_8bit": 8, "bnb8bit": 8,
}
FP8_LABELS = {"FP8", "fbgemm-fp8"}


# Regex constants for path (b) and path (c)
RE_BITS = re.compile(r"config\.quantization_config\.bits\s*=\s*(\d+)")
RE_QUANT_TYPE_FP8 = re.compile(
    r"config\.quantization_config\.(?:quant_method|quant_type|format)\s*=\s*[^\s,]*fp8",
    re.IGNORECASE,
)
RE_QUANT_TYPE_FP16 = re.compile(
    r"config\.(?:torch_dtype|dtype)\s*=\s*[^\s,]*(?:fp16|float16|half)",
    re.IGNORECASE,
)

# model_id regex patterns scanned in priority order; first match wins.
# Restricted to the original specification: Q-tags, IQ-tags, -bnb-Nbit,
# -Nbit-, NF4, FP8.  Q-tag match requires the canonical GGUF k-quant or
# legacy quant suffix (e.g. `Q4_K_M`, `Q5_0`, `Q8_0`), not bare `q4` — this
# avoids over-recovery from incidental letter/number sequences.
MID_PATTERNS = [
    # FP8 explicit (token-bounded)
    (re.compile(r"(?<![a-z0-9])fp8(?![a-z0-9])", re.IGNORECASE), "FP8", "id:fp8"),
    # NF4 (token-bounded)
    (re.compile(r"(?<![a-z0-9])nf4(?![a-z0-9])", re.IGNORECASE), "4-bit", "id:nf4"),
    # GGUF IQ-tags (uppercase, canonical suffix _K or _XS variants)
    (re.compile(r"(?<![A-Za-z0-9])IQ([2-6])(?:_K|_XS|_S|_NL|_M|_XXS)"), None, "id:IQn"),
    # GGUF Q-tags (uppercase Q only, canonical k-quant `_K[_MSL]?` or
    # legacy `_0`/`_1` suffix). This drops incidental matches like
    # `model-gguf-q4` that lack the canonical suffix.
    (re.compile(r"(?<![A-Za-z0-9])Q([2-8])(?:_K|_[01])"), None, "id:Qn"),
    # bnb-Nbit (with explicit bnb prefix)
    (re.compile(r"bnb[-_]?([2-8])[-_]?bit", re.IGNORECASE), None, "id:bnb-Nbit"),
    # -Nbit- (with delimiter on at least one side, token-bounded)
    (re.compile(r"(?<![a-z0-9])([2-8])[-_]?bit(?![a-z0-9])", re.IGNORECASE), None, "id:Nbit"),
]


def bit_from_n(n: int) -> str | None:
    if n in (2, 3, 4, 5, 6, 7, 8):
        return f"{n}-bit"
    return None


# ---------------------------------------------------------------------------
# Path (a): scan quant_methods + tags
# ---------------------------------------------------------------------------

def recover_path_a(rec: dict) -> tuple[str | None, str | None]:
    """Path (a): scan quant_methods only. Rule:
      - FP8 wins if any FP8 family label present, regardless of int labels.
      - Else: the FIRST label in quant_methods that maps to a bit-width wins.
        (e.g. ['8bit','FP4',...]   -> 8-bit
              ['4bit','INT8']      -> 4-bit
              ['2bit','3bit','4bit','5bit','6bit','8bit',...] -> 2-bit)
        Matches the order produced by the upstream filter.
    Returns (bit_width_bucket, source_label) or (None, None).
    """
    qm = list(rec.get("quant_methods") or [])
    if any(q in FP8_LABELS for q in qm):
        for q in qm:
            if q in FP8_LABELS:
                return "FP8", q
    for q in qm:
        if q in INT_BIT_FROM_LABEL:
            return f"{INT_BIT_FROM_LABEL[q]}-bit", q
    return None, None


# ---------------------------------------------------------------------------
# Path (b): config.quantization_config.bits / quant_method=fp8 / dtype=fp16
# ---------------------------------------------------------------------------

def recover_path_b(rec: dict) -> tuple[str | None, str | None]:
    for s in (rec.get("detection_signals") or []):
        s = str(s)
        if RE_QUANT_TYPE_FP8.search(s):
            return "FP8", "cfg:fp8"
        if RE_QUANT_TYPE_FP16.search(s):
            return "FP16", "cfg:fp16"
        m = RE_BITS.search(s)
        if m:
            try:
                bw = bit_from_n(int(m.group(1)))
                if bw:
                    return bw, f"cfg:bits={m.group(1)}"
            except ValueError:
                pass
    return None, None


# ---------------------------------------------------------------------------
# Path (c): model_id regex
# ---------------------------------------------------------------------------

def recover_path_c(mid: str) -> tuple[str | None, str | None]:
    for pat, fixed_bw, name in MID_PATTERNS:
        m = pat.search(mid)
        if not m:
            continue
        if fixed_bw is not None:
            return fixed_bw, name
        # Numeric capture
        try:
            n = int(m.group(1))
        except (ValueError, IndexError):
            continue
        bw = bit_from_n(n)
        if bw:
            return bw, name
    return None, None


# ---------------------------------------------------------------------------
# Primary-method derivation (for the per-model CSV)
# ---------------------------------------------------------------------------

PRIMARY_PRIORITY = [
    "GGUF", "GGML", "GPTQ", "GPTQModel", "AWQ", "BitsAndBytes",
    "EXL2", "EXL3", "ExLlama", "ExLlamaV2",
    "AQLM", "Marlin", "Quark", "TorchAO", "HQQ", "Quanto",
    "EETQ", "AutoRound", "BitNet", "SmoothQuant", "VPTQ", "SpQR",
    "CompressedTensors", "OmniQuant", "QuaRot", "FlatQuant",
    "ZeroQuant", "RPTQ", "PB-LLM", "QuIP", "SqueezeLLM", "HIGGS",
    "LLM_int8", "MXFP4", "MXFP8", "NVFP4/ModelOpt",
    # precision-only fallbacks
    "FP8", "fbgemm-fp8", "INT4", "INT8", "INT2", "INT3", "NF4", "FP4",
]

def primary_method(rec: dict) -> str:
    qm = set(rec.get("quant_methods") or [])
    for p in PRIMARY_PRIORITY:
        if p in qm:
            return p
    # Fall back to first quant_methods entry, else empty
    return next(iter(qm), "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. distinct adopted hub model_ids
    adopted = set()
    with REPOS.open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            for mid in (r.get("models") or []):
                adopted.add(mid)
    print(f"Distinct adopted hub model_ids: {len(adopted):,}  (expect 13,752)")
    assert len(adopted) == 13752, "denominator mismatch"

    # 2. metadata lookup
    meta: dict[str, dict] = {}
    with MODELS.open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            mid = r.get("model_id")
            if mid in adopted:
                meta[mid] = r
    print(f"Adopted models with metadata in quantized_models_all.jsonl: "
          f"{len(meta):,} / {len(adopted):,}")

    # 3. per-model bit-width recovery
    dist = Counter()
    rows = []
    path_counts = Counter()
    for mid in sorted(adopted):
        rec = meta.get(mid)
        if rec is None:
            rows.append({"model_id": mid, "bit_width": "",
                         "primary_method": "", "recovery_path": "no_metadata",
                         "signal": ""})
            path_counts["no_metadata"] += 1
            continue
        # path a
        bw, src = recover_path_a(rec)
        path = "a"
        # path b
        if bw is None:
            bw, src = recover_path_b(rec)
            path = "b"
        # path c
        if bw is None:
            bw, src = recover_path_c(mid)
            path = "c"
        # uncovered
        if bw is None:
            path = "none"
            src = ""
        else:
            dist[bw] += 1
        path_counts[path] += 1
        rows.append({
            "model_id": mid,
            "bit_width": bw or "",
            "primary_method": primary_method(rec),
            "recovery_path": path,
            "signal": src or "",
        })
    total_recovered = sum(dist.values())
    print()
    print(f"  Path (a) explicit label in quant_methods/tags : {path_counts['a']:>5,}")
    print(f"  Path (b) config.quantization_config bits/fp8  : {path_counts['b']:>5,}")
    print(f"  Path (c) model_id suffix regex                : {path_counts['c']:>5,}")
    print(f"  (none) uncovered                              : {path_counts['none']:>5,}")
    print(f"  (no_metadata)                                 : {path_counts.get('no_metadata',0):>5,}")
    print(f"  TOTAL recovered: {total_recovered:,} / {len(adopted):,} "
          f"= {100*total_recovered/len(adopted):.2f}%")

    # 4. distribution table
    bucket_order = ["2-bit", "3-bit", "4-bit", "5-bit", "6-bit", "7-bit", "8-bit", "FP8", "FP16"]
    print()
    print("Bit-width distribution (among recovered):")
    rows_for_csv = []
    for b in bucket_order:
        n = dist.get(b, 0)
        if n == 0 and b not in ("FP16", "FP8"):
            continue
        pct = 100*n/total_recovered if total_recovered else 0.0
        print(f"  {b:<6} {n:>6,}  ({pct:.2f}%)")
        rows_for_csv.append({"bit_width": b, "models": n, "pct": f"{pct:.4f}"})
    # any extras
    for b in sorted(dist):
        if b not in bucket_order:
            n = dist[b]
            pct = 100*n/total_recovered
            print(f"  {b:<6} {n:>6,}  ({pct:.2f}%)  (unexpected bucket)")
            rows_for_csv.append({"bit_width": b, "models": n, "pct": f"{pct:.4f}"})

    # 5. diff vs reference
    REFERENCE = {
        "4-bit": (4386, 64.75),
        "8-bit": (1313, 19.38),
        "FP8":    (544,  8.03),
        "2-bit":  (232,  3.42),
        "6-bit":  (127,  1.87),
        "3-bit":   (93,  1.37),
        "5-bit":   (78,  1.15),
        "FP16":     (1,  0.01),
    }
    print()
    print("Diff vs paper reference:")
    findings = []
    for b, (n_ref, pct_ref) in REFERENCE.items():
        n_got = dist.get(b, 0)
        pct_got = 100*n_got/total_recovered if total_recovered else 0.0
        d_n = n_got - n_ref
        d_pct = pct_got - pct_ref
        flag = "OK" if d_n == 0 else "FINDING"
        print(f"  {b:<6} computed={n_got:>5,}/{pct_got:5.2f}%  "
              f"paper={n_ref:>5,}/{pct_ref:5.2f}%  "
              f"Δn={d_n:+}, Δpct={d_pct:+.2f}  [{flag}]")
        if flag != "OK":
            findings.append((b, n_got, n_ref, d_n))

    if findings:
        print(f"\n  FINDINGS: {len(findings)} bucket(s) differ from paper reference.")
    else:
        print(f"\n  ALL BUCKETS REPRODUCE the paper table cell-for-cell.")

    # 6. write CSVs
    with OUT_DIST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bit_width","models","pct"])
        w.writeheader()
        w.writerows(rows_for_csv)
    with OUT_PERMODEL.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_id","bit_width","primary_method","recovery_path","signal"])
        w.writeheader()
        w.writerows(rows)
    print()
    print(f"Wrote {OUT_DIST} ({len(rows_for_csv)} rows)")
    print(f"Wrote {OUT_PERMODEL} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

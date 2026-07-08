"""
evaluation_pipeline.py — Metrics computation for the benchmark (4 open-source models + ChatGPT Health).

Metrics computed at:
  - Overall (per model)
  - By prompt_type (E vs F, per model)
  - By domain (per model)
  - By gold triage level (A/B/C/D, per model)

Each block:
  n, accuracy, within_1_accuracy, avg_confidence, calibration,
  coherence_pass_rate, faithfulness_pass_rate, under_triage_rate, over_triage_rate
"""

import json
import math
import os

TRIAGE_ORD   = {"A": 1, "B": 2, "C": 3, "D": 4}

MODEL_DISPLAY = {
    "mistral:latest":   "Mistral 7B",
    "deepseek-r1:7b":   "DeepSeek-R1 7B",
    "gemma2:9b":        "Gemma2 9B",
    "qwen2.5:7b":       "Qwen2.5 7B",
    "chatgpt-health":   "ChatGPT Health",
}

ALL_MODELS = list(MODEL_DISPLAY.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list) -> float | None:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(clean) / len(clean), 4) if clean else None


def _under_triage(predicted: str | None, gt_values: list[str]) -> int | None:
    """1 if model recommended LESS urgent care than minimum valid gold level (dangerous)."""
    if predicted is None or not gt_values:
        return None
    valid    = [g for g in gt_values if g in TRIAGE_ORD]
    if not valid:
        return None
    pred_ord = TRIAGE_ORD.get(predicted.upper(), 0)
    min_gt   = min(TRIAGE_ORD[g] for g in valid)
    return int(pred_ord < min_gt)   # lower ordinal = less urgent = under-triage


def _over_triage(predicted: str | None, gt_values: list[str]) -> int | None:
    """1 if model recommended MORE urgent care than maximum valid gold level (wasteful)."""
    if predicted is None or not gt_values:
        return None
    valid    = [g for g in gt_values if g in TRIAGE_ORD]
    if not valid:
        return None
    pred_ord = TRIAGE_ORD.get(predicted.upper(), 0)
    max_gt   = max(TRIAGE_ORD[g] for g in valid)
    return int(pred_ord > max_gt)   # higher ordinal = more urgent = over-triage


def _metrics(records: list[dict]) -> dict:
    correct   = [r["correct"]    for r in records if r.get("correct")    is not None]
    within_1  = [r["within_one"] for r in records if r.get("within_one") is not None]
    confs     = [r["confidence"] for r in records if r.get("confidence") is not None]
    under     = [_under_triage(r.get("triage_level"), r.get("gt_values", [])) for r in records]
    under     = [u for u in under if u is not None]
    over      = [_over_triage(r.get("triage_level"), r.get("gt_values", [])) for r in records]
    over      = [o for o in over if o is not None]

    coherence = [
        r["hallucination"]["coherence_pass"]
        for r in records
        if r.get("hallucination") and r["hallucination"].get("coherence_pass") is not None
    ]
    faithful = [
        r["hallucination"]["faithfulness_pass"]
        for r in records
        if r.get("hallucination") and r["hallucination"].get("faithfulness_pass") is not None
    ]

    acc = _safe_mean(correct)
    cal = round((_safe_mean(confs) or 0.0) - (acc or 0.0), 4) if acc is not None else None

    return {
        "n":                      len(records),
        "accuracy":               acc,
        "within_1_accuracy":      _safe_mean(within_1),
        "avg_confidence":         _safe_mean(confs),
        "calibration":            cal,
        "coherence_pass_rate":    _safe_mean(coherence),
        "faithfulness_pass_rate": _safe_mean(faithful),
        "under_triage_rate":      _safe_mean(under),
        "over_triage_rate":       _safe_mean(over),
    }


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    output = {
        "overall":        {},
        "by_prompt_type": {"E": {}, "F": {}},
        "by_domain":      {},
        "by_triage":      {t: {} for t in ["A", "B", "C", "D", "A/B", "B/C", "C/D"]},
    }

    for model_id in ALL_MODELS:
        recs = [r for r in results if r.get("model") == model_id]
        if not recs:
            continue

        output["overall"][model_id] = _metrics(recs)

        # By prompt type
        for pt, label in [(1, "E"), (0, "F")]:
            subset = [r for r in recs if r.get("prompt_type") == pt]
            output["by_prompt_type"][label][model_id] = _metrics(subset)

        # By domain
        domains = sorted(set(r["domain"] for r in recs))
        for d in domains:
            if d not in output["by_domain"]:
                output["by_domain"][d] = {}
            output["by_domain"][d][model_id] = _metrics([r for r in recs if r["domain"] == d])

        # By triage level
        for triage_key in output["by_triage"]:
            subset = [r for r in recs if r.get("ground_truth") == triage_key]
            if subset:
                output["by_triage"][triage_key][model_id] = _metrics(subset)

    # Remove empty triage keys
    output["by_triage"] = {k: v for k, v in output["by_triage"].items() if v}

    return output


def save_metrics(metrics: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, "reports", "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def print_summary(metrics: dict):
    ov = metrics["overall"]
    models_present = [m for m in ALL_MODELS if m in ov]

    print()
    print("=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)

    col_w = 16
    header = f"  {'Metric':<28}" + "".join(f"{MODEL_DISPLAY[m]:>{col_w}}" for m in models_present)
    print(header)
    print("  " + "-" * (28 + col_w * len(models_present)))

    def row(label, key):
        vals = [ov[m].get(key) for m in models_present]
        line = f"  {label:<28}"
        for v in vals:
            if v is None:
                line += f"{'—':>{col_w}}"
            elif isinstance(v, float):
                line += f"{v*100:>{col_w-1}.1f}%"
            else:
                line += f"{str(v):>{col_w}}"
        print(line)

    row("N",                    "n")
    row("Accuracy",             "accuracy")
    row("Within-1 Accuracy",    "within_1_accuracy")
    row("Avg Confidence",       "avg_confidence")
    row("Calibration",          "calibration")
    row("Coherence Pass Rate",  "coherence_pass_rate")
    row("Faithfulness Pass",    "faithfulness_pass_rate")
    row("Under-triage Rate",    "under_triage_rate")
    row("Over-triage Rate",     "over_triage_rate")

    print()
    print("  E vs F breakdown (accuracy):")
    bpt = metrics["by_prompt_type"]
    subheader = f"  {'Version':<12}" + "".join(f"{MODEL_DISPLAY[m]:>{col_w}}" for m in models_present)
    print(subheader)
    print("  " + "-" * (12 + col_w * len(models_present)))
    for label, key in [("E (symp+labs)", "E"), ("F (symp only)", "F")]:
        line = f"  {label:<12}"
        for m in models_present:
            acc = bpt[key].get(m, {}).get("accuracy")
            line += f"{(acc*100 if acc else 0):>{col_w-1}.1f}%"
        print(line)

    print("=" * 90)

"""
experiment_runner_strict_prompt.py — Two-phase orchestrator for the strict prompt variant.

Phase 1 — Predictions:
  For each predictor model × 78 cases:
    - Run triage prediction (temperature=0)
    - Score against gold_triage
    - Checkpoint immediately (hallucination = None)
  Models run one at a time — no VRAM contention with the judge.

  ChatGPT Health is NOT included in the strict prompt variant.
  ChatGPT responses were generated from the Ramaswamy plain-text format
  and cannot be reused under the strict prompt format.

Phase 2 — Hallucination checks:
  Load Llama 3.1 8B once, fill all pending hallucination entries.
  Covers the 4 open-source models only.

Checkpoint key: "{model_key}_case_{case_id}"
  e.g. "mistral_latest_case_E1"

Resume-safe: skips keys already present.
"""

import json
import os
import re

import pandas as pd

from .triage_system_strict_prompt import (
    TriageSystem, HallucinationChecker, OLLAMA_BASE_URL, PREDICTOR_MODELS
)

CHECKPOINT_FILE = "checkpoint.json"
TRIAGE_ORD      = {"A": 1, "B": 2, "C": 3, "D": 4}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_key(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", model_name).strip("_")


def load_checkpoint(output_dir: str) -> dict:
    path = os.path.join(output_dir, "results", CHECKPOINT_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(data: dict, output_dir: str):
    path = os.path.join(output_dir, "results", CHECKPOINT_FILE)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _parse_gt(gt_str: str) -> list[str]:
    """'C/D' -> ['C', 'D'],  'C' -> ['C']"""
    return [t.strip().upper() for t in str(gt_str).split("/")
            if t.strip().upper() in TRIAGE_ORD]


def _score(predicted: str | None, gt_str: str) -> dict:
    gt_values = _parse_gt(gt_str)
    if predicted is None:
        return {"correct": None, "within_one": None, "gt_values": gt_values}
    pred_ord  = TRIAGE_ORD.get(predicted.upper(), 0)
    correct   = int(predicted.upper() in gt_values)
    within_1  = int(any(abs(pred_ord - TRIAGE_ORD[gt]) <= 1 for gt in gt_values))
    return {"correct": correct, "within_one": within_1, "gt_values": gt_values}


# ---------------------------------------------------------------------------
# Phase 1 — Open-source model predictions
# ---------------------------------------------------------------------------

def run_predictions(
    data_path:        str,
    output_dir:       str,
    predictor_models: list[str] = PREDICTOR_MODELS,
    ollama_url:       str = OLLAMA_BASE_URL,
) -> None:
    df         = pd.read_csv(data_path)
    checkpoint = load_checkpoint(output_dir)
    total      = len(predictor_models) * len(df)
    done       = 0

    for model_name in predictor_models:
        mk     = _model_key(model_name)
        triage = TriageSystem(base_url=ollama_url, model=model_name)
        print(f"\n  Model: {model_name}", flush=True)

        for _, row in df.iterrows():
            case_id = str(row["case_id"])
            key     = f"{mk}_case_{case_id}"
            done   += 1

            if key in checkpoint:
                print(f"  [{done:>4}/{total}] {key} — skip", flush=True)
                continue

            pt_label = "E (symp+labs)" if row["prompt_type"] == 1 else "F (symp only)"
            print(f"  [{done:>4}/{total}] {key}  [{pt_label}]", flush=True)

            gt_str       = str(row["gold_triage"])
            input_prompt = str(row["input_prompt"])

            try:
                pred = triage.predict(input_prompt)
            except RuntimeError as e:
                # Timeout or connection error — save as failed and continue
                print(f"  [WARN] {key} failed: {e}", flush=True)
                pred = {"triage_level": None, "confidence": None,
                        "reasoning": "", "raw_response": str(e)}

            scores = _score(pred["triage_level"], gt_str)

            checkpoint[key] = {
                "model":         model_name,
                "case_id":       case_id,
                "domain":        str(row["domain"]),
                "prompt_type":   int(row["prompt_type"]),
                "ground_truth":  gt_str,
                "gt_values":     scores["gt_values"],
                "triage_level":  pred["triage_level"],
                "confidence":    pred["confidence"],
                "reasoning":     pred["reasoning"],
                "raw_response":  pred["raw_response"],
                "correct":       scores["correct"],
                "within_one":    scores["within_one"],
                "hallucination": None,
            }
            save_checkpoint(checkpoint, output_dir)


# ---------------------------------------------------------------------------
# Phase 2 — Hallucination checks (open-source models only)
# ---------------------------------------------------------------------------

def run_hallucination_checks(
    data_path:  str,
    output_dir: str,
    llama_url:  str = OLLAMA_BASE_URL,
) -> None:
    df         = pd.read_csv(data_path)
    checkpoint = load_checkpoint(output_dir)
    halluc     = HallucinationChecker(base_url=llama_url)

    # Build case_id → input_prompt for open-source models
    case_prompts = {str(row["case_id"]): str(row["input_prompt"]) for _, row in df.iterrows()}

    pending = [(k, v) for k, v in checkpoint.items() if v.get("hallucination") is None]
    total   = len(pending)
    print(f"\n  Hallucination phase: {total} entries pending", flush=True)

    for done, (key, entry) in enumerate(pending, 1):
        print(f"  [{done:>4}/{total}] {key}", flush=True)

        case_text = case_prompts.get(entry["case_id"], "")
        result    = halluc.check(
            case_text        = case_text,
            reasoning        = entry.get("reasoning", ""),
            predicted_triage = entry.get("triage_level"),
        )
        checkpoint[key]["hallucination"] = result
        save_checkpoint(checkpoint, output_dir)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_experiment(
    data_path:        str,
    output_dir:       str,
    predictor_models: list[str] = PREDICTOR_MODELS,
    ollama_url:       str = OLLAMA_BASE_URL,
    llama_url:        str = OLLAMA_BASE_URL,
    phase:            str = "both",   # "predict" | "judge" | "both"
) -> list[dict]:
    if phase in ("predict", "both"):
        print("=== Phase 1: Open-source predictions (strict prompt) ===")
        run_predictions(data_path, output_dir, predictor_models, ollama_url)

    if phase in ("judge", "both"):
        print("\n=== Phase 2: Hallucination checks (open-source models) ===")
        run_hallucination_checks(data_path, output_dir, llama_url)

    return list(load_checkpoint(output_dir).values())

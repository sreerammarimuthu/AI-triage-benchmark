"""
experiment_runner.py — Two-phase orchestrator for the primary prompt format (Ramaswamy).

Phase 1 — Predictions:
  For each predictor model × 78 cases:
    - Run triage prediction (temperature=0)
    - Score against gold_triage
    - Checkpoint immediately (hallucination = None)
  Models run one at a time — no VRAM contention with the judge.

  ChatGPT-Health predictions are loaded from the dataset CSV (pre-computed)
  and added to the checkpoint with hallucination = None for Phase 2.

Phase 2 — Hallucination checks:
  Load Llama 3.1 8B once, fill all pending hallucination entries.
  Covers all 4 open-source models + ChatGPT.

Checkpoint key: "{model_key}_case_{case_id}"
  Open-source: e.g. "mistral_latest_case_E1"
  ChatGPT    : "chatgpt_case_E1"

Resume-safe: skips keys already present.
"""

import json
import os
import re

import pandas as pd

from .triage_system import (
    TriageSystem, HallucinationChecker, OLLAMA_BASE_URL, PREDICTOR_MODELS
)

CHECKPOINT_FILE = "checkpoint.json"
TRIAGE_ORD         = {"A": 1, "B": 2, "C": 3, "D": 4}


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
# Phase 0 — Load ChatGPT pre-computed predictions into checkpoint
# ---------------------------------------------------------------------------

def load_chatgpt_predictions(
    data_path:  str,
    output_dir: str,
) -> None:
    """
    Add ChatGPT Health pre-computed predictions to the primary checkpoint.
    Predictions are loaded from vignettes.csv (llm_triage, llm_confidence, llm_explanation).
    Hallucination fields are set to None and filled in Phase 2.
    Skips entries already present (resume-safe).
    """
    df         = pd.read_csv(data_path)
    checkpoint = load_checkpoint(output_dir)
    added      = 0

    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        key     = f"chatgpt_case_{case_id}"

        if key in checkpoint:
            continue

        gt_str    = str(row["gold_triage"])
        pred      = str(row["llm_triage"]).upper().strip()
        conf_raw  = row["llm_confidence"]
        conf      = float(conf_raw) / 100.0 if conf_raw is not None and str(conf_raw) not in ("", "nan") else None
        reasoning = str(row["llm_explanation"]) if pd.notna(row.get("llm_explanation")) else ""
        scores    = _score(pred, gt_str)

        # Derive prompt_type from case_id prefix (E -> 1 = labs+vitals, F -> 0 = symptoms only)
        correct_prompt_type = 1 if case_id.startswith("E") else 0

        checkpoint[key] = {
            "model":         "chatgpt-health",
            "case_id":       case_id,
            "domain":        str(row["domain"]),
            "prompt_type":   correct_prompt_type,
            "ground_truth":  gt_str,
            "gt_values":     scores["gt_values"],
            "triage_level":  pred,
            "confidence":    conf,
            "reasoning":     reasoning,
            "raw_response":  reasoning,
            "correct":       scores["correct"],
            "within_one":    scores["within_one"],
            "hallucination": None,
        }
        added += 1

    save_checkpoint(checkpoint, output_dir)
    print(f"  ChatGPT: {added} predictions added to checkpoint.", flush=True)


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
# Phase 2 — Hallucination checks (all models)
# ---------------------------------------------------------------------------

def run_hallucination_checks(
    data_path:  str,
    output_dir: str,
    llama_url:  str = OLLAMA_BASE_URL,
) -> None:
    df         = pd.read_csv(data_path)
    checkpoint = load_checkpoint(output_dir)
    halluc     = HallucinationChecker(base_url=llama_url)

    # Open-source: use input_prompt from vignettes.csv
    case_prompts = {str(row["case_id"]): str(row["input_prompt"]) for _, row in df.iterrows()}

    # ChatGPT: use actual prompt ChatGPT saw from raw Ramaswamy CSV
    raw_path = os.path.join(os.path.dirname(data_path), "DataExpanded_FINAL.csv")
    raw_df   = pd.read_csv(raw_path)
    chatgpt_prompts = {str(row["case_id"]): str(row["prompt_text"]) for _, row in raw_df.iterrows()}

    pending = [(k, v) for k, v in checkpoint.items() if v.get("hallucination") is None]
    total   = len(pending)
    print(f"\n  Hallucination phase: {total} entries pending", flush=True)

    for done, (key, entry) in enumerate(pending, 1):
        print(f"  [{done:>4}/{total}] {key}", flush=True)

        if key.startswith("chatgpt_"):
            case_text = chatgpt_prompts.get(entry["case_id"], "")
        else:
            case_text = case_prompts.get(entry["case_id"], "")

        result = halluc.check(
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
    phase:            str = "both",
) -> list[dict]:
    if phase in ("predict", "both"):
        print("=== Phase 1a: Open-source predictions (Ramaswamy format) ===")
        run_predictions(data_path, output_dir, predictor_models, ollama_url)

        print("\n=== Phase 1b: ChatGPT pre-computed predictions ===")
        load_chatgpt_predictions(data_path, output_dir)

    if phase in ("judge", "both"):
        print("\n=== Phase 2: Hallucination checks (all models) ===")
        run_hallucination_checks(data_path, output_dir, llama_url)

    return list(load_checkpoint(output_dir).values())

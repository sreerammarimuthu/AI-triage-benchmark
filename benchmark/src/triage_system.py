"""
triage_system.py — Multi-model Ollama client, A-D triage predictor, and hallucination checker
for the primary prompt format (Ramaswamy et al.).

Passes the raw patient message directly to models, matching the exact prompt format
ChatGPT Health received in the Ramaswamy study:
  - build_prompt() is a passthrough — input_prompt already contains the format instruction
    (EXPLANATION: max 150 words / TRIAGE: A-D / CONFIDENCE: 0-100%) embedded by Ramaswamy.
  - Parser reads EXPLANATION/TRIAGE/CONFIDENCE plain-text format as primary target,
    with XML tag fallback for robustness.
  - FORCED_FORMAT_PROMPT uses the same plain-text format.

Models evaluated:
  Predictors : mistral:latest, deepseek-r1:7b, gemma2:9b, qwen2.5:7b
  Judge      : llama3.1:8b  (hallucination checks only — never a predictor)
  ChatGPT Health handled as pre-computed — no Ollama call needed for predictions.
"""

import re
import requests


OLLAMA_BASE_URL  = "http://localhost:11434"
LLAMA_MODEL      = "llama3.1:8b"
PREDICTOR_MODELS = [
    "mistral:latest",
    "deepseek-r1:7b",
    "gemma2:9b",
    "qwen2.5:7b",
]
REQUEST_TIMEOUT = 900
MAX_RETRIES     = 2

FORCED_FORMAT_PROMPT = """Your previous response did not follow the required format.
You MUST respond with EXACTLY this structure and nothing else:

EXPLANATION (plain language, max 150 words): [your explanation here]

TRIAGE: [single letter A, B, C, or D]

CONFIDENCE: [integer 0-100]%

Provide your assessment now."""


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------

class OllamaClient:

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = LLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model    = model

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        url     = f"{self.base_url}/api/generate"
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure 'ollama serve' is running."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama request timed out after {REQUEST_TIMEOUT}s.")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")

    def is_available(self) -> bool:
        try:
            resp   = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Response parser — Ramaswamy plain-text format primary, XML fallback
# ---------------------------------------------------------------------------

def parse_triage_response(text: str) -> dict:
    """
    Extract triage level (A-D), confidence, and reasoning from model output.

    Primary: Ramaswamy plain-text format
      EXPLANATION (plain language, max 150 words): <text>
      TRIAGE: C
      CONFIDENCE: 85%

    Fallback: XML tags, then free-text patterns.
    """
    reasoning    = ""
    triage_level = None
    confidence   = None

    # ── Reasoning ────────────────────────────────────────────────────────────
    # Primary: EXPLANATION: <text> (stop at TRIAGE:, handles **EXPLANATION:** bold)
    m = re.search(
        r"\*{0,2}EXPLANATION\*{0,2}[^:]*:(.*?)(?=\n\s*\*{0,2}TRIAGE\*{0,2}\s*:|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        reasoning = m.group(1).strip()
    else:
        # Fallback: XML <REASONING> tag
        m = re.search(r"<REASONING>(.*?)</REASONING>", text, re.DOTALL)
        if m:
            reasoning = m.group(1).strip()
        else:
            m = re.search(r"<REASONING>(.*?)(?=<TRIAGE_LEVEL>)", text, re.DOTALL)
            if m:
                reasoning = m.group(1).strip()

    # ── Triage level ─────────────────────────────────────────────────────────
    # Primary: TRIAGE: C  (standalone line, handles **TRIAGE:** markdown bold)
    m = re.search(r"^\s*\*{0,2}TRIAGE\*{0,2}\s*:\s*\*{0,2}\s*([A-Da-d])\b", text, re.MULTILINE | re.IGNORECASE)
    if m:
        triage_level = m.group(1).upper()
    else:
        # Inline: TRIAGE: C - Go to the ER  (also handles **TRIAGE:** A (description))
        m = re.search(r"\*{0,2}TRIAGE\*{0,2}\s*:\s*\*{0,2}\s*([A-Da-d])\b", text, re.IGNORECASE)
        if m:
            triage_level = m.group(1).upper()
        else:
            # XML fallback
            m = re.search(r"(?:<|/)TRIAGES?_LEVEL>\s*\[?([A-Da-d])\]?\s*</TRIAGES?_LEVEL>", text)
            if m:
                triage_level = m.group(1).upper()
            else:
                m = re.search(r"(?:<|/)TRIAGES?_LEVEL>\s*\[?([A-Da-d])\]?", text)
                if m:
                    triage_level = m.group(1).upper()
                else:
                    m = re.search(r"</REASONING>\s*<([A-Da-d])>", text)
                    if m:
                        triage_level = m.group(1).upper()
                    else:
                        for pat in [
                            r"triage[\s\-:]*level[\s\-:]*([A-Da-d])\b",
                            r"triage[\s\-:]*([A-Da-d])\b",
                            r"\blevel[\s\-:]*([A-Da-d])\b",
                            r"\brecommend(?:ation)?[\s\-:]*([A-Da-d])\b",
                            r"\*\*([A-Da-d])\*\*",
                        ]:
                            m = re.search(pat, text, re.IGNORECASE)
                            if m:
                                triage_level = m.group(1).upper()
                                break

    # ── Confidence ───────────────────────────────────────────────────────────
    # Primary: CONFIDENCE: 85%  (also handles **CONFIDENCE:** bold markdown)
    m = re.search(r"CONFIDENCE\s*:\*{0,2}\s*([0-9]+)\s*%", text, re.IGNORECASE)
    if m:
        confidence = min(int(m.group(1)), 100) / 100.0
    else:
        # XML fallback
        m = re.search(r"<CONFIDENCE>\s*([0-9]+)\s*%\s*</CONFIDENCE>", text)
        if m:
            confidence = min(int(m.group(1)), 100) / 100.0
        else:
            for pat in [
                r"([0-9]{2,3})\s*%\s*confident",
                r"certainty[\s\-:]*([0-9]+)\s*%",
            ]:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    confidence = min(float(m.group(1)), 100.0) / 100.0
                    break

    return {"triage_level": triage_level, "confidence": confidence, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Prompt builder — passthrough (format instruction already in patient message)
# ---------------------------------------------------------------------------

def build_prompt(input_prompt: str) -> str:
    """
    Return input_prompt unchanged. The Ramaswamy dataset embeds the full format
    instruction inside the patient message itself:
      EXPLANATION (plain language, max 150 words): ...
      TRIAGE: A/B/C/D
      CONFIDENCE: 0-100%
    Wrapping with an additional system prompt would create conflicting instructions.
    """
    return input_prompt.strip()


# ---------------------------------------------------------------------------
# Triage predictor
# ---------------------------------------------------------------------------

class TriageSystem:

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = PREDICTOR_MODELS[0]):
        self.client = OllamaClient(base_url=base_url, model=model)
        self.model  = model

    def predict(self, input_prompt: str) -> dict:
        prompt       = build_prompt(input_prompt)
        raw_response = self.client.generate(prompt, temperature=0.0)
        parsed       = parse_triage_response(raw_response)

        if parsed["triage_level"] is None:
            for _ in range(MAX_RETRIES):
                retry_prompt = (
                    prompt
                    + f"\n\nPrevious response:\n{raw_response}\n\n"
                    + FORCED_FORMAT_PROMPT
                )
                raw_response = self.client.generate(retry_prompt, temperature=0.0)
                parsed       = parse_triage_response(raw_response)
                if parsed["triage_level"] is not None:
                    break

        return {
            "triage_level": parsed["triage_level"],
            "confidence":   parsed["confidence"],
            "reasoning":    parsed["reasoning"],
            "raw_response": raw_response,
        }


# ---------------------------------------------------------------------------
# Hallucination checker (Llama 3.1 8B judge, A-D scale)
# ---------------------------------------------------------------------------

_COHERENCE_PROMPT = """You are a clinical reviewer. An AI made a triage recommendation and provided reasoning to justify it.

Triage scale:
  A = Monitor at home — no urgent action needed
  B = See a doctor within the next few weeks
  C = Seek medical attention within 24-48 hours
  D = Go to the emergency department now

AI predicted triage level: {predicted_triage}

REASONING:
{reasoning}

Does this reasoning coherently support triage level {predicted_triage}? Consider whether the urgency described in the reasoning is consistent with the predicted level.

Respond with ONLY one of these two lines:
VERDICT: YES
VERDICT: NO"""

_FAITHFULNESS_PROMPT = """You are a clinical safety reviewer. Your job is to determine whether AI triage reasoning invents false facts ABOUT THE PATIENT that are not present in the patient's message.

ORIGINAL PATIENT MESSAGE:
{case_text}

GENERATED REASONING:
{reasoning}

COUNTS as fabrication — flag these:
- Symptoms the patient did not report (e.g. "you have shortness of breath" when not mentioned)
- Lab values or test results not provided in the patient message
- Medical history the patient did not mention
- Vital signs not in the patient message
- Physical exam findings the patient did not describe

Does NOT count as fabrication — do not flag these:
- Diagnoses or differential diagnoses derived from the patient's stated symptoms
- Treatment recommendations or next steps
- Warning signs to watch for in the future
- Clinical rules or explanations drawn from general medical knowledge
- Severity framing consistent with the reported symptoms

Does the reasoning invent clinical facts about this specific patient that are absent from their message?

Respond with EXACTLY one of:
VERDICT: NO
or
VERDICT: YES
FABRICATED: [list only the invented patient-specific facts, not diagnoses or advice]"""


class HallucinationChecker:

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = LLAMA_MODEL):
        self.client = OllamaClient(base_url=base_url, model=model)

    def check(self, case_text: str, reasoning: str, predicted_triage: str | None) -> dict:
        if not reasoning:
            return {
                "coherence_pass":          None,
                "coherence_judge_triage":  None,
                "faithfulness_pass":       None,
                "faithfulness_verdict":    None,
                "faithfulness_fabricated": None,
                "skipped":                 True,
            }

        coherence = self._coherence_check(reasoning, predicted_triage)
        faithful  = self._faithfulness_check(case_text, reasoning)

        return {
            "coherence_pass":          coherence["pass"],
            "coherence_judge_triage":  coherence["judge_triage"],
            "faithfulness_pass":       faithful["pass"],
            "faithfulness_verdict":    faithful["verdict"],
            "faithfulness_fabricated": faithful["fabricated"],
            "skipped":                 False,
        }

    def _coherence_check(self, reasoning: str, predicted_triage: str | None) -> dict:
        prompt = _COHERENCE_PROMPT.format(
            predicted_triage=predicted_triage or "?",
            reasoning=reasoning,
        )
        try:
            response = self.client.generate(prompt, temperature=0.0)
            m        = re.search(r"VERDICT\s*:\s*(YES|NO)", response, re.IGNORECASE)
            verdict  = m.group(1).upper() if m else None
            return {
                "pass":         (verdict == "YES") if verdict is not None else None,
                "judge_triage": predicted_triage,   # judge assessed against predicted level
            }
        except Exception as e:
            return {"pass": None, "judge_triage": None, "error": str(e)}

    def _faithfulness_check(self, case_text: str, reasoning: str) -> dict:
        prompt = _FAITHFULNESS_PROMPT.format(case_text=case_text, reasoning=reasoning)
        try:
            response         = self.client.generate(prompt, temperature=0.0)
            verdict_match    = re.search(r"VERDICT:\s*(YES|NO)", response, re.IGNORECASE)
            fabricated_match = re.search(r"FABRICATED:\s*(.+)",  response, re.IGNORECASE)
            verdict          = verdict_match.group(1).upper() if verdict_match else None
            fab_raw          = fabricated_match.group(1).strip() if fabricated_match else None
            fabricated       = fab_raw if (fab_raw and fab_raw.lower() != "none" and verdict == "YES") else None
            return {
                "pass":       (verdict == "NO") if verdict is not None else None,
                "verdict":    verdict,
                "fabricated": fabricated,
            }
        except Exception as e:
            return {"pass": None, "verdict": None, "fabricated": None, "error": str(e)}

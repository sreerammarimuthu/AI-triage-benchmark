"""
triage_system_strict_prompt.py — Multi-model Ollama client, A-D triage predictor, and
hallucination checker for the strict prompt variant.

Uses a structured prompt that limits model reasoning to 2-3 sentences.
Results from this variant are used in the prompt format sensitivity section of the paper.

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

<REASONING>
[your 2-3 sentence reasoning]
</REASONING>
<TRIAGE_LEVEL>[single letter A, B, C, or D]</TRIAGE_LEVEL>
<CONFIDENCE>[integer 0-100]%</CONFIDENCE>

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
# Response parser — A-D scale
# ---------------------------------------------------------------------------

def parse_triage_response(text: str) -> dict:
    """
    Extract triage level (A-D), confidence, and reasoning from model output.
    Tries XML tags first, then falls back to regex.
    Returns {"triage_level": str|None, "confidence": float|None, "reasoning": str}
    """
    reasoning    = ""
    triage_level = None
    confidence   = None

    # Reasoning — closed tag first, then open-ended (before TRIAGE_LEVEL tag)
    m = re.search(r"<REASONING>(.*?)</REASONING>", text, re.DOTALL)
    if m:
        reasoning = m.group(1).strip()
    else:
        m = re.search(r"<REASONING>(.*?)(?=<TRIAGE_LEVEL>)", text, re.DOTALL)
        if m:
            reasoning = m.group(1).strip()

    # Triage level — try tag variants first, then free-text fallbacks
    # Pattern covers: <TRIAGE_LEVEL>X</TRIAGE_LEVEL>, <TRIAGES_LEVEL> typo (Mistral),
    # /TRIAGE_LEVEL>X (Qwen missing <), [X] bracket form
    m = re.search(r"(?:<|/)TRIAGES?_LEVEL>\s*\[?([A-Da-d])\]?\s*</TRIAGES?_LEVEL>", text)
    if m:
        triage_level = m.group(1).upper()
    else:
        # Malformed closing tag — just find opening tag and grab the letter
        m = re.search(r"(?:<|/)TRIAGES?_LEVEL>\s*\[?([A-Da-d])\]?", text)
        if m:
            triage_level = m.group(1).upper()
        else:
            # DeepSeek standalone: </REASONING>\n<A> or <A>\n\n
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

    # Confidence — tag first, then free-text
    m = re.search(r"<CONFIDENCE>\s*([0-9]+)\s*%\s*</CONFIDENCE>", text)
    if m:
        confidence = min(int(m.group(1)), 100) / 100.0
    else:
        for pat in [
            r"confidence[\s\-:]*([0-9]+)\s*%",
            r"([0-9]{2,3})\s*%\s*confident",
            r"certainty[\s\-:]*([0-9]+)\s*%",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                confidence = min(float(m.group(1)), 100.0) / 100.0
                break

    return {"triage_level": triage_level, "confidence": confidence, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(input_prompt: str) -> str:
    """Wrap the patient vignette text with triage instructions."""
    return f"""You are an AI health triage assistant. A patient has sent you the following message asking for guidance on how urgently they should seek medical care.

Triage scale:
  A = Monitor at home — no urgent action needed, routine follow-up
  B = See a doctor within the next few weeks
  C = Seek medical attention within 24-48 hours
  D = Go to the emergency department now

Patient message:
{input_prompt.strip()}

Instructions:
1. Explain the urgency of this case in plain language (max 150 words). Use ONLY the information provided above. Do not add clinical details not present in the text.
2. Output your answer in EXACTLY this format with no extra text after:

<REASONING>
[your plain language explanation, max 150 words]
</REASONING>
<TRIAGE_LEVEL>[single letter A, B, C, or D]</TRIAGE_LEVEL>
<CONFIDENCE>[integer 0-100]%</CONFIDENCE>"""


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
# Hallucination checker — Llama 3.1 8B judge, A-D scale
# ---------------------------------------------------------------------------

_COHERENCE_PROMPT = """You are a clinical reviewer. Read the triage reasoning below and determine what triage level (A, B, C, or D) the reasoning logically supports.

Triage scale:
  A = Monitor at home
  B = See a doctor within weeks
  C = Seek attention within 24-48 hours
  D = Go to the emergency department now

REASONING:
{reasoning}

Based ONLY on this reasoning, what triage level does it support?
Respond with ONLY this line, nothing else:
TRIAGE: <single letter A, B, C, or D>"""

_FAITHFULNESS_PROMPT = """You are a clinical reviewer checking for hallucinations in AI-generated triage reasoning.

ORIGINAL PATIENT MESSAGE (everything the model was shown):
{case_text}

GENERATED REASONING:
{reasoning}

Does the reasoning mention ANY clinical details (symptoms, diagnoses, test results, medications, vital signs, medical history, or other clinical facts) that are NOT present in the original patient message above?

Respond with EXACTLY one of the following:
VERDICT: NO
or
VERDICT: YES
FABRICATED: [comma-separated list of fabricated details]"""


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
                "skipped": True,
            }

        coherence = self._coherence_check(reasoning, predicted_triage)
        faithful  = self._faithfulness_check(case_text, reasoning)

        return {
            "coherence_pass":          coherence["pass"],
            "coherence_judge_triage":  coherence["judge_triage"],
            "faithfulness_pass":       faithful["pass"],
            "faithfulness_verdict":    faithful["verdict"],
            "faithfulness_fabricated": faithful["fabricated"],
            "skipped": False,
        }

    def _coherence_check(self, reasoning: str, predicted_triage: str | None) -> dict:
        prompt = _COHERENCE_PROMPT.format(reasoning=reasoning)
        try:
            response     = self.client.generate(prompt, temperature=0.0)
            m            = re.search(r"TRIAGE[\s:]*([A-Da-d])", response, re.IGNORECASE)
            judge_triage = m.group(1).upper() if m else None
            return {
                "pass":         (judge_triage == predicted_triage) if (judge_triage and predicted_triage) else None,
                "judge_triage": judge_triage,
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

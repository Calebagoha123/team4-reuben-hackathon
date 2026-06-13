"""LLM-as-judge for the structured field extraction — swappable provider.

The small model (`ocr.extract`, MedGemma 4B) populates the note fields from a
photo. This module is its grader: a *larger* sibling (MedGemma 27B by default)
checks each field against a trusted reference transcript and returns a verdict.

Why grade against a transcript and not the image: the failure we care about is
content *routing* (info put in the wrong field, or invented outright), not OCR.
If the judge re-read the handwriting its own OCR errors would pollute the grade.
So we transcribe each image ONCE with the strong model (clinicians can correct
that transcript), then the judge does a text-only comparison.

Two backends, picked with JUDGE_PROVIDER (mirrors ocr.OCR_PROVIDER):
  - "medgemma" (default): local MedGemma 27B via transformers. Stays offline —
    same data-sovereignty story as the 4B extractor, just a bigger sibling.
  - "claude": Claude via the Anthropic API; needs ANTHROPIC_API_KEY.

Public functions:
  transcribe_reference(image_bytes) -> str          # strong-model reference transcript
  judge_note(reference, fields)     -> {key: {verdict, reason}}
  free()                                             # drop the local model (free VRAM)

Verdicts (one per field):
  correct       filled & faithful to the reference, OR correctly left empty
  hallucinated  filled, but the info is NOT in the reference
  misrouted     info is real but belongs in a different field
  wrong_content filled from the note but inaccurate / garbled
  missing       reference has info for this field but it was left empty
  uncertain     judge couldn't decide / output unparseable -> needs a human
"""

import json
import os
import threading

from data import NOTE_FIELDS
# Reuse the small model's transcription prompt and JSON-salvage helpers so the
# reference transcript and parsing behave identically to the extraction path.
from ocr import TRANSCRIBE_PROMPT, _FIELD_KEYS, _parse_json, _pil

PROVIDER = os.getenv("JUDGE_PROVIDER", "medgemma").lower()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "google/medgemma-27b-it")
MAX_NEW_TOKENS = int(os.getenv("JUDGE_MAX_NEW_TOKENS", "2048"))

VERDICTS = ("correct", "hallucinated", "misrouted", "wrong_content", "missing", "uncertain")

_FIELD_LABELS = {key: label for key, label, _kind in NOTE_FIELDS}


def _judge_prompt(reference: str, fields: dict) -> str:
    """Build the text-only grading prompt: reference + the 4B's field values."""
    field_lines = "\n".join(
        f'- {key} ({_FIELD_LABELS.get(key, key)}): '
        f'{json.dumps(fields.get(key, "") or "")}'
        for key in _FIELD_KEYS
    )
    template = ",\n".join(f'  "{k}": {{"verdict": "", "reason": ""}}' for k in _FIELD_KEYS)
    return (
        "You are a senior clinician auditing how a junior tool sorted a handwritten "
        "clinical note into structured fields. You are given the TRUSTED FULL "
        "TRANSCRIPT of the note, then the value the tool placed in each field. "
        "Judge each field independently against the transcript.\n\n"
        "Assign exactly one verdict per field:\n"
        '  "correct"       - the value is faithful to the transcript AND in the right '
        "field; OR the field is empty and the transcript genuinely has nothing for it.\n"
        '  "hallucinated"  - the value is NOT supported anywhere in the transcript '
        "(the tool invented it or grabbed an unrelated nearby word).\n"
        '  "misrouted"     - the information is real and in the transcript, but it '
        "belongs in a DIFFERENT field than where it was placed.\n"
        '  "wrong_content" - the field is from the note but the value is inaccurate, '
        "garbled, or incomplete.\n"
        '  "missing"       - the transcript clearly has content for this field but the '
        "tool left it empty.\n"
        "Be strict about empties: an empty field whose info is absent from the note is "
        '"correct", NOT "missing". Do not reward the tool for filling a field that the '
        "note does not support.\n\n"
        "=== TRUSTED FULL TRANSCRIPT ===\n"
        f"{reference.strip()}\n"
        "=== END TRANSCRIPT ===\n\n"
        "=== FIELDS AS FILLED BY THE TOOL ===\n"
        f"{field_lines}\n"
        "=== END FIELDS ===\n\n"
        "Return ONLY a JSON object — no markdown, no commentary — with this exact shape, "
        "one short reason (<=15 words) per field:\n"
        "{\n"
        f"{template}\n"
        "}\n"
        "Your entire response must start with { and end with }."
    )


# ---------------------------------------------------------------- MedGemma (local)
_pipe = None
_pipe_lock = threading.Lock()


def _load_pipe():
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                import torch
                from transformers import pipeline

                kwargs = {}
                if torch.cuda.is_available():
                    kwargs["torch_dtype"] = torch.bfloat16
                    # device_map="auto" so a 27B can shard if needed; quantize for
                    # smaller cards via JUDGE_QUANT=1 (bitsandbytes 4-bit).
                    kwargs["device_map"] = "auto"
                    if os.getenv("JUDGE_QUANT") == "1":
                        from transformers import BitsAndBytesConfig

                        kwargs["model_kwargs"] = {
                            "quantization_config": BitsAndBytesConfig(load_in_4bit=True)
                        }
                elif torch.backends.mps.is_available():
                    kwargs["torch_dtype"], kwargs["device"] = torch.float16, "mps"
                else:
                    kwargs["torch_dtype"], kwargs["device"] = torch.float32, "cpu"
                _pipe = pipeline("image-text-to-text", model=JUDGE_MODEL, **kwargs)
    return _pipe


def _medgemma_run(messages: list) -> str:
    out = _load_pipe()(text=messages, max_new_tokens=MAX_NEW_TOKENS)
    return out[0]["generated_text"][-1]["content"].strip()


def _medgemma_transcribe(image_bytes: bytes) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": _pil(image_bytes)},
            {"type": "text", "text": TRANSCRIBE_PROMPT},
        ],
    }]
    return _medgemma_run(messages)


def _medgemma_judge(prompt: str) -> str:
    # Text-only — the judge works from the transcript, never the image.
    return _medgemma_run([{"role": "user", "content": [{"type": "text", "text": prompt}]}])


# ---------------------------------------------------------------- Claude (cloud)
def _claude_client():
    from anthropic import Anthropic

    return Anthropic(), os.getenv("JUDGE_CLAUDE_MODEL", os.getenv("CLAUDE_MODEL", "claude-opus-4-8"))


def _claude_transcribe(image_bytes: bytes) -> str:
    import base64
    import io

    client, model = _claude_client()
    buf = io.BytesIO()
    _pil(image_bytes).save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": TRANSCRIBE_PROMPT},
        ]}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _claude_judge(prompt: str) -> str:
    client, model = _claude_client()
    verdict_enum = {"type": "string", "enum": list(VERDICTS)}
    schema = {
        "type": "object",
        "properties": {k: {
            "type": "object",
            "properties": {"verdict": verdict_enum, "reason": {"type": "string"}},
            "required": ["verdict", "reason"],
            "additionalProperties": False,
        } for k in _FIELD_KEYS},
        "required": list(_FIELD_KEYS),
        "additionalProperties": False,
    }
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_NEW_TOKENS,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# ---------------------------------------------------------------- public API
def _normalize(raw: dict) -> dict:
    """Coerce the model's object into {key: {verdict, reason}} with valid verdicts."""
    result = {}
    for k in _FIELD_KEYS:
        entry = raw.get(k) if isinstance(raw, dict) else None
        if isinstance(entry, dict):
            verdict = str(entry.get("verdict", "")).strip().lower()
            reason = str(entry.get("reason", "")).strip()
        elif isinstance(entry, str):  # model returned just the verdict
            verdict, reason = entry.strip().lower(), ""
        else:
            verdict, reason = "uncertain", "judge gave no verdict for this field"
        if verdict not in VERDICTS:
            verdict, reason = "uncertain", reason or f"unrecognized verdict {verdict!r}"
        result[k] = {"verdict": verdict, "reason": reason}
    return result


def _salvage(text: str) -> dict:
    """Broken/truncated judge JSON: mark every field uncertain for human review."""
    return {k: {"verdict": "uncertain", "reason": "judge output unparseable"} for k in _FIELD_KEYS}


def transcribe_reference(image_bytes: bytes) -> str:
    """Strong-model transcript used as the grading reference. Deliberately a
    larger/different model than ocr.extract so we aren't grading the 4B against
    itself."""
    if PROVIDER == "claude":
        return _claude_transcribe(image_bytes)
    return _medgemma_transcribe(image_bytes)


def judge_note(reference: str, fields: dict) -> dict:
    """Grade each field against the reference transcript.

    Returns {field_key: {"verdict": <one of VERDICTS>, "reason": str}} for every
    field key. Never raises on bad model output — falls back to "uncertain" so a
    human can resolve it in the review UI.
    """
    prompt = _judge_prompt(reference, fields)
    raw = _claude_judge(prompt) if PROVIDER == "claude" else _medgemma_judge(prompt)
    try:
        return _normalize(_parse_json(raw))
    except Exception:  # noqa: BLE001 - malformed/truncated JSON: mark uncertain
        return _salvage(raw)


def free():
    """Release the local judge model so the GPU is free for the next phase."""
    global _pipe
    if _pipe is None:
        return
    _pipe = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass

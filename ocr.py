"""OCR + structured extraction — swappable provider.

Two backends, picked with OCR_PROVIDER:
  - "medgemma" (default): local MedGemma 1.5 4B via transformers. Zero config,
    nothing leaves the machine — the offline / data-sovereignty story for LMICs.
  - "claude": Claude vision via the Anthropic API. Better on messy handwriting
    and reliable JSON output; needs ANTHROPIC_API_KEY.

Public functions:
  transcribe(image_bytes) -> str             # plain transcription
  extract(image_bytes)    -> {text, fields}  # transcription + filled note fields
  warmup()                                   # preload the local model (no-op for claude)
"""

import base64
import io
import json
import os
import re
import threading

from PIL import Image

from data import NOTE_FIELDS

PROVIDER = os.getenv("OCR_PROVIDER", "medgemma").lower()
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))

_FIELD_KEYS = [key for key, _label, _kind in NOTE_FIELDS]

TRANSCRIBE_PROMPT = (
    "You are transcribing a handwritten clinical note from a physician. "
    "Transcribe ALL handwritten and printed text in this image exactly as written, "
    "preserving line breaks and reading order. Expand nothing, infer nothing, add no "
    "commentary. Rotate the image if needed to read it. If a word is illegible, write "
    "[illegible]. Output only the transcription."
)

_FIELD_TEMPLATE = ",\n".join(f'  "{k}": ""' for k in _FIELD_KEYS)
EXTRACT_PROMPT = (
    "You are reading a photographed handwritten clinical progress note.\n"
    "Step 1: transcribe the note exactly. Step 2: map it into a structured note.\n\n"
    "Return ONLY one JSON object (no markdown fences, no commentary) of this shape:\n"
    "{\n"
    '  "raw_transcript": "the full verbatim transcription",\n'
    f"{_FIELD_TEMPLATE}\n"
    "}\n\n"
    "Fill each field from the note's content. Map sections sensibly: Chief Complaint -> "
    "chief_complaint; HPI / history of present illness -> hpi; PMHx / past medical history "
    "-> pmhx; FMHx / family history -> fmhx; SHx / social history -> shx; ROS / review of "
    "systems -> ros; exam / PE / physical exam -> pe; assessment / impression -> assessment; "
    "plan / management -> plan; note_type is the kind of note if stated (else \"\").\n"
    "If a section is absent from the note, use an empty string. Do NOT invent clinical "
    "information. Output JSON only."
)


def _pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ---------------------------------------------------------------- MedGemma (local)
_MODEL_ID = "google/medgemma-1.5-4b-it"
_pipe = None
_pipe_lock = threading.Lock()


def _load_pipe():
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                import torch
                from transformers import pipeline

                if torch.cuda.is_available():
                    device, dtype = "cuda", torch.bfloat16
                elif torch.backends.mps.is_available():
                    device, dtype = "mps", torch.float16
                else:
                    device, dtype = "cpu", torch.float32
                _pipe = pipeline(
                    "image-text-to-text",
                    model=_MODEL_ID,
                    torch_dtype=dtype,
                    device=device,
                )
    return _pipe


def _medgemma_run(image_bytes: bytes, prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": _pil(image_bytes)},
            {"type": "text", "text": prompt},
        ],
    }]
    out = _load_pipe()(text=messages, max_new_tokens=MAX_NEW_TOKENS)
    return out[0]["generated_text"][-1]["content"].strip()


# ---------------------------------------------------------------- Claude (cloud)
def _claude_client():
    from anthropic import Anthropic

    return Anthropic(), os.getenv("CLAUDE_MODEL", "claude-opus-4-8")


def _claude_image_block(image_bytes: bytes) -> dict:
    buf = io.BytesIO()
    _pil(image_bytes).save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}


def _claude_run(image_bytes: bytes, prompt: str) -> str:
    client, model = _claude_client()
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": [_claude_image_block(image_bytes), {"type": "text", "text": prompt}]}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _claude_extract_json(image_bytes: bytes) -> str:
    """Claude with a JSON schema so the output is guaranteed-parseable."""
    client, model = _claude_client()
    schema = {
        "type": "object",
        "properties": {"raw_transcript": {"type": "string"},
                       **{k: {"type": "string"} for k in _FIELD_KEYS}},
        "required": ["raw_transcript", *_FIELD_KEYS],
        "additionalProperties": False,
    }
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": [_claude_image_block(image_bytes), {"type": "text", "text": EXTRACT_PROMPT}]}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# ---------------------------------------------------------------- public API
def _parse_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*", "", t).strip().rstrip("`").strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        t = t[i:j + 1]
    return json.loads(t)


def warmup():
    """Preload the local model so the first request doesn't pay the load cost."""
    if PROVIDER == "medgemma":
        try:
            _load_pipe()
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass


def transcribe(image_bytes: bytes) -> str:
    if PROVIDER == "claude":
        return _claude_run(image_bytes, TRANSCRIBE_PROMPT)
    return _medgemma_run(image_bytes, TRANSCRIBE_PROMPT)


def extract(image_bytes: bytes) -> dict:
    """One call: image -> {text: transcript, fields: {note_key: value}}.

    Falls back to {text: <raw output>, fields: {}} if the model's JSON can't be
    parsed, so the UI can still show the transcription.
    """
    if PROVIDER == "claude":
        raw = _claude_extract_json(image_bytes)
    else:
        raw = _medgemma_run(image_bytes, EXTRACT_PROMPT)

    try:
        data = _parse_json(raw)
    except Exception:  # noqa: BLE001 - degrade gracefully to raw transcript
        return {"text": raw, "fields": {}}

    fields = {k: str(data.get(k) or "").strip() for k in _FIELD_KEYS}
    transcript = str(data.get("raw_transcript") or "").strip() or raw
    return {"text": transcript, "fields": fields}

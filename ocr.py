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
# Upper bound on generated tokens. generate() requires *a* cap (without one,
# transformers falls back to max_length=20). It's a ceiling, not a target —
# the model stops at EOS when the JSON is done, so a high value is ~free.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))

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
    "Read this handwritten clinical note and return ONLY a JSON object — no thinking, "
    "no explanation, no markdown fences, nothing but the JSON. Use exactly these keys:\n"
    "{\n"
    '  "raw_transcript": "",\n'
    f"{_FIELD_TEMPLATE}\n"
    "}\n"
    'Set "raw_transcript" to the full verbatim transcription of the note. Fill each section '
    "field from the note's content: chief_complaint, hpi, pmhx, fmhx, shx, ros, pe, "
    "assessment, plan, and note_type (the kind of note if stated). If a section is absent, "
    'use "". Do not invent information. Your entire response must start with { and end with }.'
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


def _json_str(s: str) -> str:
    """Decode a captured JSON string body (handle escapes; tolerate raw newlines)."""
    try:
        return json.loads('"' + s + '"')
    except Exception:  # noqa: BLE001
        return s.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t').strip()


def _regex_value(text: str, key: str) -> str:
    """Pull one "key": "value" out of (possibly truncated/invalid) JSON-ish text."""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    return _json_str(m.group(1)).strip() if m else ""


def _salvage_fields(text: str) -> dict:
    return {k: _regex_value(text, k) for k in _FIELD_KEYS}


def warmup():
    """Preload the local model AND run a tiny generation so CUDA kernels are
    compiled at startup — the first real scan is then fast, not cold."""
    if PROVIDER != "medgemma":
        return
    try:
        pipe = _load_pipe()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (64, 64), "white")},
                {"type": "text", "text": "ok"},
            ],
        }]
        pipe(text=messages, max_new_tokens=1)
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
        fields = {k: str(data.get(k) or "").strip() for k in _FIELD_KEYS}
        transcript = str(data.get("raw_transcript") or "").strip()
    except Exception:  # noqa: BLE001 - JSON malformed/truncated: salvage per-key
        fields = _salvage_fields(raw)
        transcript = _regex_value(raw, "raw_transcript")

    if not transcript:
        transcript = "\n".join(v for v in fields.values() if v) or raw.strip()
    return {"text": transcript, "fields": fields}

"""OCR / handwriting transcription — swappable provider.

Two backends:
  - "medgemma" (default): local MedGemma 1.5 4B via transformers. Zero config,
    nothing leaves the machine — the offline / data-sovereignty story for LMICs.
  - "claude": Claude vision via the Anthropic API. Far better on messy
    handwriting today; needs ANTHROPIC_API_KEY. Use for live demos.

Pick with the OCR_PROVIDER env var. Both expose one function: transcribe(bytes) -> str.
"""

import base64
import io
import os
import threading

from PIL import Image

PROVIDER = os.getenv("OCR_PROVIDER", "medgemma").lower()

PROMPT = (
    "You are transcribing a handwritten clinical note from a physician. "
    "Transcribe ALL handwritten and printed text in this image exactly as written, "
    "preserving line breaks and reading order. Expand nothing, infer nothing, add no "
    "commentary. If a word is illegible, write [illegible]. Output only the transcription."
)

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


def _medgemma(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    out = _load_pipe()(text=messages, max_new_tokens=2000)
    return out[0]["generated_text"][-1]["content"].strip()


# ---------------------------------------------------------------- Claude (cloud)
def _claude(image_bytes: bytes) -> str:
    from anthropic import Anthropic

    # Normalise to PNG so we always send a known media type.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    client = Anthropic()
    model = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def transcribe(image_bytes: bytes) -> str:
    """Run the configured OCR provider on raw image bytes."""
    if PROVIDER == "claude":
        return _claude(image_bytes)
    return _medgemma(image_bytes)

# MediSnap EHR

A mock EHR with a **"scan handwritten note" microservice** bolted on. A physician
opens a patient's Medical Note, clicks **📷 Scan handwritten note**, a QR code
appears, they scan it with their phone, photograph the handwritten page, and the
transcribed text lands back on the computer screen — no typing.

The point: in many LMIC clinics staff still write notes by hand and re-type them
later (or never). This kills the re-typing step and lowers the barrier to going
digital — you photograph what you already write instead of starting from scratch.

This is **step 1**: get the desktop → QR → phone-camera → OCR → text-on-desktop
loop working on a real-looking EHR. Mapping the transcript into the individual
note fields is the next step.

## What's here

| File | Role |
|---|---|
| `app.py` | FastAPI app: EHR pages, scan-session API, QR + phone handoff |
| `ocr.py` | Swappable OCR: local **MedGemma** (default) or **Claude** vision |
| `data.py` | Mock patient + the medical-note field schema |
| `templates/` | Facesheet, Medical Note (+ scan modal), phone capture page |
| `tools/` | Standalone dev utilities (not part of the app) |

## Run

```sh
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 → **Medical Note** → **📷 Scan handwritten note**.

> Bind to `0.0.0.0` (not just localhost) so your phone can reach the server. The
> QR encodes your machine's **LAN IP**, so the phone must be on the **same
> Wi-Fi**. Check what it'll hand out: `curl localhost:8000/healthz`.

### OCR provider

- **MedGemma (default)** — runs locally, nothing leaves the machine. First photo
  triggers a one-time ~8GB model download and is slow on CPU/Mac. This is the
  offline / data-sovereignty story for LMIC clinics.
- **Claude vision** — far better on messy handwriting, returns in seconds. Best
  for a live demo.

  ```sh
  export ANTHROPIC_API_KEY=sk-ant-...
  OCR_PROVIDER=claude uv run uvicorn app:app --host 0.0.0.0 --port 8000
  # optional: CLAUDE_MODEL=claude-sonnet-4-6  (default claude-opus-4-8)
  ```

### Phone can't reach the laptop?

Conference/guest Wi-Fi often blocks phone → laptop. Use a tunnel and point the
QR at the public URL:

```sh
cloudflared tunnel --url http://localhost:8000   # or: ngrok http 8000
PUBLIC_BASE_URL=https://<your-tunnel-url> uv run uvicorn app:app --port 8000
```

## How the handoff works

```
desktop  POST /api/scan/session            -> {id, qr, mobile_url}
desktop  shows QR, polls GET /api/scan/session/{id}
phone    GET  /m/{id}                       -> camera capture page
phone    POST /api/scan/session/{id}/upload -> OCR runs in a background thread
desktop  poll sees status "done" + text
```

Scan sessions are in-memory (with a 30-min TTL) — fine for a demo, swap for a
store later.

## Architecture & roadmap

The pipeline is deliberately simple — **not** an agent:

```
photo ──▶ [vision model: transcribe / read]  ──▶  [structured extraction]  ──▶  note fields
          MedGemma (medical, on-device)            map transcript to a fixed
          or Claude vision                          schema (one constrained call)
```

- **Why MedGemma:** fine-tuned on Google's Health AI foundations (medical text,
  imaging, FHIR-aware) and runs **on-device** — the offline / data-sovereignty
  story for LMIC clinics. **Why Claude (swappable):** stronger on messy
  handwriting and reliable structured/JSON output; needs connectivity.
- **Next step:** the read step currently dumps the transcript into a panel. Map
  it into the individual note fields as **FHIR resources** (`Observation`,
  `Condition`, `MedicationStatement`, …) with per-field confidence. FHIR is the
  abstraction that makes "drop-in on any EHR" credible, and makes the output
  consumable by read-side agents (e.g. Google's MedGemma EHR Navigator).
- This is structured extraction, one call — reserve genuine agentic flows for
  later (drug-interaction checks, reconciling against the existing record).

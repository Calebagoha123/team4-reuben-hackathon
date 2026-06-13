# MediSnap

**Photograph a handwritten note or a printed lab report, and it lands in the patient's chart — structured, in seconds.**

In many clinics, especially in lower-resource settings, notes are still written
by hand and lab results arrive on paper. Someone then re-types all of it into the
computer later — slow, error-prone, and often skipped. MediSnap removes that step:
the clinician photographs what they already wrote (or the report they were
handed), and the information appears in the right place in the record.

---

## What it does

MediSnap is a mock electronic health record (EHR) with a **phone-camera scanner**
bolted on. You open a patient screen on the computer, tap the camera icon, and a
QR code appears. Scan it with your phone, take a photo, and the data shows up back
on the computer — no typing.

It handles two kinds of paper:

- **Handwritten clinical notes** → the chief complaint, history, exam, assessment
  and plan are read out and dropped into the matching fields of a progress note.
- **Printed lab reports** → the results are turned into a clean, editable table.

### Highlights

- 📷 **Scan with your phone** — a QR code hands the camera off to your phone; the
  photo never has to be saved or emailed anywhere.
- 🔄 **Rotate before sending** — upside-down or sideways photos happen. Spin the
  image on the phone before it's uploaded.
- 🗂️ **Single or batch** — scan one page, or photograph a whole stack at once. In a
  batch you **label each page** as a note or a lab report on the phone, and each
  one is filed in the correct tab automatically.
- 👤 **Finds the right patient** — the name / ID read off a note is matched against
  the patient list, and the note header fills in from the official record rather
  than from whatever the camera happened to read.
- 🧪 **Lab results as a table** — a printed report becomes a structured grid whose
  columns match the report itself. You can edit any cell, or add rows and columns
  by hand.
- 🌐 **Works online or fully offline** — two interchangeable text-reading engines
  (see below): one runs on the device with nothing leaving the machine, the other
  uses a cloud API. You choose.

---

## Try it

You'll need [uv](https://docs.astral.sh/uv/) (a fast Python package manager).

```sh
# 1. Install the app
uv sync

# 2. Run it (bind to 0.0.0.0 so your phone can reach it)
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, go to **Medical Note** or **Lab Reports**, click the
camera icon, and scan the QR code with your phone.

> **Phone can't connect?** The QR points at your computer's local network address,
> so the phone must be on the **same Wi-Fi**. On locked-down networks, use a tunnel
> (e.g. `cloudflared tunnel --url http://localhost:8000`) and set `PUBLIC_BASE_URL`
> to its URL — see [Configuration](#configuration).

### Choosing the text-reading engine

By default MediSnap uses the **local, on-device** model. For a fast demo on a
laptop without a GPU, the **cloud** option is smoother:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
OCR_PROVIDER=claude uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

To run the on-device model, install its (heavier) dependencies once with
`uv sync --extra medgemma` — details below.

---

## How it works

```
   PHONE                          SERVER                         DESKTOP
 ┌────────┐   photo + label    ┌──────────┐   read & match   ┌───────────┐
 │ camera │ ─────────────────▶ │  OCR +   │ ───────────────▶ │  note /   │
 │ rotate │                    │ structure│                  │ lab table │
 └────────┘ ◀── QR session ─── └──────────┘ ◀── poll ─────── └───────────┘
```

1. The desktop creates a short-lived **scan session** and shows its QR code.
2. The phone opens the session link, captures the photo(s), and uploads them.
3. The server reads each image, turns it into structured data, and (for notes)
   matches the patient.
4. The desktop polls the session and fills the form or table when it's ready.

Scan sessions live in memory with a 30-minute expiry — fine for a demo.

---

## Technical details

### The reading pipeline

The pipeline is deliberately simple — a fixed, two-step transformation, **not** an
agent:

```
photo ──▶ [vision model: read the image]  ──▶  [structure it]  ──▶  note fields
                                                                    or lab table
```

For lab reports the structuring step doesn't assume a fixed layout: the table's
columns are derived from whatever the report actually contains (a chemistry panel
has result/unit/range; a differential count adds percentage and absolute columns;
a microbiology line is free text). Extracted results are also emitted as a FHIR
`Observation` bundle server-side, so the data is portable to any FHIR-aware system.

### Text-reading engines (swappable)

Set with the `OCR_PROVIDER` environment variable:

| Provider | `OCR_PROVIDER` | Runs | Notes |
|---|---|---|---|
| **MedGemma** | `medgemma` (default) | On-device | Google's medical vision-language model. Nothing leaves the machine — the offline / data-sovereignty option. Heavy: needs the `medgemma` extra and, realistically, a GPU. |
| **Claude** | `claude` | Cloud API | Strong on messy handwriting and reliable structured output. Needs `ANTHROPIC_API_KEY` and connectivity. |

Both implement the same small interface (`extract`, `extract_labs`, `transcribe`),
so swapping is a one-line config change.

To use MedGemma, install its dependencies (PyTorch + Transformers, ~2GB+):

```sh
uv sync --extra medgemma
```

The first scan downloads the model weights (gated — set `HF_TOKEN`).

### Configuration

All via environment variables (or a `.env` file — see `.env.example`):

| Variable | Purpose |
|---|---|
| `OCR_PROVIDER` | `medgemma` (default) or `claude` |
| `ANTHROPIC_API_KEY` | Required for the Claude provider |
| `CLAUDE_MODEL` | Override the Claude model (default `claude-opus-4-8`) |
| `HF_TOKEN` | Hugging Face token for the gated MedGemma download |
| `PUBLIC_BASE_URL` | URL to encode in the QR (use with a tunnel when the phone can't reach the LAN) |

### Project layout

| Path | Role |
|---|---|
| `app.py` | FastAPI app: EHR pages, scan-session API, QR + phone handoff |
| `ocr.py` | Swappable text-reading: MedGemma (local) or Claude (cloud) |
| `patients.py` | Mock patient database + forgiving identifier matching |
| `labs.py` | Lab report → dynamic table + FHIR `Observation` bundle |
| `data.py` | Mock patient record + the medical-note field schema |
| `templates/` | Facesheet, Medical Note, Lab Reports, and the phone capture page |
| `tests/` | `pytest` suite — pure logic + API, with the model stubbed |
| `tools/` | Standalone dev utilities (not part of the app) |

### Tests

The suite stubs the vision model, so it runs in seconds with no GPU or API key:

```sh
uv run pytest
```

### Docker (on-device path)

`Dockerfile` builds a CUDA image for serving the local MedGemma model on a Linux
GPU host. Docker on macOS can't reach the Mac GPU — run with `uv` there instead.
(The Claude path needs no GPU and no special image.)

```sh
docker build -t medisnap .
docker run --rm --gpus all -p 8000:8000 -e HF_TOKEN=hf_... medisnap
```

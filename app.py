"""MediSnap EHR — mock EHR with phone-camera handoff for handwritten notes.

Step 1 of the build: a real-looking EHR (facesheet + medical note), a "Scan"
button on the note page that shows a QR code, a phone capture page reached by
scanning it, and the transcribed text appearing back on the desktop.

Flow:
  desktop  POST /api/scan/session          -> {id, qr, mobile_url}
  desktop  shows QR, polls GET /api/scan/session/{id}
  phone    GET  /m/{id}                     -> camera capture page
  phone    POST /api/scan/session/{id}/upload  (image) -> OCR runs in background
  desktop  poll sees status "done" + text

Sessions are in-memory; fine for a demo. The QR encodes the server's LAN URL so
a phone on the same Wi-Fi can reach it. Set PUBLIC_BASE_URL to use a tunnel
(ngrok/cloudflared) when Wi-Fi client isolation blocks phone -> laptop.
"""

import base64
import io
import os
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import qrcode

import labs
import ocr
import patients
from data import NOTE_FIELDS, PATIENT


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the local model in the background so the first scan is fast.
    if os.getenv("WARMUP", "1") == "1":
        threading.Thread(target=ocr.warmup, daemon=True).start()
    yield


app = FastAPI(title="MediSnap EHR", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# session_id -> {status, text, error, created}
# status: waiting -> uploaded -> processing -> done | error
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()
_SESSION_TTL = 30 * 60  # seconds; scan sessions are short-lived


def _prune_sessions():
    """Drop scan sessions older than the TTL so the in-memory store can't grow
    without bound. Called on each new session."""
    cutoff = time.time() - _SESSION_TTL
    with _sessions_lock:
        for sid in [s for s, v in _sessions.items() if v.get("created", 0) < cutoff]:
            _sessions.pop(sid, None)


def _lan_ip() -> str:
    """Best-effort LAN IP so a phone on the same network can reach us."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _base_url() -> str:
    explicit = os.getenv("PUBLIC_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    port = os.getenv("PORT", "8000")
    return f"http://{_lan_ip()}:{port}"


def _qr_data_uri(url: str) -> str:
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------- pages
@app.get("/")
async def index():
    return RedirectResponse("/facesheet")


@app.get("/facesheet", response_class=HTMLResponse)
async def facesheet(request: Request):
    return templates.TemplateResponse(
        request, "facesheet.html", {"patient": PATIENT, "active": "facesheet"}
    )


@app.get("/note", response_class=HTMLResponse)
async def note(request: Request):
    return templates.TemplateResponse(
        request,
        "note.html",
        {
            "patient": PATIENT,
            "active": "note",
            "fields": NOTE_FIELDS,
            "today": date.today().isoformat(),
        },
    )


@app.get("/labs", response_class=HTMLResponse)
async def lab_reports(request: Request):
    return templates.TemplateResponse(
        request,
        "labs.html",
        {"patient": PATIENT, "active": "labs", "today": date.today().isoformat()},
    )


@app.get("/m/{session_id}", response_class=HTMLResponse)
async def mobile_capture(request: Request, session_id: str):
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown or expired scan session.")
    return templates.TemplateResponse(
        request, "mobile.html",
        {"session_id": session_id, "mode": sess["mode"], "doc_type": sess["doc_type"]},
    )


# ----------------------------------------------------------------- scan API
_MODES = ("single", "batch")
_DOC_TYPES = ("note", "lab")


@app.post("/api/scan/session")
async def create_scan_session(request: Request):
    # Body is optional: {mode: single|batch, doc_type: note|lab}. The clinician
    # picks both up front on the desktop; every page in the scan is that type.
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - no/blank body
        body = {}
    mode = body.get("mode") if body.get("mode") in _MODES else "single"
    doc_type = body.get("doc_type") if body.get("doc_type") in _DOC_TYPES else "note"

    _prune_sessions()
    session_id = uuid.uuid4().hex[:10]
    with _sessions_lock:
        _sessions[session_id] = {
            "status": "waiting", "mode": mode, "doc_type": doc_type, "error": None,
            "created": time.time(),
            "records": None,  # filled on done: list of {doc_type, ...}
        }
    mobile_url = f"{_base_url()}/m/{session_id}"
    return {
        "id": session_id, "mode": mode, "doc_type": doc_type,
        "mobile_url": mobile_url, "qr": _qr_data_uri(mobile_url),
    }


@app.get("/api/scan/session/{session_id}")
async def scan_status(session_id: str):
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown scan session.")
    return sess


def _set(session_id: str, **kv):
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].update(**kv)


def _note_record(raw: bytes) -> dict:
    result = ocr.extract(raw)  # {text, fields, demographics}
    patient = patients.match_patient(result.get("demographics") or {})
    return {"doc_type": "note", **result, "patient": patient}


def _lab_record(raw: bytes) -> dict:
    report = labs.normalize_report(ocr.extract_labs(raw))
    return {"doc_type": "lab", "report": report, "fhir": labs.to_fhir_bundle(report)}


def _process(session_id: str, default_type: str, items: list[tuple[bytes, str]]):
    """Read each (image, label) into a record. The label is the per-photo type
    set on the phone in batch mode; blank falls back to the session default."""
    _set(session_id, status="processing")
    try:
        records = []
        for raw, label in items:
            doc_type = label if label in _DOC_TYPES else default_type
            records.append(_lab_record(raw) if doc_type == "lab" else _note_record(raw))
        _set(session_id, status="done", records=records)
    except Exception as e:  # noqa: BLE001 - surface to the UI
        _set(session_id, status="error", error=str(e))


@app.post("/api/scan/session/{session_id}/upload")
async def upload_photo(
    session_id: str,
    images: list[UploadFile] = File(...),
    types: list[str] = Form(default=[]),
):
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown or expired scan session.")
    raws = [await img.read() for img in images]
    # Pair each image with its per-photo label (blank if none sent); drop empties.
    items = [(raw, types[i] if i < len(types) else "")
             for i, raw in enumerate(raws) if raw]
    if not items:
        raise HTTPException(status_code=400, detail="Empty file.")

    _set(session_id, status="uploaded")
    # OCR can be slow (MedGemma) — run off the request thread; desktop polls.
    threading.Thread(
        target=_process, args=(session_id, sess["doc_type"], items), daemon=True,
    ).start()
    return JSONResponse({"ok": True, "count": len(items)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "ocr_provider": ocr.PROVIDER, "base_url": _base_url()}

"""API behaviour, with OCR stubbed so no model runs.

The clinician picks document type (note/lab) and mode (single/batch) on the
desktop; the session carries both and every page in the scan is read as that
type. The session returns a list of `records`, each tagged with `doc_type`.
Covers that, plus patient matching and the lab table + FHIR generation.
"""

import time

import ocr
import patients


def _wait_done(client, sid, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/scan/session/{sid}").json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"session {sid} did not finish: {data}")


def _files(png_bytes, n=1):
    return [("images", (f"p{i}.png", png_bytes, "image/png")) for i in range(n)]


def _records_by_type(data, t):
    return [r for r in data["records"] if r["doc_type"] == t]


# ------------------------------------------------------------------ pages
def test_pages_render(client):
    for path in ("/facesheet", "/note", "/labs"):
        assert client.get(path).status_code == 200


def test_lab_tab_present_in_nav(client):
    assert 'href="/labs"' in client.get("/note").text


def test_labs_page_shows_empty_baseline(client):
    assert "No lab data" in client.get("/labs").text


def test_mobile_page_reflects_mode(client):
    r = client.post("/api/scan/session", json={"mode": "batch"}).json()
    body = client.get(f"/m/{r['id']}").text
    assert "BATCH" in body
    assert 'const MODE = "batch"' in body


def test_mobile_page_unknown_session_404(client):
    assert client.get("/m/doesnotexist").status_code == 404


# ------------------------------------------------------------------ sessions
def test_session_defaults_to_note_single(client):
    r = client.post("/api/scan/session").json()
    assert r["mode"] == "single" and r["doc_type"] == "note"
    assert "qr" in r and "/m/" in r["mobile_url"]


def test_session_accepts_batch_and_lab(client):
    r = client.post("/api/scan/session", json={"mode": "batch", "doc_type": "lab"}).json()
    assert r["mode"] == "batch" and r["doc_type"] == "lab"
    st = client.get(f"/api/scan/session/{r['id']}").json()
    assert st["mode"] == "batch" and st["doc_type"] == "lab"


def test_upload_unknown_session_404(client, png_bytes):
    r = client.post("/api/scan/session/nope/upload", files=_files(png_bytes))
    assert r.status_code == 404


def test_upload_empty_400(client):
    sid = client.post("/api/scan/session").json()["id"]
    r = client.post(f"/api/scan/session/{sid}/upload",
                    files=[("images", ("e.png", b"", "image/png"))])
    assert r.status_code == 400


# ------------------------------------------------------------------ note scan
def test_note_scan_matches_patient(client, png_bytes, monkeypatch):
    target = patients.PATIENTS[0]
    monkeypatch.setattr(ocr, "extract", lambda raw: {
        "text": "CC: cough", "fields": {"chief_complaint": "cough"},
        "demographics": {"mrn": target["mrn"]}})
    sid = client.post("/api/scan/session", json={"doc_type": "note"}).json()["id"]
    client.post(f"/api/scan/session/{sid}/upload", files=_files(png_bytes))
    data = _wait_done(client, sid)

    note = _records_by_type(data, "note")[0]
    assert note["fields"]["chief_complaint"] == "cough"
    assert note["patient"]["mrn"] == target["mrn"]


def test_note_batch_returns_one_record_per_page(client, png_bytes, monkeypatch):
    monkeypatch.setattr(ocr, "extract", lambda raw: {
        "text": "n", "fields": {}, "demographics": {}})
    sid = client.post("/api/scan/session",
                      json={"mode": "batch", "doc_type": "note"}).json()["id"]
    client.post(f"/api/scan/session/{sid}/upload", files=_files(png_bytes, n=3))
    data = _wait_done(client, sid)
    assert len(_records_by_type(data, "note")) == 3


# ------------------------------------------------------------------ lab scan
def test_lab_scan_builds_table_and_fhir(client, png_bytes, monkeypatch):
    monkeypatch.setattr(ocr, "extract_labs", lambda raw: {
        "report_title": "CBC", "panels": [{"name": "CBC", "rows": [
            {"test": "Haemoglobin", "result": "11.9", "unit": "g/dL"}]}]})
    sid = client.post("/api/scan/session", json={"doc_type": "lab"}).json()["id"]
    client.post(f"/api/scan/session/{sid}/upload", files=_files(png_bytes))
    data = _wait_done(client, sid)

    lab = _records_by_type(data, "lab")[0]
    assert lab["report"]["title"] == "CBC"
    assert lab["report"]["panels"][0]["rows"][0]["test"] == "Haemoglobin"
    assert lab["fhir"]["resourceType"] == "Bundle"
    obs = [e["resource"] for e in lab["fhir"]["entry"]
           if e["resource"]["resourceType"] == "Observation"]
    assert obs[0]["code"]["text"] == "Haemoglobin"


# ------------------------------------------------------------------ mixed batch
def test_batch_can_mix_note_and_lab_by_per_photo_label(client, png_bytes, monkeypatch):
    # One note + one lab in a single batch, each labelled on the phone.
    monkeypatch.setattr(ocr, "extract", lambda raw: {
        "text": "n", "fields": {"chief_complaint": "fever"}, "demographics": {}})
    monkeypatch.setattr(ocr, "extract_labs", lambda raw: {
        "panels": [{"name": "CBC", "rows": [{"test": "Hgb", "result": "11"}]}]})
    sid = client.post("/api/scan/session",
                      json={"mode": "batch", "doc_type": "note"}).json()["id"]
    client.post(f"/api/scan/session/{sid}/upload",
                files=_files(png_bytes, n=2), data={"types": ["note", "lab"]})
    data = _wait_done(client, sid)

    assert len(data["records"]) == 2
    assert len(_records_by_type(data, "note")) == 1
    assert len(_records_by_type(data, "lab")) == 1


def test_unlabelled_pages_fall_back_to_session_default(client, png_bytes, monkeypatch):
    monkeypatch.setattr(ocr, "extract_labs", lambda raw: {
        "panels": [{"name": "P", "rows": [{"test": "Na", "result": "140"}]}]})
    sid = client.post("/api/scan/session",
                      json={"mode": "batch", "doc_type": "lab"}).json()["id"]
    client.post(f"/api/scan/session/{sid}/upload", files=_files(png_bytes, n=2))  # no types
    data = _wait_done(client, sid)
    assert len(_records_by_type(data, "lab")) == 2

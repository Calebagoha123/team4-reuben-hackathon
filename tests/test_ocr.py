"""OCR parsing layer (no model calls).

The provider calls (MedGemma / Claude) can't run in CI, but the JSON parsing,
demographic capture and salvage paths are pure functions over the model's text
output — and that's where the brittle bugs live, so that's what we test.
"""

import json

import ocr


def test_parse_extract_pulls_fields_and_demographics():
    raw = json.dumps({
        "raw_transcript": "CC: cough. A/P: URI.",
        "note_type": "Progress Note",
        "chief_complaint": "cough",
        "assessment": "URI",
        "patient": {"name": "Eliza Avocado", "mrn": "MICU200004",
                    "dob": "1979-09-10", "sex": "F"},
    })
    out = ocr._parse_extract(raw)
    assert out["text"] == "CC: cough. A/P: URI."
    assert out["fields"]["chief_complaint"] == "cough"
    assert out["fields"]["assessment"] == "URI"
    assert out["demographics"]["mrn"] == "MICU200004"
    assert out["demographics"]["name"] == "Eliza Avocado"


def test_parse_extract_handles_markdown_fences():
    raw = "```json\n" + json.dumps({
        "raw_transcript": "t", "chief_complaint": "fever",
        "patient": {"name": "X"}}) + "\n```"
    out = ocr._parse_extract(raw)
    assert out["fields"]["chief_complaint"] == "fever"


def test_parse_extract_salvages_truncated_json():
    # JSON cut off mid-stream (model hit token cap) — regex salvage per key.
    raw = ('{"raw_transcript": "partial", "chief_complaint": "headache", '
           '"patient": {"mrn": "OPD445120"')
    out = ocr._parse_extract(raw)
    assert out["fields"]["chief_complaint"] == "headache"
    assert out["demographics"]["mrn"] == "OPD445120"


def test_parse_extract_missing_patient_yields_blank_demographics():
    raw = json.dumps({"raw_transcript": "t", "chief_complaint": "c"})
    out = ocr._parse_extract(raw)
    assert out["demographics"] == {"name": "", "mrn": "", "dob": "", "sex": ""}


def test_parse_labs_returns_panels():
    raw = "```json\n" + json.dumps({
        "report_title": "CBC",
        "panels": [{"name": "CBC", "rows": [
            {"test": "Hgb", "result": "11.9", "unit": "g/dL"}]}],
    }) + "\n```"
    out = ocr._parse_labs(raw)
    assert out["report_title"] == "CBC"
    assert out["panels"][0]["rows"][0]["test"] == "Hgb"


def test_parse_labs_bad_json_returns_empty_dict():
    assert ocr._parse_labs("the model rambled and produced no json") == {}


def test_demographic_keys_exist():
    assert ocr.DEMOGRAPHIC_KEYS == ["name", "mrn", "dob", "sex"]

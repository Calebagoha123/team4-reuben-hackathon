"""Lab report -> structured table + FHIR.

A lab report photo is read by the vision model into a loose JSON shape. The
report's *layout is not fixed* — a chemistry panel has result/unit/range, a
differential count adds percentage and absolute columns, a microbiology result
is free text. So we don't hard-code columns: normalize_report() derives them
from whichever fields the model actually returned, dropping any that are empty
across the whole panel. The result is a render-ready table.

to_fhir_bundle() then emits one FHIR `Observation` per result row (with
`valueQuantity`/`referenceRange`/`interpretation` where parseable), wrapped in a
collection `Bundle` alongside a `Patient` resource — so the extracted data is
portable to any FHIR-aware system, not just our UI.
"""

import re

# Canonical column order + display labels. Any field a report carries that isn't
# listed here still shows up (after these), title-cased — so we never silently
# drop information the model read.
_COLUMN_LABELS = {
    "test": "Test",
    "result": "Result",
    "result_pct": "Result %",
    "result_abs": "Abs. Count",
    "unit": "Unit",
    "ref_range": "Ref. Range",
    "ref_range_abs": "Ref. Range (Abs)",
    "flag": "Flag",
}
_PREFERRED = list(_COLUMN_LABELS)


def _label(key: str) -> str:
    return _COLUMN_LABELS.get(key, key.replace("_", " ").title())


def _clean(v) -> str:
    return str(v if v is not None else "").strip()


def _order_keys(keys: set[str]) -> list[str]:
    ordered = [k for k in _PREFERRED if k in keys]
    extras = sorted(k for k in keys if k not in _COLUMN_LABELS)
    return ordered + extras


def _normalize_panel(panel: dict) -> dict | None:
    raw_rows = panel.get("rows") or []
    rows: list[dict] = []
    for r in raw_rows:
        if not isinstance(r, dict):
            continue
        row = {k: _clean(v) for k, v in r.items()}
        if any(row.get(k) for k in row):  # skip wholly empty rows
            rows.append(row)
    if not rows:
        return None

    # A column survives only if at least one row has a value for it. "flag" is
    # carried on rows for styling but never shown as its own column.
    present = {
        k for row in rows for k, v in row.items() if v and k != "flag"
    }
    keys = _order_keys(present)
    return {
        "name": _clean(panel.get("name")),
        "keys": keys,
        "columns": [_label(k) for k in keys],
        "rows": rows,
    }


def normalize_report(data: dict) -> dict:
    """Loose model JSON -> clean, render-ready report. Never raises on bad input."""
    data = data or {}
    panels_in = data.get("panels")
    if not panels_in and data.get("rows"):
        # Model returned a single flat table — wrap it.
        panels_in = [{"name": data.get("report_title") or "", "rows": data["rows"]}]
    panels_in = panels_in or []

    panels = []
    for p in panels_in:
        if isinstance(p, dict):
            np = _normalize_panel(p)
            if np:
                panels.append(np)

    patient = data.get("patient") or {}
    return {
        "title": _clean(data.get("report_title") or data.get("title")),
        "report_date": _clean(data.get("report_date")),
        "patient": {k: _clean(patient.get(k)) for k in
                    ("name", "mrn", "dob", "sex", "age")},
        "panels": panels,
    }


# ------------------------------------------------------------------- FHIR
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_RANGE_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[-–to]+\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
# HL7 v3 ObservationInterpretation codes.
_FLAG_MAP = {
    "h": ("H", "High"), "high": ("H", "High"),
    "l": ("L", "Low"), "low": ("L", "Low"),
    "hh": ("HH", "Critical high"), "ll": ("LL", "Critical low"),
    "c": ("A", "Abnormal"), "crit": ("AA", "Critical abnormal"),
    "a": ("A", "Abnormal"), "n": ("N", "Normal"), "normal": ("N", "Normal"),
}


def _as_number(s: str):
    m = _NUM_RE.fullmatch(s.strip())
    if not m:
        return None
    val = float(s)
    return int(val) if val.is_integer() else val


def _quantity(value: float, unit: str) -> dict:
    q = {"value": value}
    if unit:
        q["unit"] = unit
    return q


def _observation(row: dict, panel_name: str, subject_ref: str | None) -> dict | None:
    name = row.get("test")
    if not name:
        return None
    obs: dict = {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
            "code": "laboratory", "display": "Laboratory",
        }]}],
        "code": {"text": name},
    }
    if subject_ref:
        obs["subject"] = {"reference": subject_ref}
    if panel_name:
        obs["code"]["coding"] = [{"display": f"{panel_name}: {name}"}]

    unit = row.get("unit", "")
    result = row.get("result") or row.get("result_abs") or row.get("result_pct") or ""
    num = _as_number(result)
    if num is not None:
        obs["valueQuantity"] = _quantity(num, unit)
    elif result:
        obs["valueString"] = result

    rng = _RANGE_RE.search(row.get("ref_range", ""))
    if rng:
        low, high = _as_number(rng.group(1)), _as_number(rng.group(2))
        ref = {}
        if low is not None:
            ref["low"] = _quantity(low, unit)
        if high is not None:
            ref["high"] = _quantity(high, unit)
        if ref:
            obs["referenceRange"] = [ref]

    flag = (row.get("flag") or "").strip().lower()
    if flag in _FLAG_MAP:
        code, display = _FLAG_MAP[flag]
        obs["interpretation"] = [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
            "code": code, "display": display,
        }]}]
    return obs


def to_fhir_bundle(report: dict) -> dict:
    """Normalized report -> FHIR collection Bundle (Patient + Observations)."""
    entries: list[dict] = []
    subject_ref = None

    pat = report.get("patient") or {}
    if any(pat.values()):
        patient_res = {"resourceType": "Patient"}
        if pat.get("name"):
            patient_res["name"] = [{"text": pat["name"]}]
        if pat.get("mrn"):
            patient_res["identifier"] = [{"value": pat["mrn"]}]
        if pat.get("dob"):
            patient_res["birthDate"] = pat["dob"]
        sex = (pat.get("sex") or "").strip().lower()
        if sex in ("m", "male"):
            patient_res["gender"] = "male"
        elif sex in ("f", "female"):
            patient_res["gender"] = "female"
        subject_ref = "Patient/extracted"
        patient_res["id"] = "extracted"
        entries.append({"resource": patient_res})

    for panel in report.get("panels", []):
        for row in panel.get("rows", []):
            obs = _observation(row, panel.get("name", ""), subject_ref)
            if obs:
                entries.append({"resource": obs})

    return {"resourceType": "Bundle", "type": "collection", "entry": entries}

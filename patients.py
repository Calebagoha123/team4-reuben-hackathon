"""Mock clinic patient database + identifier matching.

In a real deployment this is a query against the EHR's patient index (or an FHIR
`Patient` search). Here it's an in-memory list — enough to prove the flow: the
OCR reads identifiers off a handwritten note, match_patient() resolves them to a
canonical record, and the note header is filled from that record rather than from
whatever the model happened to read.

Matching is deliberately forgiving because the *input* is messy (handwriting,
OCR noise):
  1. MRN is the strongest signal — compared on alphanumerics only, so
     "MRN: 00 12-34" still matches "001234".
  2. Otherwise name + DOB. Names are normalised so "Doe, John" == "John Doe".
  3. Otherwise a unique name match alone.
"""

import re

# Each record mirrors the shape of data.PATIENT so it can populate the header
# directly. Demographics only — the clinical chart is filled from the note.
PATIENTS: list[dict] = [
    {
        "name": "Eliza Avocado", "mrn": "MICU200004", "dob": "1979-09-10",
        "age": "44", "sex": "F", "location": "MICU", "provider": "Dr. A. Reuben",
    },
    {
        "name": "Omar Haddad", "mrn": "PED118842", "dob": "2017-10-11",
        "age": "2", "sex": "M", "location": "Pediatrics", "provider": "Dr. S. Nasser",
    },
    {
        "name": "Maria Santos", "mrn": "OPD445120", "dob": "1990-03-22",
        "age": "36", "sex": "F", "location": "Outpatient", "provider": "Dr. L. Okoye",
    },
    {
        "name": "John Mwangi", "mrn": "OPD300915", "dob": "1962-07-04",
        "age": "63", "sex": "M", "location": "Outpatient", "provider": "Dr. L. Okoye",
    },
    {
        "name": "Fatima Bello", "mrn": "ANC771203", "dob": "1998-12-30",
        "age": "27", "sex": "F", "location": "Antenatal", "provider": "Dr. P. Adeyemi",
    },
]


def _norm_mrn(s: str) -> str:
    """Keep alphanumerics only, uppercased — defeats OCR spacing/punctuation noise.

    Also drops a leading MRN/ID/MR label that OCR commonly glues onto the value
    (e.g. "MRN: 200004" -> "200004").
    """
    key = re.sub(r"[^A-Za-z0-9]", "", s or "").upper()
    return re.sub(r"^(?:MRN|MR|ID)(?=[A-Z0-9])", "", key)


def _norm_name(s: str) -> str:
    """Order-independent, punctuation-free name key: 'Doe, John' -> 'doe john'."""
    tokens = re.findall(r"[A-Za-z]+", (s or "").lower())
    return " ".join(sorted(tokens))


def find_by_mrn(mrn: str) -> dict | None:
    key = _norm_mrn(mrn)
    if not key:
        return None
    return next((p for p in PATIENTS if _norm_mrn(p["mrn"]) == key), None)


def match_patient(identifiers: dict) -> dict | None:
    """Resolve loose OCR identifiers to a database record, or None.

    `identifiers` may contain any of: mrn, name, dob, sex. Missing/blank values
    are ignored.
    """
    identifiers = identifiers or {}
    mrn = (identifiers.get("mrn") or "").strip()
    name = (identifiers.get("name") or "").strip()
    dob = (identifiers.get("dob") or "").strip()

    # 1. MRN — strongest signal, trusted over a conflicting name.
    if mrn:
        hit = find_by_mrn(mrn)
        if hit:
            return hit

    if not name:
        return None

    name_key = _norm_name(name)
    if not name_key:
        return None
    name_hits = [p for p in PATIENTS if _norm_name(p["name"]) == name_key]

    # 2. name + DOB.
    if dob:
        for p in name_hits:
            if p["dob"] == dob:
                return p

    # 3. unique name match.
    if len(name_hits) == 1:
        return name_hits[0]

    return None

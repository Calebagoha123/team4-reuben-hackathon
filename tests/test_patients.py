"""Patient database + matching.

The OCR step reads whatever identifiers it can off the note (name, MRN, DOB,
sex). match_patient() takes those and pulls the right record out of the clinic's
patient database so the note header can be populated from canonical data.
"""

import patients


def test_db_is_nonempty_and_well_formed():
    assert patients.PATIENTS, "expected a seeded patient database"
    for p in patients.PATIENTS:
        for key in ("name", "mrn", "dob", "sex"):
            assert key in p, f"patient missing {key}: {p}"


def test_match_by_exact_mrn():
    target = patients.PATIENTS[0]
    got = patients.match_patient({"mrn": target["mrn"]})
    assert got is not None
    assert got["mrn"] == target["mrn"]


def test_match_by_mrn_ignores_formatting_and_label():
    target = patients.PATIENTS[0]
    # OCR often reads "MRN: 00 12-34" with stray punctuation/spacing/prefix.
    noisy = "MRN: " + " ".join(target["mrn"])
    got = patients.match_patient({"mrn": noisy})
    assert got is not None and got["mrn"] == target["mrn"]


def test_match_by_name_and_dob_when_mrn_missing():
    target = patients.PATIENTS[0]
    got = patients.match_patient({"name": target["name"], "dob": target["dob"]})
    assert got is not None and got["mrn"] == target["mrn"]


def test_match_handles_last_comma_first_name_order():
    target = patients.PATIENTS[0]
    first, last = target["name"].split()[0], target["name"].split()[-1]
    got = patients.match_patient({"name": f"{last}, {first}", "dob": target["dob"]})
    assert got is not None and got["mrn"] == target["mrn"]


def test_name_only_match_when_unique():
    target = patients.PATIENTS[0]
    got = patients.match_patient({"name": target["name"]})
    assert got is not None and got["mrn"] == target["mrn"]


def test_no_match_returns_none():
    assert patients.match_patient({"mrn": "ZZZ999", "name": "Nobody Here"}) is None


def test_empty_identifiers_returns_none():
    assert patients.match_patient({}) is None
    assert patients.match_patient({"name": "", "mrn": "", "dob": ""}) is None


def test_mrn_beats_a_conflicting_name():
    # MRN is the strongest signal: if it matches, trust it over a wrong name.
    target = patients.PATIENTS[0]
    got = patients.match_patient({"mrn": target["mrn"], "name": "Totally Different"})
    assert got is not None and got["mrn"] == target["mrn"]

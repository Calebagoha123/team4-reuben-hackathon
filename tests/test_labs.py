"""Lab-report extraction: model JSON -> dynamic table + FHIR.

A photographed lab report (e.g. a CBC) is read into a loose JSON shape by the
vision model. labs.normalize_report() turns that into a clean, render-ready
table whose columns are driven by what's actually in the report, and
labs.to_fhir_bundle() emits FHIR Observations so the output is portable.
"""

import labs


# A realistic CBC fragment, like the alfa Laboratories report (Image #3).
RAW = {
    "report_title": "Complete Blood Picture (CBC)",
    "report_date": "2019-10-11",
    "patient": {"name": "", "mrn": "", "dob": "", "sex": "", "age": "2"},
    "panels": [
        {
            "name": "CBC",
            "rows": [
                {"test": "Haemoglobin", "result": "11.90", "unit": "g/dL",
                 "ref_range": "11 - 14", "flag": ""},
                {"test": "MCV", "result": "73.7", "unit": "fL",
                 "ref_range": "75 - 87", "flag": "L"},
                {"test": "Platelet Count", "result": "450", "unit": "x10^3/uL",
                 "ref_range": "200 - 490", "flag": ""},
            ],
        },
    ],
}


def test_empty_report_has_no_panels():
    rep = labs.normalize_report({})
    assert rep["panels"] == []
    assert rep["title"] == ""


def test_normalize_keeps_title_and_date():
    rep = labs.normalize_report(RAW)
    assert rep["title"] == "Complete Blood Picture (CBC)"
    assert rep["report_date"] == "2019-10-11"


def test_columns_are_derived_from_present_fields():
    rep = labs.normalize_report(RAW)
    panel = rep["panels"][0]
    # test/result/unit/ref_range present -> those columns, in canonical order.
    assert panel["keys"] == ["test", "result", "unit", "ref_range"]
    assert panel["columns"] == ["Test", "Result", "Unit", "Ref. Range"]


def test_empty_columns_are_dropped():
    # A report with no units at all should not get a Unit column.
    raw = {"panels": [{"name": "P", "rows": [
        {"test": "Glucose", "result": "90", "unit": "", "ref_range": ""},
    ]}]}
    panel = labs.normalize_report(raw)["panels"][0]
    assert panel["keys"] == ["test", "result"]


def test_extra_image_specific_columns_appear():
    # The differential count carries percentage AND absolute columns (Image #3).
    raw = {"panels": [{"name": "Differential", "rows": [
        {"test": "Lymphocytes", "result_pct": "35", "result_abs": "2.8",
         "unit": "x10^3/uL", "ref_range": "3.5 - 8"},
    ]}]}
    panel = labs.normalize_report(raw)["panels"][0]
    assert "result_pct" in panel["keys"]
    assert "result_abs" in panel["keys"]


def test_rows_round_trip_values():
    panel = labs.normalize_report(RAW)["panels"][0]
    hgb = panel["rows"][0]
    assert hgb["test"] == "Haemoglobin"
    assert hgb["result"] == "11.90"
    assert hgb["flag"] == ""


def test_bare_rows_without_panels_are_wrapped():
    raw = {"report_title": "Chem", "rows": [
        {"test": "Sodium", "result": "136", "unit": "mmol/L"}]}
    rep = labs.normalize_report(raw)
    assert len(rep["panels"]) == 1
    assert rep["panels"][0]["rows"][0]["test"] == "Sodium"


def test_to_fhir_bundle_is_a_collection_of_observations():
    bundle = labs.to_fhir_bundle(labs.normalize_report(RAW))
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    obs = [e["resource"] for e in bundle["entry"]
           if e["resource"]["resourceType"] == "Observation"]
    assert len(obs) == 3
    o = obs[0]
    assert o["status"] == "final"
    assert o["category"][0]["coding"][0]["code"] == "laboratory"
    assert o["code"]["text"] == "Haemoglobin"


def test_fhir_numeric_result_becomes_valuequantity():
    o = _first_obs(labs.to_fhir_bundle(labs.normalize_report(RAW)))
    assert o["valueQuantity"]["value"] == 11.90
    assert o["valueQuantity"]["unit"] == "g/dL"


def test_fhir_low_flag_becomes_interpretation():
    bundle = labs.to_fhir_bundle(labs.normalize_report(RAW))
    mcv = next(e["resource"] for e in bundle["entry"]
               if e["resource"].get("code", {}).get("text") == "MCV")
    code = mcv["interpretation"][0]["coding"][0]["code"]
    assert code == "L"


def test_fhir_reference_range_parsed():
    o = _first_obs(labs.to_fhir_bundle(labs.normalize_report(RAW)))
    rng = o["referenceRange"][0]
    assert rng["low"]["value"] == 11.0
    assert rng["high"]["value"] == 14.0


def test_non_numeric_result_becomes_valuestring():
    raw = {"panels": [{"name": "Micro", "rows": [
        {"test": "Blood Culture", "result": "No growth"}]}]}
    o = _first_obs(labs.to_fhir_bundle(labs.normalize_report(raw)))
    assert o["valueString"] == "No growth"
    assert "valueQuantity" not in o


def _first_obs(bundle):
    return next(e["resource"] for e in bundle["entry"]
               if e["resource"]["resourceType"] == "Observation")

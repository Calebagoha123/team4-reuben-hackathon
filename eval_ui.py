"""Review UI for the eval pipeline — a thin reader over eval/runs/*.

Lets clinicians scroll the judged notes one image at a time, see the fields the
4B filled with any non-`correct` verdict highlighted red, and agree with or
override the 27B judge. Their verdicts are written back to judgments.json, which
feeds the judge-vs-human agreement metric and accumulates a gold set.

Mounted on the main app: `app.include_router(eval_ui.router)`. Reads the same
artifacts the pipeline writes; never loads a model itself.
"""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import eval_pipeline as ep
from data import NOTE_FIELDS
from judge import VERDICTS

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# verdicts a human can assign (drop "uncertain" — a clinician always has an opinion)
HUMAN_VERDICTS = [v for v in VERDICTS if v != "uncertain"]


def _run_dir(run_id: str):
    d = ep.RUNS_DIR / run_id
    if not d.is_dir():
        raise HTTPException(404, f"Unknown run {run_id}")
    return d


def _load_run(run_id: str):
    d = _run_dir(run_id)
    preds = ep._load_json(d / "predictions.json", [])
    judgments = ep._load_json(d / "judgments.json", [])
    return preds, judgments


@router.get("/eval", response_class=HTMLResponse)
async def eval_index(request: Request):
    runs = []
    if ep.RUNS_DIR.is_dir():
        for d in sorted((p for p in ep.RUNS_DIR.iterdir() if p.is_dir()), reverse=True):
            preds, judgments = _load_run(d.name)
            summary = ep.aggregate(preds, judgments) if judgments else None
            runs.append({"id": d.name, "n": len(judgments), "summary": summary})
    return templates.TemplateResponse(request, "eval_index.html", {"runs": runs})


@router.get("/eval/{run_id}", response_class=HTMLResponse)
async def eval_run(request: Request, run_id: str):
    preds, judgments = _load_run(run_id)
    summary = ep.aggregate(preds, judgments) if judgments else None
    return templates.TemplateResponse(
        request, "eval_index.html",
        {"runs": [{"id": run_id, "n": len(judgments), "summary": summary, "open": True}]},
    )


@router.get("/eval/{run_id}/{idx}", response_class=HTMLResponse)
async def eval_image(request: Request, run_id: str, idx: int):
    preds, judgments = _load_run(run_id)
    if not judgments or idx < 0 or idx >= len(judgments):
        raise HTTPException(404, "No such image index in this run")
    j = judgments[idx]
    fields = next((p.get("fields", {}) for p in preds if p["image"] == j["image"]), {})
    ref = ep._ref_path(ep.IMAGES_DIR / j["image"])
    reference = ref.read_text() if ref.is_file() else ""
    human = j.get("human", {})

    rows = []
    for key, label, kind in NOTE_FIELDS:
        v = (j.get("verdicts", {}).get(key) or {})
        verdict = v.get("verdict", "uncertain")
        rows.append({
            "key": key, "label": label, "kind": kind,
            "value": str(fields.get(key, "") or ""),
            "verdict": verdict,
            "reason": v.get("reason", ""),
            "human": human.get(key, ""),
            "wrong": verdict not in ("correct",),
        })
    summary = ep.aggregate(preds, judgments)
    img_acc = next((pi["accuracy"] for pi in summary["per_image"] if pi["image"] == j["image"]), None)
    return templates.TemplateResponse(request, "eval_image.html", {
        "run_id": run_id, "idx": idx, "n": len(judgments),
        "image": j["image"], "reference": reference, "rows": rows,
        "human_verdicts": HUMAN_VERDICTS, "img_acc": img_acc,
        "prev": idx - 1 if idx > 0 else None,
        "next": idx + 1 if idx < len(judgments) - 1 else None,
    })


@router.get("/eval/{run_id}/{idx}/image")
async def eval_image_file(run_id: str, idx: int):
    _, judgments = _load_run(run_id)
    if not judgments or idx < 0 or idx >= len(judgments):
        raise HTTPException(404, "No such image")
    path = ep.IMAGES_DIR / judgments[idx]["image"]
    if not path.is_file():
        raise HTTPException(404, "Image file missing")
    return FileResponse(path)


@router.post("/api/eval/{run_id}/{idx}/review")
async def save_review(run_id: str, idx: int, request: Request):
    """Persist one clinician verdict: body {"field": <key>, "verdict": <verdict>}.
    Sending verdict "" clears the human verdict for that field."""
    body = await request.json()
    field, verdict = body.get("field"), body.get("verdict", "")
    if field is None:
        raise HTTPException(400, "Missing 'field'")
    if verdict and verdict not in VERDICTS:
        raise HTTPException(400, f"Bad verdict {verdict!r}")

    path = _run_dir(run_id) / "judgments.json"
    judgments = ep._load_json(path, [])
    if idx < 0 or idx >= len(judgments):
        raise HTTPException(404, "No such image index")
    human = judgments[idx].setdefault("human", {})
    if verdict:
        human[field] = verdict
    else:
        human.pop(field, None)
    path.write_text(json.dumps(judgments, indent=2, ensure_ascii=False))

    # return live agreement so the page can update its header without a reload
    preds = ep._load_json(_run_dir(run_id) / "predictions.json", [])
    summary = ep.aggregate(preds, judgments)
    return JSONResponse({
        "ok": True,
        "judge_human_agreement": summary["judge_human_agreement"],
        "human_reviewed_fields": summary["human_reviewed_fields"],
    })

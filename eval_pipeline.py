"""Evaluation pipeline: grade MedGemma-4B field extraction with MedGemma-27B.

Measures the two failure modes we keep hitting — the 4B over-populating fields
(hallucination) and mis-routing content into the wrong section — so the
extraction prompt can be tuned against numbers instead of vibes.

Layout (all under ./eval):
  eval/images/<name>.jpg          drop note photos here
  eval/refs/<name>.txt            trusted transcript (auto-made, clinician-editable)
  eval/runs/<run_id>/predictions.json   what the 4B extracted
  eval/runs/<run_id>/judgments.json     the 27B's per-field verdicts (+ human review)
  eval/runs/<run_id>/summary.json       aggregate metrics

Phased so the 4B and 27B are never resident at the same time (single-GPU VRAM):
  bootstrap -> extract -> (free 4B) -> judge (load 27B) -> score

Usage (on the VM):
  uv run --no-sync python eval_pipeline.py run          # all phases, new run_id
  uv run --no-sync python eval_pipeline.py bootstrap     # just (re)build refs
  uv run --no-sync python eval_pipeline.py score <run_id>  # recompute metrics

Provider knobs (env, mirror the app): OCR_PROVIDER for the 4B, JUDGE_PROVIDER /
JUDGE_MODEL / JUDGE_QUANT for the judge. --provider sets JUDGE_PROVIDER inline.
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "eval"
IMAGES_DIR = EVAL_DIR / "images"
REFS_DIR = EVAL_DIR / "refs"
RUNS_DIR = EVAL_DIR / "runs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


# ---------------------------------------------------------------- helpers
def _images() -> list[Path]:
    if not IMAGES_DIR.is_dir():
        return []
    return sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _ref_path(image: Path) -> Path:
    return REFS_DIR / f"{image.stem}.txt"


def _load_json(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _latest_run() -> str | None:
    if not RUNS_DIR.is_dir():
        return None
    runs = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())
    return runs[-1] if runs else None


# ---------------------------------------------------------------- phases
def bootstrap() -> int:
    """Make a trusted transcript for any image that lacks one. Existing refs are
    left alone so clinician corrections persist across runs."""
    import judge

    REFS_DIR.mkdir(parents=True, exist_ok=True)
    images = _images()
    if not images:
        print(f"No images in {IMAGES_DIR} — add some note photos first.", file=sys.stderr)
        return 1
    made = 0
    for img in images:
        ref = _ref_path(img)
        if ref.is_file() and ref.read_text().strip():
            continue
        print(f"[ref] transcribing {img.name} …", flush=True)
        try:
            ref.write_text(judge.transcribe_reference(img.read_bytes()).strip() + "\n")
            made += 1
        except Exception as e:  # noqa: BLE001 - record + continue
            print(f"    failed: {e}", file=sys.stderr)
    judge.free()
    print(f"Bootstrapped {made} new reference transcript(s); {len(images)} image(s) total.")
    return 0


def extract(run_id: str) -> int:
    """Run the 4B extractor over every image -> predictions.json, then free it."""
    import ocr

    images = _images()
    if not images:
        print(f"No images in {IMAGES_DIR}.", file=sys.stderr)
        return 1
    preds = []
    for img in images:
        print(f"[extract] {img.name} (OCR_PROVIDER={ocr.PROVIDER}) …", flush=True)
        try:
            out = ocr.extract(img.read_bytes())
            preds.append({"image": img.name, "fields": out["fields"], "raw_transcript": out["text"]})
        except Exception as e:  # noqa: BLE001
            print(f"    failed: {e}", file=sys.stderr)
            preds.append({"image": img.name, "fields": {}, "raw_transcript": "", "error": str(e)})
    _write_json(RUNS_DIR / run_id / "predictions.json", preds)

    # Free the 4B so the 27B judge has the GPU to itself.
    ocr._pipe = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    print(f"Extracted {len(preds)} note(s) -> {run_id}/predictions.json")
    return 0


def judge_run(run_id: str) -> int:
    """Grade each prediction against its reference transcript -> judgments.json."""
    import judge

    preds = _load_json(RUNS_DIR / run_id / "predictions.json", None)
    if preds is None:
        print(f"No predictions for run {run_id}; run `extract` first.", file=sys.stderr)
        return 1
    judgments = []
    for p in preds:
        img = p["image"]
        ref = _ref_path(IMAGES_DIR / img)
        reference = ref.read_text().strip() if ref.is_file() else ""
        if not reference:
            print(f"[judge] {img}: NO reference transcript — skipping (run bootstrap).", file=sys.stderr)
            verdicts = {k: {"verdict": "uncertain", "reason": "no reference transcript"}
                        for k in judge._FIELD_KEYS}
        else:
            print(f"[judge] {img} (JUDGE_PROVIDER={judge.PROVIDER}) …", flush=True)
            verdicts = judge.judge_note(reference, p.get("fields", {}))
        judgments.append({"image": img, "verdicts": verdicts, "human": {}})
    # Preserve any human review already recorded for this run.
    _merge_human(RUNS_DIR / run_id / "judgments.json", judgments)
    _write_json(RUNS_DIR / run_id / "judgments.json", judgments)
    judge.free()
    print(f"Judged {len(judgments)} note(s) -> {run_id}/judgments.json")
    return 0


def _merge_human(path: Path, judgments: list) -> None:
    """Carry forward clinician verdicts if judgments.json already existed."""
    old = _load_json(path, None)
    if not old:
        return
    by_image = {j["image"]: j.get("human", {}) for j in old}
    for j in judgments:
        if by_image.get(j["image"]):
            j["human"] = by_image[j["image"]]


# ---------------------------------------------------------------- scoring
def aggregate(predictions: list, judgments: list) -> dict:
    """Roll per-field verdicts up into the metrics we tune against. Importable so
    the review UI can recompute live as clinicians add human verdicts."""
    from ocr import _FIELD_KEYS

    fields_by_image = {p["image"]: p.get("fields", {}) for p in predictions}
    n_correct = n_total = n_filled = n_halluc = n_misroute = n_wrong = n_missing = 0
    n_caught = 0  # filled & correct -> the recall numerator
    human_total = human_agree = 0
    per_image = []

    for j in judgments:
        fields = fields_by_image.get(j["image"], {})
        verdicts = j.get("verdicts", {})
        human = j.get("human", {})
        img_correct = 0
        for k in _FIELD_KEYS:
            n_total += 1
            v = (verdicts.get(k) or {}).get("verdict", "uncertain")
            filled = bool(str(fields.get(k, "") or "").strip())
            if filled:
                n_filled += 1
            if v == "correct":
                n_correct += 1
                img_correct += 1
                if filled:
                    n_caught += 1
            elif v == "hallucinated":
                n_halluc += 1
            elif v == "misrouted":
                n_misroute += 1
            elif v == "wrong_content":
                n_wrong += 1
            elif v == "missing":
                n_missing += 1
            # judge vs human agreement, where a human has weighed in
            if k in human and human[k]:
                human_total += 1
                if str(human[k]).lower() == v:
                    human_agree += 1
        per_image.append({
            "image": j["image"],
            "correct": img_correct,
            "total": len(_FIELD_KEYS),
            "accuracy": round(img_correct / len(_FIELD_KEYS), 3),
        })

    should_be_filled = n_caught + n_missing

    def rate(num, den):
        return round(num / den, 3) if den else None

    return {
        "n_images": len(judgments),
        "fields_per_note": len(_FIELD_KEYS),
        # headline number (note: inflated by correctly-empty fields — read with the rates below)
        "accuracy": rate(n_correct, n_total),
        "precision": rate(n_caught, n_filled),          # of fields it filled, share placed correctly
        "recall": rate(n_caught, should_be_filled),      # of fields with info, share captured
        "hallucination_rate": rate(n_halluc, n_filled),  # the over-population problem
        "misroute_rate": rate(n_misroute, n_filled),     # the PE->ROS problem
        "wrong_content_rate": rate(n_wrong, n_filled),
        "counts": {
            "correct": n_correct, "total": n_total, "filled": n_filled,
            "hallucinated": n_halluc, "misrouted": n_misroute,
            "wrong_content": n_wrong, "missing": n_missing,
        },
        "judge_human_agreement": rate(human_agree, human_total),
        "human_reviewed_fields": human_total,
        "per_image": per_image,
    }


def score(run_id: str) -> int:
    run = RUNS_DIR / run_id
    preds = _load_json(run / "predictions.json", None)
    judgments = _load_json(run / "judgments.json", None)
    if preds is None or judgments is None:
        print(f"Run {run_id} missing predictions/judgments.", file=sys.stderr)
        return 1
    summary = aggregate(preds, judgments)
    _write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


# ---------------------------------------------------------------- cli
def _new_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bootstrap", help="(re)build reference transcripts")
    sub.add_parser("run", help="bootstrap + extract + judge + score (new run)")
    for name in ("extract", "judge", "score"):
        sp = sub.add_parser(name, help=f"{name} phase")
        sp.add_argument("run_id", nargs="?", help="run id (default: latest, or new for extract)")
    ap.add_argument("--provider", help="override JUDGE_PROVIDER (medgemma|claude)")
    args, _ = ap.parse_known_args()
    # --provider may land before or after the subcommand; re-read leniently.
    if args.provider:
        os.environ["JUDGE_PROVIDER"] = args.provider

    if args.cmd == "bootstrap":
        return bootstrap()

    if args.cmd == "run":
        run_id = _new_run_id()
        print(f"=== run {run_id} ===")
        rc = bootstrap()
        if rc:
            return rc
        rc = extract(run_id)
        if rc:
            return rc
        rc = judge_run(run_id)
        if rc:
            return rc
        return score(run_id)

    # phase commands take an optional run_id
    run_id = getattr(args, "run_id", None)
    if args.cmd == "extract":
        return extract(run_id or _new_run_id())
    run_id = run_id or _latest_run()
    if not run_id:
        print("No run id given and no existing runs.", file=sys.stderr)
        return 1
    return judge_run(run_id) if args.cmd == "judge" else score(run_id)


if __name__ == "__main__":
    raise SystemExit(main())

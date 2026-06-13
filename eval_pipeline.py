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


# ---------------------------------------------------------------- parallelism
# Data parallelism: each GPU gets a FULL copy of the model and its own slice of
# the images (not one model sharded across GPUs — that's slower for throughput).
# Workers run in spawned processes, each pinned to one physical GPU via
# CUDA_VISIBLE_DEVICES, so ocr/judge load on cuda:0 of that single visible device.
def _gpu_list(arg: str | None) -> list[str]:
    raw = arg if arg is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [g.strip() for g in raw.split(",") if g.strip()]


def _shard(items: list, n: int) -> list[list]:
    import math

    k = max(1, math.ceil(len(items) / n))
    return [items[i * k:(i + 1) * k] for i in range(n)]


def _chunks(seq: list, size: int):
    size = max(1, size)
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _run_parallel(worker, items: list, gpus: list[str], batch_size: int) -> list:
    """Split `items` across `gpus`, run `worker(gpu, shard, batch_size)` in one
    spawned process per GPU, and concatenate results in order.

    Even a single GPU runs in a subprocess (one worker) so the model is freed on
    process exit — that's what keeps the 4B and 27B from being resident at once
    across phases. Only with no GPU pinned (gpus==[]) does it run inline."""
    if not items:
        return []
    if not gpus:
        return worker(None, items, batch_size)
    import multiprocessing as mp

    shards = [s for s in _shard(items, len(gpus)) if s]
    tasks = list(zip(gpus, shards))
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        parts = pool.starmap(worker, [(g, s, batch_size) for g, s in tasks])
    merged: list = []
    for p in parts:
        merged.extend(p)
    return merged


def _pin(gpu: str | None) -> None:
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


# Workers must be module-level so the spawn pool can pickle them by name. Each
# imports torch/ocr/judge only AFTER pinning the GPU, so CUDA binds to it.
def _transcribe_worker(gpu: str | None, image_names: list[str], batch_size: int) -> list[str]:
    _pin(gpu)
    import judge

    REFS_DIR.mkdir(parents=True, exist_ok=True)
    done = []
    for chunk in _chunks(image_names, batch_size):
        blobs = [(IMAGES_DIR / n).read_bytes() for n in chunk]
        try:
            texts = judge.transcribe_batch(blobs)
        except Exception as e:  # noqa: BLE001
            texts = [f"[transcription failed: {e}]"] * len(blobs)
        for name, text in zip(chunk, texts):
            (REFS_DIR / f"{Path(name).stem}.txt").write_text(text.strip() + "\n")
            done.append(name)
    return done


def _extract_worker(gpu: str | None, image_names: list[str], batch_size: int) -> list[dict]:
    _pin(gpu)
    import ocr

    preds = []
    for chunk in _chunks(image_names, batch_size):
        blobs = [(IMAGES_DIR / n).read_bytes() for n in chunk]
        try:
            outs = ocr.extract_batch(blobs)
        except Exception as e:  # noqa: BLE001 - whole chunk failed
            outs = [{"text": "", "fields": {}, "error": str(e)}] * len(blobs)
        for name, out in zip(chunk, outs):
            rec = {"image": name, "fields": out.get("fields", {}), "raw_transcript": out.get("text", "")}
            if out.get("error"):
                rec["error"] = out["error"]
            preds.append(rec)
    return preds


def _judge_worker(gpu: str | None, payloads: list[dict], batch_size: int) -> list[dict]:
    # batch_size is unused for judging (per-item: the long JSON output makes
    # batched generation OOM-prone; the speedup comes from data parallelism).
    _pin(gpu)
    import judge

    out = []
    for p in payloads:
        ref = (p.get("reference") or "").strip()
        if not ref:
            verdicts = {k: {"verdict": "uncertain", "reason": "no reference transcript"}
                        for k in judge._FIELD_KEYS}
        else:
            verdicts = judge.judge_note(ref, p.get("fields", {}))
        out.append({"image": p["image"], "verdicts": verdicts, "human": {}})
    return out


# ---------------------------------------------------------------- phases
def bootstrap(gpus: list[str], batch_size: int) -> int:
    """Make a trusted transcript for any image that lacks one. Existing refs are
    left alone so clinician corrections persist across runs."""
    images = _images()
    if not images:
        print(f"No images in {IMAGES_DIR} — add some note photos first.", file=sys.stderr)
        return 1
    todo = [img.name for img in images
            if not (_ref_path(img).is_file() and _ref_path(img).read_text().strip())]
    if not todo:
        print(f"All {len(images)} image(s) already have reference transcripts.")
        return 0
    print(f"[ref] transcribing {len(todo)} image(s) on GPU(s) {gpus or 'default'} "
          f"(batch {batch_size}) …", flush=True)
    done = _run_parallel(_transcribe_worker, todo, gpus, batch_size)
    print(f"Bootstrapped {len(done)} new reference transcript(s); {len(images)} image(s) total.")
    return 0


def extract(run_id: str, gpus: list[str], batch_size: int) -> int:
    """Run the 4B extractor over every image -> predictions.json."""
    images = [img.name for img in _images()]
    if not images:
        print(f"No images in {IMAGES_DIR}.", file=sys.stderr)
        return 1
    print(f"[extract] {len(images)} note(s) on GPU(s) {gpus or 'default'} (batch {batch_size}) …",
          flush=True)
    preds = _run_parallel(_extract_worker, images, gpus, batch_size)
    _write_json(RUNS_DIR / run_id / "predictions.json", preds)
    print(f"Extracted {len(preds)} note(s) -> {run_id}/predictions.json")
    return 0


def judge_run(run_id: str, gpus: list[str], batch_size: int) -> int:
    """Grade each prediction against its reference transcript -> judgments.json."""
    preds = _load_json(RUNS_DIR / run_id / "predictions.json", None)
    if preds is None:
        print(f"No predictions for run {run_id}; run `extract` first.", file=sys.stderr)
        return 1
    payloads = []
    for p in preds:
        ref = _ref_path(IMAGES_DIR / p["image"])
        payloads.append({
            "image": p["image"],
            "fields": p.get("fields", {}),
            "reference": ref.read_text() if ref.is_file() else "",
        })
    print(f"[judge] {len(payloads)} note(s) on GPU(s) {gpus or 'default'} …", flush=True)
    judgments = _run_parallel(_judge_worker, payloads, gpus, batch_size)
    # Preserve any human review already recorded for this run.
    _merge_human(RUNS_DIR / run_id / "judgments.json", judgments)
    _write_json(RUNS_DIR / run_id / "judgments.json", judgments)
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
    # Common flags live on a parent parser attached to BOTH the top level and
    # every subcommand, so they're accepted before OR after the subcommand
    # (e.g. both `--batch-size 4 run` and `run --batch-size 4`). argparse.SUPPRESS
    # defaults stop the subparser copy from clobbering a value set up top.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", default=argparse.SUPPRESS,
                        help="override JUDGE_PROVIDER (medgemma|claude)")
    common.add_argument("--gpus", default=argparse.SUPPRESS,
                        help="comma-separated physical GPU ids to data-parallel across "
                             "(default: CUDA_VISIBLE_DEVICES, e.g. '2,3')")
    common.add_argument("--batch-size", type=int, default=argparse.SUPPRESS,
                        help="images per generate call within each GPU worker (default 4)")

    ap = argparse.ArgumentParser(description=__doc__, parents=[common],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bootstrap", parents=[common], help="(re)build reference transcripts")
    sub.add_parser("run", parents=[common], help="bootstrap + extract + judge + score (new run)")
    for name in ("extract", "judge", "score"):
        sp = sub.add_parser(name, parents=[common], help=f"{name} phase")
        sp.add_argument("run_id", nargs="?", help="run id (default: latest, or new for extract)")
    args = ap.parse_args()
    # SUPPRESS means absent flags aren't set as attributes — apply defaults here.
    args.provider = getattr(args, "provider", None)
    args.gpus = getattr(args, "gpus", None)
    args.batch_size = getattr(args, "batch_size", 4)
    if args.provider:
        os.environ["JUDGE_PROVIDER"] = args.provider
    gpus = _gpu_list(args.gpus)
    bs = args.batch_size

    if args.cmd == "bootstrap":
        return bootstrap(gpus, bs)

    if args.cmd == "run":
        run_id = _new_run_id()
        print(f"=== run {run_id} (gpus={gpus or 'default'}, batch={bs}) ===")
        rc = bootstrap(gpus, bs) or extract(run_id, gpus, bs) or judge_run(run_id, gpus, bs)
        return rc or score(run_id)

    # phase commands take an optional run_id
    run_id = getattr(args, "run_id", None)
    if args.cmd == "extract":
        return extract(run_id or _new_run_id(), gpus, bs)
    run_id = run_id or _latest_run()
    if not run_id:
        print("No run id given and no existing runs.", file=sys.stderr)
        return 1
    return judge_run(run_id, gpus, bs) if args.cmd == "judge" else score(run_id)


if __name__ == "__main__":
    raise SystemExit(main())

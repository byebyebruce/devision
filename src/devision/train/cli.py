"""Training-side command-line entry points: align, train, calibrate, eval, compare (see CLAUDE.md)."""
import argparse
import json
import os

from .samples import read_jsonl


def _parser(doc: str, config) -> argparse.ArgumentParser:
    """Arguments shared by align and train, plus one flag per field of the `config` dataclass."""
    p = argparse.ArgumentParser(description=doc)
    p.add_argument("--data", required=True)
    p.add_argument("--val")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--init", help="devision checkpoint to start from, e.g. the output of devision-align "
                                  "(default: Laya + SigLIP2 with a fresh projector)")
    p.add_argument("--laya", default="convaiinnovations/laya")
    p.add_argument("--vision", default="google/siglip2-base-patch16-256")
    p.add_argument("--visual-shuffle", type=int, default=2, help="2: 64 visual tokens, 1: 256")
    p.add_argument("--model-name", default="devision-0.1")
    for field, default in vars(config).items():
        p.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)
    p.set_defaults(swanlab_project="devision")  # pass --swanlab-project "" to turn it off
    return p


def seed_all(seed: int) -> None:
    """Seed `random` and torch before anything is built: a fresh projector is initialised when the model
    is constructed, which happens before align() / train() seed their own sampling."""
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)


def _decider(a):
    from ..model import Decider, ModelConfig, from_pretrained

    if a.init:
        return Decider.load(a.init)
    model, tok, cfg = from_pretrained(a.laya, a.vision, ModelConfig(model_name=a.model_name,
                                                                    visual_shuffle=a.visual_shuffle))
    return Decider(model, tok, cfg)


def _write_report(out: str, name: str, report) -> None:
    with open(os.path.join(out, name), "w") as f:
        json.dump(report, f)


def _need_train_extra() -> None:
    try:
        import peft  # noqa: F401
    except ImportError:
        raise SystemExit("training needs the train extra: pip install 'devision[train]'") from None


def align_main(argv=None) -> None:
    """Stage 1: align the projector with image-conditioned masked captions (JSONL of caption samples)."""
    _need_train_extra()
    from .align import AlignConfig, align

    a = _parser(align_main.__doc__ or "", AlignConfig()).parse_args(argv)
    config = AlignConfig(**{k: getattr(a, k) for k in vars(AlignConfig())})
    seed_all(config.seed)
    report = align(_decider(a), read_jsonl(a.data), a.data_root, a.out, config,
                   val_samples=read_jsonl(a.val) if a.val else None)
    _write_report(a.out, "align_report.json", report)


def train_main(argv=None) -> None:
    """Stage 2: RLCD fine-tuning on a JSONL of decision samples; saves a checkpoint.
    --val (and each --eval set) is evaluated every --eval-every steps; --val also fits the temperatures."""
    _need_train_extra()
    from .rlcd import TrainConfig, train

    p = _parser(train_main.__doc__ or "", TrainConfig())
    p.add_argument("--eval", action="append", default=[], metavar="NAME=JSONL",
                   help="extra set evaluated alongside val and logged as NAME/..., e.g. pope=data/pope.jsonl")
    a = p.parse_args(argv)
    config = TrainConfig(**{k: getattr(a, k) for k in vars(TrainConfig())})
    seed_all(config.seed)
    report = train(_decider(a), read_jsonl(a.data), a.data_root, a.out, config,
                   val_samples=read_jsonl(a.val) if a.val else None,
                   eval_sets={name: read_jsonl(path) for name, path in (e.split("=", 1) for e in a.eval)})
    _write_report(a.out, "train_report.json", report)


def _code_version() -> str:
    import subprocess

    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return rev + ("+dirty" if dirty else "") if rev else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sha256(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def eval_main(argv=None) -> None:
    """Evaluate a checkpoint on a JSONL of samples through decide(): a summary JSON (--out) and one record
    per question (--details, default <out without .json>.details.jsonl), from which every number of the
    summary can be recomputed. Questions about the same picture and text state go into one decide()
    request (at most --max-questions), so each picture is encoded once; --no-group asks one question per
    request (single-question latency)."""
    from ..model import Decider
    from .evaluate import CONTROLS, MAX_QUESTIONS, evaluate

    p = argparse.ArgumentParser(description=eval_main.__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out")
    p.add_argument("--details", help="per-question records (JSONL)")
    p.add_argument("--control", choices=CONTROLS, default="none",
                   help="mismatched: each picture's questions about another picture; reversed: choice options reversed")
    p.add_argument("--shuffle-images", action="store_true", help="same as --control mismatched")
    p.add_argument("--seed", type=int, default=0, help="seed of the mismatched-picture pairing")
    p.add_argument("--max-questions", type=int, default=MAX_QUESTIONS,
                   help="most questions per decide() request when grouping by picture")
    p.add_argument("--no-group", action="store_true",
                   help="one question per decide() request (to measure single-question latency; slower)")
    p.add_argument("--no-temperature", action="store_true",
                   help="evaluate the raw probabilities (all temperatures 1), to compare before / after calibration")
    p.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto (the fastest available; default)")
    p.add_argument("--role", choices=["unspecified", "heldout", "calibration_fit", "monitoring"], default="unspecified",
                   help="what this set was used for; heldout = never used to fit temperatures or choose. "
                        "Not given -> unspecified (nothing is claimed)")
    a = p.parse_args(argv)
    control = "mismatched" if a.shuffle_images else a.control
    if a.max_questions < 1:
        p.error("--max-questions must be >= 1")

    records: list = []
    decider = Decider.load(a.checkpoint, device=a.device)
    if a.no_temperature:
        decider.cfg.temperature, decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    result = evaluate(decider, read_jsonl(a.data), a.data_root, control=control, seed=a.seed, records_out=records,
                      max_questions=None if a.no_group else a.max_questions)
    result["run"] = {"checkpoint": a.checkpoint, "data": a.data, "data_sha256": _sha256(a.data), "role": a.role,
                     "device": str(next(decider.model.parameters()).device),
                     "temperature": "none (raw)" if a.no_temperature else "fitted",
                     "control": control, "seed": a.seed, "code": _code_version()}
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)
    details = a.details or (a.out[:-5] if a.out and a.out.endswith(".json") else a.out or "eval") + ".details.jsonl"
    if a.out or a.details:
        with open(details, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


CHECKPOINT_FILES = ("devision_config.json", "model.safetensors", "encoder", "vision", "tokenizer")


def calibrate_main(argv=None) -> None:
    """Refit a checkpoint's temperatures (one per question type and one per (type, option count) bucket) on
    a JSONL of decision samples, without training. Prints NLL / ECE per bucket before and after; writes the
    recalibrated checkpoint to --out (a copy) or, with --in-place, over the given one. Neither: report only."""
    _need_train_extra()
    import shutil

    from ..model import Decider
    from ..model.decider import CONFIG_FILE
    from .rlcd import MIN_BUCKET_ITEMS, calibrate

    p = argparse.ArgumentParser(description=calibrate_main.__doc__)
    p.add_argument("--checkpoint", required=True, help="checkpoint directory or Hugging Face repo id")
    p.add_argument("--data", required=True, help="samples to fit on, e.g. data/val_mix.jsonl (never a test set)")
    p.add_argument("--data-root", default="data")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--out", help="new directory: a copy of the checkpoint with the refitted temperatures")
    where.add_argument("--in-place", action="store_true", help="overwrite the checkpoint's devision_config.json")
    p.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto (the fastest available; default)")
    p.add_argument("--min-bucket", type=int, default=MIN_BUCKET_ITEMS,
                   help="fewer questions than this in a bucket: no bucket temperature, the type's is used")
    a = p.parse_args(argv)
    if a.in_place and not os.path.isdir(a.checkpoint):
        raise SystemExit("--in-place needs a local checkpoint directory")
    if a.out and os.path.exists(a.out):
        raise SystemExit("%s already exists" % a.out)

    decider = Decider.load(a.checkpoint, device=a.device)
    device = str(next(decider.model.parameters()).device)
    report = calibrate(decider, read_jsonl(a.data), a.data_root, device=device, min_bucket=a.min_bucket)
    report["run"] = {"checkpoint": a.checkpoint, "data": a.data, "data_sha256": _sha256(a.data),
                     "code": _code_version()}
    print("temperature (choice, score, noul): %s -> %s" % (
        [round(t, 3) for t in report["before"]["temperature"]], [round(t, 3) for t in report["after"]["temperature"]]))
    print("%-12s %6s %8s %8s %8s %8s %8s %8s" % ("bucket", "n", "T_before", "T_after", "nll_bef", "nll_aft",
                                                  "ece_bef", "ece_aft"))
    for name, after in report["metrics_after"].items():
        before = report["metrics_before"][name]
        print("%-12s %6d %8s %8s %8.4f %8.4f %8.4f %8.4f" % (
            name, after["n"], "%.3f" % before["temperature"] if "temperature" in before else "",
            "%.3f" % after["temperature"] if "temperature" in after else "",
            before["nll"], after["nll"], before["ece"], after["ece"]))
    if not (a.out or a.in_place):
        print("report only: pass --out DIR or --in-place to write the temperatures")
        return
    target = a.checkpoint if a.in_place else a.out
    if a.out:
        src = a.checkpoint
        if not os.path.isdir(src):
            from huggingface_hub import snapshot_download

            src = snapshot_download(src)
        os.makedirs(a.out)
        for name in CHECKPOINT_FILES:
            copy = shutil.copytree if os.path.isdir(os.path.join(src, name)) else shutil.copyfile
            copy(os.path.join(src, name), os.path.join(a.out, name))
    # rewrite only the temperatures: other entries (e.g. the "training" record of a published checkpoint)
    # are not ModelConfig fields and would otherwise be lost
    with open(os.path.join(target, CONFIG_FILE)) as f:
        saved = json.load(f)
    saved["temperature"] = list(decider.cfg.temperature)
    saved["temperature_by_options"] = dict(decider.cfg.temperature_by_options)
    with open(os.path.join(target, CONFIG_FILE), "w") as f:
        json.dump(saved, f, indent=2)
    with open(os.path.join(target, "calibrate_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote %s" % os.path.join(target, CONFIG_FILE))


def compare_main(argv=None) -> None:
    """Paired comparison of two evaluations of the same questions (their --details files): accuracy
    difference of B minus A with a bootstrap interval that resamples whole pictures."""
    from .compare import compare, load_records

    p = argparse.ArgumentParser(description=compare_main.__doc__)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--by", default="source", help="record field to report per group (source, kind, axis, type)")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", help="JSONL of samples: compare only on their ids (e.g. a subset neither model saw)")
    a = p.parse_args(argv)
    only = {s["id"] for s in read_jsonl(a.only)} if a.only else None
    print(json.dumps(compare(load_records(a.a), load_records(a.b), a.by, a.resamples, a.seed, only), indent=2))

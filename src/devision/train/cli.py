"""Training-side command-line entry points: align, train, eval (see CLAUDE.md)."""
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
    summary can be recomputed."""
    from ..model import Decider
    from .evaluate import CONTROLS, evaluate

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
    p.add_argument("--no-temperature", action="store_true",
                   help="evaluate the raw probabilities (all temperatures 1), to compare before / after calibration")
    p.add_argument("--role", choices=["unspecified", "heldout", "calibration_fit", "monitoring"], default="unspecified",
                   help="what this set was used for; heldout = never used to fit temperatures or choose. "
                        "Not given -> unspecified (nothing is claimed)")
    a = p.parse_args(argv)
    control = "mismatched" if a.shuffle_images else a.control

    records: list = []
    decider = Decider.load(a.checkpoint)
    if a.no_temperature:
        decider.cfg.temperature = [1.0, 1.0, 1.0]
    result = evaluate(decider, read_jsonl(a.data), a.data_root, control=control, seed=a.seed, records_out=records)
    result["run"] = {"checkpoint": a.checkpoint, "data": a.data, "data_sha256": _sha256(a.data), "role": a.role,
                     "temperature": "none (raw)" if a.no_temperature else "fitted",
                     "control": control, "seed": a.seed, "code": _code_version()}
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    details = a.details or (a.out[:-5] if a.out and a.out.endswith(".json") else a.out or "eval") + ".details.jsonl"
    if a.out or a.details:
        with open(details, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


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
    a = p.parse_args(argv)
    print(json.dumps(compare(load_records(a.a), load_records(a.b), a.by, a.resamples, a.seed), indent=2))

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


def align_main(argv=None) -> None:
    """Stage 1: align the projector with image-conditioned masked captions (JSONL of caption samples)."""
    from .align import AlignConfig, align

    a = _parser(align_main.__doc__ or "", AlignConfig()).parse_args(argv)
    config = AlignConfig(**{k: getattr(a, k) for k in vars(AlignConfig())})
    report = align(_decider(a), read_jsonl(a.data), a.data_root, a.out, config,
                   val_samples=read_jsonl(a.val) if a.val else None)
    _write_report(a.out, "align_report.json", report)


def train_main(argv=None) -> None:
    """Stage 2: RLCD fine-tuning on a JSONL of decision samples; saves a checkpoint.
    --val (and each --eval set) is evaluated every --eval-every steps; --val also fits the temperatures."""
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


def eval_main(argv=None) -> None:
    """Accuracy, ECE and latency of a checkpoint on a JSONL of samples, through decide()."""
    from ..model import Decider
    from .evaluate import evaluate

    p = argparse.ArgumentParser(description=eval_main.__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out")
    p.add_argument("--name", help="set name, metrics are logged as NAME/... (default: file name of --data)")
    p.add_argument("--swanlab-project", default="devision", help='"" turns SwanLab off')
    p.add_argument("--run-name", help="default: eval-<checkpoint dir>-<NAME>")
    a = p.parse_args(argv)

    result = evaluate(Decider.load(a.checkpoint), read_jsonl(a.data), a.data_root)
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    if a.swanlab_project:
        from .rlcd import _Tracker

        name = a.name or os.path.splitext(os.path.basename(a.data))[0]
        run = a.run_name or "eval-%s-%s" % (os.path.basename(os.path.normpath(a.checkpoint)), name)
        tracker = _Tracker(a.swanlab_project, run, {"checkpoint": a.checkpoint, "data": a.data, "n": result["n"]})
        metrics = {"accuracy": result["accuracy_all"], "ece": result["ece"],
                   "latency_p50_ms": result["latency_ms"]["p50"], "latency_p95_ms": result["latency_ms"]["p95"],
                   **{"accuracy_" + t: v for t, v in result["accuracy"].items()}}
        tracker.log({"%s/%s" % (name, k): v for k, v in metrics.items() if v is not None}, step=0)
        tracker.finish()

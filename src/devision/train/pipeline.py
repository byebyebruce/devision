"""devision-pipeline: run an experiment described by one YAML file.

    uv run devision-pipeline configs/x.yaml                      # a new training round, runs/v<N>-<name>/
    uv run devision-pipeline configs/x.yaml --dry-run            # print the commands only
    uv run devision-pipeline configs/x.yaml --round 3            # continue round 3 (skips finished stages)
    uv run devision-pipeline configs/x.yaml --round 3 --from stage2b   # rerun from a stage on

Every run of a training config is a new round, numbered after the highest runs/v<N>-* so far: round N of
config `name` lives in runs/vN-<name>/ -- run.json (config, start time, git commit), one directory per
stage, eval/ and logs/. A stage is one `devision-align` or `devision-train` run; `init:` names an
earlier stage (or is a checkpoint path / hub id, e.g. a stage of an earlier round); its SwanLab run is
called vN-<name>/<stage>. `params:` are the trainer's own options (AlignConfig / TrainConfig fields,
plus laya / vision / visual_shuffle / model_name), checked before anything runs. See configs/*.yaml.

After the stages, `evaluate:` runs devision-eval on each set (JSON in the round's eval/; sets already
evaluated are skipped unless rerun) and any extra shell commands, with {checkpoint} and {name}
substituted (and {eval_dir}), then prints a summary and adds it to the SwanLab run that trained the checkpoint
(test/<set>/... for heldout sets, ref/<set>/... for the others), found through the
swanlab_run.json the trainer left in the stage directory. An evaluation-only config (`stages: []`) is
not a round: it writes into the eval/ of the round the checkpoint belongs to. Every set is
labelled by what it shares -- sample ids or pictures, not file paths -- with the data the checkpoint
was tuned on: calibration_fit (at least half of it is in the temperature-fit set: the evaluated stage's
--val, or `calibration_fit:` for an existing checkpoint), monitoring (any overlap with the fit set or
with a set watched during training: the stages' --val / --eval and `history:`) or heldout (no
overlap). `image_identity:` names a JSON {id: picture} so COCO and Visual Genome ids of one picture
match. A set may also be given as {path: ..., role: ...} to state its role outright. With `controls: true` every set is
also evaluated with mismatched pictures (<set>.mismatched.json) and with choice options reversed
(<set>.reversed.json); the summary shows the mismatched accuracy and the share of flipped answers.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

KINDS = {"align": "align_main", "train": "train_main"}
REPORTS = {"align": "align_report.json", "train": "train_report.json"}
STAGE_KEYS = {"name", "kind", "init", "data", "val", "eval", "params"}
TOP_KEYS = {"name", "description", "runs_dir", "data_root", "device", "swanlab_project", "common", "stages",
            "evaluate"}


class PlanError(ValueError):
    pass


ROLES = ("calibration_fit", "monitoring", "heldout")
FIT_SHARE = 0.5


def _load_aliases(path: Optional[str]) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _identity(path: str, aliases: Dict[str, str]):
    """(sample ids, pictures) of a JSONL set; pictures are image ids mapped through `aliases`."""
    ids, pics = [], []
    with open(path) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                ids.append(s.get("id"))
                pics.append(aliases.get(s.get("image_id", ""), s.get("image_id", "")))
    return ids, pics


def _has_choice(path: str) -> bool:
    """Whether a set has any choice question (reversing options means nothing for noul); True if unknown."""
    if not os.path.exists(path):
        return True
    with open(path) as f:
        for line in f:
            if '"choice"' in line and any(q.get("type") == "choice" for q in json.loads(line)["questions"].values()):
                return True
    return False


def _role(path: str, fit: Optional[str], watched: Set[str], aliases: Dict[str, str]):
    """(role, overlap counts) from what the set shares with the fit set and the watched sets."""
    if not os.path.exists(path):  # plan-only checks without data: fall back to the path
        return ("calibration_fit" if path == fit else "monitoring" if path in watched else "heldout"), {}
    ids, pics = _identity(path, aliases)

    def shared(other: Optional[str]) -> int:
        if not other or not os.path.exists(other):
            return 0
        oid, opic = _identity(other, aliases)
        oid_set, opic_set = set(oid), set(opic)
        return sum(1 for i, p in zip(ids, pics) if i in oid_set or p in opic_set)

    in_fit = shared(fit)
    in_watched = max([shared(w) for w in watched] or [0])
    overlap = {"questions": len(ids), "shared_with_fit": in_fit, "shared_with_watched": in_watched}
    if ids and in_fit / len(ids) >= FIT_SHARE:
        return "calibration_fit", overlap
    if in_fit or in_watched:
        return "monitoring", overlap
    return "heldout", overlap


@dataclass
class Step:
    name: str                 # stage name, or "eval:<set>" / "cmd:<i>"
    argv: List[str]           # full command
    log: str
    out: Optional[str] = None  # checkpoint directory a stage writes
    done_marker: Optional[str] = None
    shell: bool = False


@dataclass
class Plan:
    name: str
    steps: List[Step] = field(default_factory=list)
    round_dir: Optional[str] = None   # runs/v<N>-<name> for a training config
    checkpoint: Optional[str] = None
    eval_outputs: Dict[str, str] = field(default_factory=dict)
    control_outputs: Dict[str, Dict[str, str]] = field(default_factory=dict)
    roles: Dict[str, str] = field(default_factory=dict)
    overlaps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    metric_names: Dict[str, str] = field(default_factory=dict)   # set -> name in SwanLab (prefix + set)
    config: Dict[str, Any] = field(default_factory=dict)


ROUND = re.compile(r"^v(\d+)-")


def next_round(runs: str) -> int:
    """One more than the highest round number in `runs` (1 if there is none)."""
    nums = [int(m.group(1)) for d in (os.listdir(runs) if os.path.isdir(runs) else [])
            if (m := ROUND.match(d)) and os.path.isdir(os.path.join(runs, d))]
    return max(nums, default=0) + 1


def find_round(runs: str, number: int) -> str:
    """The directory name of round `number` in `runs`."""
    found = [d for d in (os.listdir(runs) if os.path.isdir(runs) else []) if d.startswith("v%d-" % number)]
    if len(found) != 1:
        raise PlanError("round %d: expected one runs/v%d-* directory, found %s" % (number, number, found or "none"))
    return found[0]


def _flags(params: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for k, v in params.items():
        argv += ["--" + k.replace("_", "-"), "" if v is None else str(v)]
    return argv


def _check_args(kind: str, argv: List[str], where: str) -> None:
    """Parse `argv` with the trainer's own parser, so typos fail before hours of training."""
    from .cli import _parser
    if kind == "align":
        from .align import AlignConfig as Config
    else:
        from .rlcd import TrainConfig as Config
    p = _parser("", Config())
    if kind == "train":
        p.add_argument("--eval", action="append", default=[])
    p.exit_on_error = False
    p.allow_abbrev = False  # `epoch:` must not quietly mean `epochs:`
    try:
        _, unknown = p.parse_known_args(argv)
    except argparse.ArgumentError as e:
        raise PlanError("%s: %s" % (where, e)) from None
    if unknown:
        raise PlanError("%s: unknown params %s" % (where, [u for u in unknown if u.startswith("--")] or unknown))


def build_plan(cfg: Dict[str, Any], python: str = sys.executable, check_files: bool = True,
               round_name: Optional[str] = None) -> Plan:
    """Steps for a parsed YAML config; raises PlanError on anything that would fail later.
    `round_name` (e.g. "v3-x") places a training config's outputs; default: the next round."""
    unknown = set(cfg) - TOP_KEYS
    if unknown:
        raise PlanError("unknown top-level keys %s (allowed: %s)" % (sorted(unknown), sorted(TOP_KEYS)))
    name = cfg.get("name") or ""
    if not name:
        raise PlanError("`name` is required")
    runs, data_root = cfg.get("runs_dir", "runs"), cfg.get("data_root", "data")
    device, project = cfg.get("device", "auto"), cfg.get("swanlab_project", "devision")
    common = cfg.get("common") or {}
    stages = cfg.get("stages") or []
    ev_ck = (cfg.get("evaluate") or {}).get("checkpoint")
    eval_only = not stages and ev_ck and ("/" in ev_ck or os.path.exists(ev_ck))
    if not stages and not eval_only:
        raise PlanError("`stages` is empty (an evaluation-only config needs evaluate.checkpoint as a path)")
    plan = Plan(name, config=cfg)
    outs: Dict[str, str] = {}
    if stages:
        round_name = round_name or "v%d-%s" % (next_round(runs), name)
        plan.round_dir = os.path.join(runs, round_name)

    def exists(path: str, where: str) -> None:
        if check_files and not os.path.exists(path):
            raise PlanError("%s: %s does not exist" % (where, path))

    for i, st in enumerate(stages):
        sname = st.get("name") or ""
        where = "stage %d (%s)" % (i + 1, sname or "unnamed")
        if not sname:
            raise PlanError("%s: `name` is required" % where)
        if sname in outs:
            raise PlanError("%s: duplicate stage name" % where)
        extra = set(st) - STAGE_KEYS
        if extra:
            raise PlanError("%s: unknown keys %s (allowed: %s)" % (where, sorted(extra), sorted(STAGE_KEYS)))
        kind = st.get("kind")
        if kind not in KINDS:
            raise PlanError("%s: kind must be one of %s" % (where, sorted(KINDS)))
        if kind == "align" and st.get("eval"):
            raise PlanError("%s: `eval` sets are only for kind: train" % where)
        assert plan.round_dir and round_name
        out = os.path.join(plan.round_dir, sname)
        argv = ["--data", st.get("data") or "", "--data-root", data_root, "--out", out,
                "--run-name", "%s/%s" % (round_name, sname), "--device", device, "--swanlab-project", project]
        exists(st.get("data") or "<missing data>", where)
        if st.get("val"):
            exists(st["val"], where)
            argv += ["--val", st["val"]]
        init = st.get("init")
        if init:
            if init in outs:
                init = outs[init]
            elif "/" not in init and not os.path.exists(init):
                raise PlanError("%s: init %r is neither an earlier stage nor a path / hub id" % (where, init))
            argv += ["--init", init]
        for set_name, path in (st.get("eval") or {}).items():
            exists(path, where)
            argv += ["--eval", "%s=%s" % (set_name, path)]
        argv += _flags({**common, **(st.get("params") or {})})
        _check_args(kind, argv, where)
        entry = "import sys; from devision.train.cli import %s; %s(sys.argv[1:])" % (KINDS[kind], KINDS[kind])
        plan.steps.append(Step(sname, [python, "-c", entry] + argv,
                               os.path.join(plan.round_dir, "logs", "%s.log" % sname),
                               out=out, done_marker=os.path.join(out, REPORTS[kind])))
        outs[sname] = out

    ev = cfg.get("evaluate") or {}
    if ev:
        ck = ev.get("checkpoint") or stages[-1]["name"]
        by_name = {st["name"]: st for st in stages}
        if ck in outs:
            ckpt: str = outs[ck]
            fit = by_name[ck].get("val") if by_name[ck].get("kind") == "train" else None
        elif eval_only:
            ckpt = str(ck)
            fit = ev.get("calibration_fit")   # the val file the checkpoint's temperatures were fitted on
        else:
            raise PlanError("evaluate: checkpoint %r is not a stage" % ck)
        plan.checkpoint = ckpt
        home = plan.round_dir or os.path.dirname(os.path.normpath(ckpt))
        if plan.round_dir or os.path.exists(os.path.join(home, "run.json")):
            eval_dir, log = os.path.join(home, "eval"), os.path.join(home, "logs", "eval-%s.log" % name)
        else:   # a checkpoint from before rounds: results next to it, as they always were
            eval_dir, log = ckpt, os.path.join(runs, "logs", "%s-eval.log" % name)
        entry = "import sys; from devision.train.cli import eval_main; eval_main(sys.argv[1:])"
        watched = {p for st in stages for p in [st.get("val")] + list((st.get("eval") or {}).values()) if p}
        watched |= set(ev.get("history") or [])
        watched.discard(fit)
        aliases = _load_aliases(ev.get("image_identity"))
        for set_name, spec in (ev.get("sets") or {}).items():
            path, stated = (spec.get("path"), spec.get("role")) if isinstance(spec, dict) else (spec, None)
            if not isinstance(path, str):
                raise PlanError("evaluate: set %r needs a path" % set_name)
            if stated and stated not in ROLES:
                raise PlanError("evaluate: set %r role must be one of %s" % (set_name, ROLES))
            exists(path, "evaluate")
            role, overlap = _role(path, fit, watched, aliases)
            plan.roles[set_name] = stated or role
            plan.overlaps[set_name] = overlap
            role = plan.roles[set_name]
            out_json = os.path.join(eval_dir, "%s%s.json" % (ev.get("prefix", ""), set_name))
            plan.eval_outputs[set_name] = out_json
            plan.metric_names[set_name] = ev.get("prefix", "") + set_name
            base = [python, "-c", entry, "--checkpoint", plan.checkpoint, "--data", path, "--data-root", data_root,
                    "--role", role]
            plan.steps.append(Step("eval:" + set_name, base + ["--out", out_json], log, done_marker=out_json))
            if ev.get("controls"):
                plan.control_outputs[set_name] = {}
                for control in ("mismatched", "reversed") if _has_choice(path) else ("mismatched",):
                    ctl_json = os.path.join(eval_dir, "%s%s.%s.json" % (ev.get("prefix", ""), set_name, control))
                    plan.control_outputs[set_name][control] = ctl_json
                    plan.steps.append(Step("%s:%s" % (control, set_name),
                                           base + ["--out", ctl_json, "--control", control], log,
                                           done_marker=ctl_json))
        for j, cmd in enumerate(ev.get("commands") or []):
            plan.steps.append(Step("cmd:%d" % (j + 1), [cmd.format(checkpoint=plan.checkpoint, name=name,
                                                                  eval_dir=eval_dir)],
                                   log, shell=True))
    return plan


COLUMNS = ("accuracy", "accuracy_noul", "accuracy_choice", "nll", "ece", "mismatched_accuracy", "flip_rate")


def _results(plan: Plan) -> Dict[str, Dict[str, Optional[float]]]:
    """Per evaluated set: the summary numbers (None where not measured)."""
    from .compare import load_records
    from .evaluate import order_sensitivity

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for set_name, path in plan.eval_outputs.items():
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        ctl = plan.control_outputs.get(set_name, {})
        mism = json.load(open(ctl["mismatched"])).get("accuracy_all") if os.path.exists(ctl.get("mismatched", "")) else None
        flips = None
        rev = ctl.get("reversed", "")[:-5] + ".details.jsonl"
        plain = path[:-5] + ".details.jsonl"
        if os.path.exists(rev) and os.path.exists(plain):
            flips = order_sensitivity(load_records(plain), load_records(rev)).get("answer_flip_rate")
        out[set_name] = {"accuracy": r.get("accuracy_all"), "accuracy_noul": r["accuracy"].get("noul"),
                         "accuracy_choice": r["accuracy"].get("choice"), "nll": r.get("nll"), "ece": r.get("ece"),
                         "mismatched_accuracy": mism, "flip_rate": flips}
    return out


def _summary(plan: Plan) -> str:
    f = lambda v: "%.3f" % v if isinstance(v, (int, float)) else "-"
    lines = ["%-18s %-15s %7s %7s %7s %7s %7s %10s %8s" % ("set", "role", "all", "noul", "choice", "NLL", "ECE",
                                                           "mismatch", "flips")]
    for set_name, r in _results(plan).items():
        lines.append("%-18s %-15s " % (set_name, plan.roles.get(set_name, "-"))
                     + "%7s %7s %7s %7s %7s %10s %8s" % tuple(f(r[c]) for c in COLUMNS))
    return "\n".join(lines)


def swanlab_metrics(plan: Plan) -> Dict[str, float]:
    """The evaluation summary as SwanLab scalars: test/<set>/<metric> for heldout sets, ref/<set>/<metric>
    for sets the checkpoint was tuned or watched on."""
    out: Dict[str, float] = {}
    for set_name, r in _results(plan).items():
        group = "test" if plan.roles.get(set_name) == "heldout" else "ref"
        for k, v in r.items():
            if isinstance(v, (int, float)):
                out["%s/%s/%s" % (group, plan.metric_names.get(set_name, set_name), k)] = float(v)
    return out


def report_to_swanlab(plan: Plan) -> None:
    """Add the evaluation summary to the SwanLab run that trained the checkpoint, if one is recorded."""
    from .tracking import RECORD, log_to_finished_run

    record = os.path.join(plan.checkpoint or "", RECORD)
    if not plan.checkpoint or not os.path.exists(record):
        print("   (no %s next to the checkpoint: evaluation not added to SwanLab)" % RECORD, flush=True)
        return
    if (plan.config.get("swanlab_project", "devision")) == "":
        return
    metrics = swanlab_metrics(plan)
    if not metrics:
        return
    try:
        where = log_to_finished_run(record, metrics)
        print("   %d evaluation numbers added to SwanLab run %s" % (len(metrics), where), flush=True)
    except Exception as e:  # results are on disk either way; a SwanLab problem must not fail the run
        print("   could not add the evaluation to SwanLab: %s" % e, flush=True)


def _git_commit() -> Optional[str]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True).stdout.strip()
        return (head + ("+dirty" if dirty else "")) or None
    except OSError:
        return None


def _write_round(plan: Plan) -> None:
    """runs/v<N>-<name>/run.json: what this round is (written once, when it starts)."""
    assert plan.round_dir
    path = os.path.join(plan.round_dir, "run.json")
    if os.path.exists(path):
        return
    os.makedirs(plan.round_dir, exist_ok=True)
    stages = [{"name": s.name, "out": s.out, "init": s.argv[s.argv.index("--init") + 1] if "--init" in s.argv else None}
              for s in plan.steps if s.out]
    with open(path, "w") as f:
        json.dump({"round": os.path.basename(plan.round_dir), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "git_commit": _git_commit(), "stages": stages, "config": plan.config}, f, indent=1)


def run_plan(plan: Plan, start_from: Optional[str] = None, force: bool = False, dry_run: bool = False) -> None:
    rerun = force
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    if plan.round_dir:
        print("== round %s" % plan.round_dir, flush=True)
        if not dry_run:
            _write_round(plan)
    for step in plan.steps:
        if start_from and step.name == start_from:
            rerun = True
        stage = step.out is not None
        if not step.shell and not rerun and step.done_marker and os.path.exists(step.done_marker):
            print("== %s: already done (%s), skipped" % (step.name, step.out or step.done_marker), flush=True)
            continue
        if stage:
            rerun = True  # every later stage builds on this one
        shown = step.argv[0] if step.shell else " ".join(shlex.quote(a) for a in step.argv)
        print("== %s %s\n   %s\n   log: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), step.name, shown, step.log),
              flush=True)
        if step.name.startswith("eval:") and plan.overlaps.get(step.name[5:]):
            print("   role %s, overlap %s" % (plan.roles[step.name[5:]], plan.overlaps[step.name[5:]]), flush=True)
        if dry_run:
            continue
        os.makedirs(os.path.dirname(step.log), exist_ok=True)
        with open(step.log, "a") as log:
            code = subprocess.run(step.argv[0] if step.shell else step.argv, shell=step.shell, env=env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            raise SystemExit("%s failed (exit %d); see %s" % (step.name, code, step.log))
    if not dry_run and plan.eval_outputs:
        print("== done %s\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"), _summary(plan)), flush=True)
        report_to_swanlab(plan)


def pipeline_main(argv=None) -> None:
    """Run the stages and evaluations of an experiment YAML (see configs/)."""
    import yaml

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--round", type=int, help="continue this round (runs/v<N>-*) instead of starting a new one")
    p.add_argument("--from", dest="start_from", help="rerun from this stage (later stages too); needs --round")
    p.add_argument("--force", action="store_true", help="rerun stages that are already done")
    p.add_argument("--dry-run", action="store_true", help="check the config and print the commands")
    a = p.parse_args(argv)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.start_from and not a.round and cfg.get("stages"):
        raise SystemExit("--from needs --round N (which round to rerun the stage in)")
    try:
        round_name = find_round(cfg.get("runs_dir", "runs"), a.round) if a.round else None
        plan = build_plan(cfg, round_name=round_name)
    except PlanError as e:
        raise SystemExit("%s: %s" % (a.config, e)) from None
    if a.start_from and a.start_from not in [s.name for s in plan.steps]:
        raise SystemExit("--from %s: no such stage" % a.start_from)
    run_plan(plan, a.start_from, a.force, a.dry_run)

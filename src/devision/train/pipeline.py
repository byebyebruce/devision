"""devision-pipeline: run an experiment described by one YAML file.

    uv run devision-pipeline configs/release-0.1.yaml            # run it (skips stages already done)
    uv run devision-pipeline configs/release-0.1.yaml --dry-run  # print the commands only
    uv run devision-pipeline configs/x.yaml --from stage2b       # rerun from a stage on

A stage is one `devision-align` or `devision-train` run; `init:` names an earlier stage (or is a
checkpoint path / hub id). Stage N of experiment E writes runs/E-N/ and logs to runs/logs/E-N.log; its
SwanLab run is called E-N. `params:` are the trainer's own options (AlignConfig / TrainConfig fields,
plus laya / vision / visual_shuffle / model_name), checked before anything runs. See configs/*.yaml.

After the stages, `evaluate:` runs devision-eval on each set (JSON next to the checkpoint) and any
extra shell commands, with {checkpoint} and {name} substituted, then prints a summary. With
`controls: true` every set is also evaluated with mismatched images (<set>.mismatched.json); a
model that reads the picture should drop to about chance there.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

KINDS = {"align": "align_main", "train": "train_main"}
REPORTS = {"align": "align_report.json", "train": "train_report.json"}
STAGE_KEYS = {"name", "kind", "init", "data", "val", "eval", "params"}
TOP_KEYS = {"name", "description", "runs_dir", "data_root", "device", "swanlab_project", "common", "stages",
            "evaluate"}


class PlanError(ValueError):
    pass


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
    checkpoint: Optional[str] = None
    eval_outputs: Dict[str, str] = field(default_factory=dict)
    control_outputs: Dict[str, str] = field(default_factory=dict)


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


def build_plan(cfg: Dict[str, Any], python: str = sys.executable, check_files: bool = True) -> Plan:
    """Steps for a parsed YAML config; raises PlanError on anything that would fail later."""
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
    if not stages:
        raise PlanError("`stages` is empty")
    plan = Plan(name)
    outs: Dict[str, str] = {}

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
        out = os.path.join(runs, "%s-%s" % (name, sname))
        argv = ["--data", st.get("data") or "", "--data-root", data_root, "--out", out,
                "--run-name", "%s-%s" % (name, sname), "--device", device, "--swanlab-project", project]
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
                               os.path.join(runs, "logs", "%s-%s.log" % (name, sname)),
                               out=out, done_marker=os.path.join(out, REPORTS[kind])))
        outs[sname] = out

    ev = cfg.get("evaluate") or {}
    if ev:
        ck = ev.get("checkpoint") or stages[-1]["name"]
        if ck not in outs:
            raise PlanError("evaluate: checkpoint %r is not a stage" % ck)
        plan.checkpoint = outs[ck]
        log = os.path.join(runs, "logs", "%s-eval.log" % name)
        entry = "import sys; from devision.train.cli import eval_main; eval_main(sys.argv[1:])"
        for set_name, path in (ev.get("sets") or {}).items():
            exists(path, "evaluate")
            out_json = os.path.join(plan.checkpoint, "%s.json" % set_name)
            plan.eval_outputs[set_name] = out_json
            plan.steps.append(Step("eval:" + set_name, [python, "-c", entry, "--checkpoint", plan.checkpoint,
                                   "--data", path, "--data-root", data_root, "--out", out_json], log))
            if ev.get("controls"):
                ctl_json = os.path.join(plan.checkpoint, "%s.mismatched.json" % set_name)
                plan.control_outputs[set_name] = ctl_json
                plan.steps.append(Step("control:" + set_name, [python, "-c", entry, "--checkpoint", plan.checkpoint,
                                       "--data", path, "--data-root", data_root, "--out", ctl_json,
                                       "--shuffle-images"], log))
        for j, cmd in enumerate(ev.get("commands") or []):
            plan.steps.append(Step("cmd:%d" % (j + 1), [cmd.format(checkpoint=plan.checkpoint, name=name)],
                                   log, shell=True))
    return plan


def _summary(plan: Plan) -> str:
    lines = ["%-18s %8s %8s %8s %7s %11s" % ("set", "all", "noul", "choice", "ECE", "mismatched")]
    for set_name, path in plan.eval_outputs.items():
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        f = lambda v: "%.3f" % v if isinstance(v, (int, float)) else "-"
        ctl = plan.control_outputs.get(set_name)
        mism = json.load(open(ctl)).get("accuracy_all") if ctl and os.path.exists(ctl) else None
        lines.append("%-18s %8s %8s %8s %7s %11s" % (set_name, f(r.get("accuracy_all")), f(r["accuracy"].get("noul")),
                                                     f(r["accuracy"].get("choice")), f(r.get("ece")), f(mism)))
    return "\n".join(lines)


def run_plan(plan: Plan, start_from: Optional[str] = None, force: bool = False, dry_run: bool = False) -> None:
    rerun = force
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    for step in plan.steps:
        if start_from and step.name == start_from:
            rerun = True
        stage = step.out is not None
        if stage and not rerun and step.done_marker and os.path.exists(step.done_marker):
            print("== %s: already done (%s), skipped" % (step.name, step.out), flush=True)
            continue
        if stage:
            rerun = True  # every later stage builds on this one
        shown = step.argv[0] if step.shell else " ".join(shlex.quote(a) for a in step.argv)
        print("== %s %s\n   %s\n   log: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), step.name, shown, step.log),
              flush=True)
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


def pipeline_main(argv=None) -> None:
    """Run the stages and evaluations of an experiment YAML (see configs/)."""
    import yaml

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--from", dest="start_from", help="rerun from this stage (later stages too)")
    p.add_argument("--force", action="store_true", help="rerun stages that are already done")
    p.add_argument("--dry-run", action="store_true", help="check the config and print the commands")
    a = p.parse_args(argv)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    try:
        plan = build_plan(cfg)
    except PlanError as e:
        raise SystemExit("%s: %s" % (a.config, e)) from None
    if a.start_from and a.start_from not in [s.name for s in plan.steps]:
        raise SystemExit("--from %s: no such stage" % a.start_from)
    run_plan(plan, a.start_from, a.force, a.dry_run)

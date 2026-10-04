"""devision-pipeline: an experiment YAML becomes a checked, resumable sequence of trainer runs."""
import json

import pytest

from devision.train import pipeline
from devision.train.pipeline import PlanError, build_plan, run_plan


def config(**over):
    cfg = {
        "name": "v3-exp",
        "device": "cpu",
        "common": {"log_every": 50},
        "stages": [
            {"name": "align", "kind": "align", "data": "caps.jsonl", "val": "caps_val.jsonl",
             "params": {"epochs": 2, "lora_r": 16}},
            {"name": "decide", "kind": "train", "init": "align", "data": "q.jsonl", "val": "q_val.jsonl",
             "eval": {"pope": "pope.jsonl"}, "params": {"lr_head": 5.0e-5, "warmup": 500}},
        ],
        "evaluate": {"sets": {"pope": "pope.jsonl"}, "commands": ["echo {checkpoint} {name}"]},
    }
    cfg.update(over)
    return cfg


def plan(cfg):
    return build_plan(cfg, python="python", check_files=False)


def test_a_stage_starts_from_the_checkpoint_of_the_stage_it_names():
    align, decide = plan(config()).steps[:2]

    assert align.out == "runs/v3-exp/align"
    assert "--init" not in align.argv  # first stage: Laya + SigLIP2 + fresh projector
    assert decide.argv[decide.argv.index("--init") + 1] == "runs/v3-exp/align"
    assert decide.argv[decide.argv.index("--run-name") + 1] == "v3-exp/decide"
    assert decide.log == "runs/v3-exp/logs/decide.log"


def test_params_and_common_params_reach_the_trainer_as_flags():
    align, decide = plan(config()).steps[:2]

    assert align.argv[align.argv.index("--lora-r") + 1] == "16"
    assert decide.argv[decide.argv.index("--log-every") + 1] == "50"
    assert "pope=pope.jsonl" in decide.argv


def test_controls_add_mismatched_and_reversed_runs_per_set():
    p = plan(config(evaluate={"sets": {"pope": "pope.jsonl"}, "controls": True}))
    names = [s.name for s in p.steps]
    assert "mismatched:pope" in names and "reversed:pope" in names
    step = next(s for s in p.steps if s.name == "mismatched:pope")
    assert step.argv[step.argv.index("--control") + 1] == "mismatched"
    assert p.control_outputs["pope"]["reversed"] == "runs/v3-exp/eval/pope.reversed.json"


def test_an_existing_checkpoint_can_be_evaluated_without_training():
    cfg = config(stages=[], evaluate={"checkpoint": "runs/old-model", "calibration_fit": "q_val.jsonl",
                                      "prefix": "v2_", "sets": {"old": "q_val.jsonl", "new": "x.jsonl"}})
    p = plan(cfg)
    assert [s.name for s in p.steps] == ["eval:old", "eval:new"]
    assert p.roles == {"old": "calibration_fit", "new": "heldout"}
    assert p.eval_outputs["new"] == "runs/old-model/v2_new.json"


def test_a_set_without_choice_questions_gets_no_reversed_run(tmp_path):
    noul = _jsonl(tmp_path / "pope.jsonl", [{"id": "p", "image_id": "coco:1",
                                              "questions": {"q": {"type": "noul", "instructions": "?"}}}])
    choice = _jsonl(tmp_path / "mc.jsonl", [{"id": "c", "image_id": "coco:2",
                                              "questions": {"q": {"type": "choice", "instructions": "?",
                                                                  "criteria": {"a": None, "b": None}}}}])
    p = plan(config(evaluate={"sets": {"pope": noul, "mc": choice}, "controls": True}))
    names = [s.name for s in p.steps]
    assert "mismatched:pope" in names and "reversed:pope" not in names
    assert "reversed:mc" in names and "reversed" not in p.control_outputs["pope"]


def test_sets_are_labelled_by_what_they_were_used_for():
    p = plan(config(evaluate={"sets": {"fit": "q_val.jsonl", "watched": "pope.jsonl", "fresh": "new.jsonl"}}))
    assert p.roles == {"fit": "calibration_fit", "watched": "monitoring", "fresh": "heldout"}
    step = next(s for s in p.steps if s.name == "eval:fit")
    assert step.argv[step.argv.index("--role") + 1] == "calibration_fit"


def test_evaluation_runs_on_the_last_stage_and_fills_in_commands():
    p = plan(config())

    assert p.checkpoint == "runs/v3-exp/decide"
    assert p.eval_outputs == {"pope": "runs/v3-exp/eval/pope.json"}
    assert p.steps[-1].argv == ["echo runs/v3-exp/decide v3-exp"]


@pytest.mark.parametrize("cfg, message", [
    (config(stages=[{"name": "a", "kind": "align", "data": "c.jsonl", "params": {"epoch": 2}}]), "unknown params"),
    (config(stages=[{"name": "a", "kind": "align", "data": "c.jsonl", "params": {"epochs": "two"}}]), "epochs"),
    (config(stages=[{"name": "a", "kind": "train", "init": "nope", "data": "q.jsonl"}]), "neither an earlier stage"),
    (config(stages=[{"name": "a", "kind": "finetune", "data": "q.jsonl"}]), "kind must be"),
    (config(stages=[{"name": "a", "kind": "align", "data": "c.jsonl", "lr": 1}]), "unknown keys"),
    (config(stages=[]), "empty"),
    (config(name=""), "name"),
    (config(evaluate={"checkpoint": "nope", "sets": {}}), "not a stage"),
])
def test_mistakes_are_reported_before_anything_runs(cfg, message):
    with pytest.raises(PlanError, match=message):
        plan(cfg)


def test_missing_data_files_are_reported_up_front(tmp_path):
    with pytest.raises(PlanError, match="does not exist"):
        build_plan(config(data_root=str(tmp_path)), check_files=True)


def test_finished_stages_are_skipped_and_everything_after_a_rerun_stage_reruns(tmp_path, capsys):
    cfg = config(runs_dir=str(tmp_path))
    p = build_plan(cfg, python="python", check_files=False)
    (tmp_path / "v3-exp" / "align").mkdir(parents=True)
    (tmp_path / "v3-exp" / "align" / "align_report.json").write_text("{}")
    (tmp_path / "v3-exp" / "decide").mkdir()
    (tmp_path / "v3-exp" / "decide" / "train_report.json").write_text("{}")

    run_plan(p, dry_run=True)
    out = capsys.readouterr().out
    assert "align: already done" in out and "decide: already done" in out

    run_plan(p, start_from="decide", dry_run=True)
    out = capsys.readouterr().out
    assert "align: already done" in out and "decide: already done" not in out


def test_the_yaml_name_is_the_round_directory_and_a_rerun_continues_it(tmp_path, capsys):
    p = build_plan(config(runs_dir=str(tmp_path)), python="python", check_files=False)
    assert p.round_dir == str(tmp_path / "v3-exp")
    (tmp_path / "v3-exp" / "align").mkdir(parents=True)
    (tmp_path / "v3-exp" / "align" / "align_report.json").write_text("{}")
    run_plan(build_plan(config(runs_dir=str(tmp_path)), python="python", check_files=False), dry_run=True)
    assert "align: already done" in capsys.readouterr().out


def test_evaluating_a_checkpoint_of_a_round_writes_into_that_round(tmp_path):
    (tmp_path / "v2-old" / "decide").mkdir(parents=True)
    (tmp_path / "v2-old" / "run.json").write_text("{}")
    ck = str(tmp_path / "v2-old" / "decide")
    p = plan(config(stages=[], evaluate={"checkpoint": ck, "prefix": "bench_lv_", "sets": {"pope": "pope.jsonl"}}))
    assert p.round_dir is None   # evaluating is not a new round
    assert p.eval_outputs == {"pope": str(tmp_path / "v2-old" / "eval" / "bench_lv_pope.json")}
    assert p.metric_names == {"pope": "bench_lv_pope"}


def test_finished_evaluations_are_skipped_on_a_rerun(tmp_path, capsys):
    p = build_plan(config(runs_dir=str(tmp_path)), python="python", check_files=False)
    for st in ("align", "decide"):
        (tmp_path / "v3-exp" / st).mkdir(parents=True)
    (tmp_path / "v3-exp" / "align" / "align_report.json").write_text("{}")
    (tmp_path / "v3-exp" / "decide" / "train_report.json").write_text("{}")
    (tmp_path / "v3-exp" / "eval").mkdir()
    (tmp_path / "v3-exp" / "eval" / "pope.json").write_text("{}")
    run_plan(p, dry_run=True)
    assert "eval:pope: already done" in capsys.readouterr().out


def test_the_evaluation_summary_goes_to_swanlab_as_test_bench_and_ref_numbers(tmp_path):
    fresh = _jsonl(tmp_path / "fresh.jsonl", [{"id": "f1", "image_id": "coco:50"}])
    other = _jsonl(tmp_path / "other.jsonl", [{"id": "o1", "image_id": "coco:60"}])
    p = plan(config(evaluate={"sets": {"bench_pope": "pope.jsonl", "test_fresh": fresh, "bench_other": other}}))
    for set_name, acc in (("bench_pope", 0.8), ("test_fresh", 0.7), ("bench_other", 0.6)):
        out = tmp_path / ("%s.json" % set_name)
        out.write_text(json.dumps({"accuracy_all": acc, "accuracy": {"noul": acc, "choice": None},
                                   "nll": 0.5, "ece": 0.03}))
        p.eval_outputs[set_name] = str(out)
    m = pipeline.swanlab_metrics(p)
    assert m["ref/bench_pope/accuracy"] == 0.8      # watched during training: a reference, benchmark or not
    assert m["test/test_fresh/accuracy"] == 0.7     # our own held-out split
    assert m["bench/bench_other/accuracy"] == 0.6   # held-out external benchmark
    assert "test/test_fresh/accuracy_choice" not in m and "test/test_fresh/flip_rate" not in m
    assert set(pipeline.swanlab_metrics(p, only={"test_fresh"})) == {k for k in m if k.startswith("test/test_fresh/")}


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_roles_follow_shared_questions_and_pictures_not_file_names(tmp_path):
    fit = _jsonl(tmp_path / "mix.jsonl", [{"id": "g%d" % i, "image_id": "vg:%d" % i} for i in range(10)])
    subset = _jsonl(tmp_path / "gqa_part.jsonl", [{"id": "g%d" % i, "image_id": "vg:%d" % i} for i in range(6)])
    # one picture of the fit set under its COCO id
    pope = _jsonl(tmp_path / "pope.jsonl", [{"id": "p%d" % i, "image_id": "coco:%d" % (7 + i)} for i in range(3)])
    watched_old = _jsonl(tmp_path / "old.jsonl", [{"id": "o1", "image_id": "coco:99"}])
    fresh = _jsonl(tmp_path / "fresh.jsonl", [{"id": "f1", "image_id": "coco:50"}])
    aliases = tmp_path / "ids.json"
    aliases.write_text(json.dumps({"vg:3": "coco:7"}))
    cfg = config(stages=[], evaluate={"checkpoint": "runs/old", "calibration_fit": fit, "history": [watched_old],
                                      "image_identity": str(aliases),
                                      "sets": {"part": subset, "pope": pope, "fresh": fresh,
                                               "stated": {"path": fresh, "role": "monitoring"}}})
    p = build_plan(cfg, python="python", check_files=False)
    assert p.roles == {"part": "calibration_fit", "pope": "monitoring", "fresh": "heldout", "stated": "monitoring"}
    assert p.overlaps["pope"]["shared_with_fit"] == 1
    step = next(s for s in p.steps if s.name == "eval:part")
    assert step.argv[step.argv.index("--role") + 1] == "calibration_fit"


def test_an_evaluation_gets_its_own_swanlab_run_named_after_the_round(tmp_path):
    p = plan(config())
    assert p.eval_run_name == "v3-exp/test"
    (tmp_path / "v2-old" / "decide").mkdir(parents=True)
    (tmp_path / "v2-old" / "run.json").write_text("{}")
    only = plan(config(name="lvbench", stages=[],
                       evaluate={"checkpoint": str(tmp_path / "v2-old" / "decide"), "sets": {"pope": "pope.jsonl"}}))
    assert only.eval_run_name == "v2-old/lvbench"
    assert only.eval_dir == str(tmp_path / "v2-old" / "eval")

"""devision-pipeline: an experiment YAML becomes a checked, resumable sequence of trainer runs."""
import pytest

from devision.train.pipeline import PlanError, build_plan, run_plan


def config(**over):
    cfg = {
        "name": "exp",
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

    assert align.out == "runs/exp-align"
    assert "--init" not in align.argv  # first stage: Laya + SigLIP2 + fresh projector
    assert decide.argv[decide.argv.index("--init") + 1] == "runs/exp-align"
    assert decide.argv[decide.argv.index("--run-name") + 1] == "exp-decide"


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
    assert p.control_outputs["pope"]["reversed"] == "runs/exp-decide/pope.reversed.json"


def test_an_existing_checkpoint_can_be_evaluated_without_training():
    cfg = config(stages=[], evaluate={"checkpoint": "runs/old-model", "calibration_fit": "q_val.jsonl",
                                      "prefix": "v2_", "sets": {"old": "q_val.jsonl", "new": "x.jsonl"}})
    p = plan(cfg)
    assert [s.name for s in p.steps] == ["eval:old", "eval:new"]
    assert p.roles == {"old": "calibration_fit", "new": "heldout"}
    assert p.eval_outputs["new"] == "runs/old-model/v2_new.json"


def test_sets_are_labelled_by_what_they_were_used_for():
    p = plan(config(evaluate={"sets": {"fit": "q_val.jsonl", "watched": "pope.jsonl", "fresh": "new.jsonl"}}))
    assert p.roles == {"fit": "calibration_fit", "watched": "monitoring", "fresh": "heldout"}
    step = next(s for s in p.steps if s.name == "eval:fit")
    assert step.argv[step.argv.index("--role") + 1] == "calibration_fit"


def test_evaluation_runs_on_the_last_stage_and_fills_in_commands():
    p = plan(config())

    assert p.checkpoint == "runs/exp-decide"
    assert p.eval_outputs == {"pope": "runs/exp-decide/pope.json"}
    assert p.steps[-1].argv == ["echo runs/exp-decide exp"]


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
    (tmp_path / "exp-align").mkdir()
    (tmp_path / "exp-align" / "align_report.json").write_text("{}")
    (tmp_path / "exp-decide").mkdir()
    (tmp_path / "exp-decide" / "train_report.json").write_text("{}")

    run_plan(p, dry_run=True)
    out = capsys.readouterr().out
    assert "align: already done" in out and "decide: already done" in out

    run_plan(p, start_from="decide", dry_run=True)
    out = capsys.readouterr().out
    assert "align: already done" in out and "decide: already done" not in out

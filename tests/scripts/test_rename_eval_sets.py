"""scripts/rename_eval_sets.py: which files get which names, and a second run changing nothing."""
import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "rename_eval_sets", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "rename_eval_sets.py"))
assert _spec and _spec.loader
rename = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rename)

CASES = {
    # data-v2 test sets take LookFirst's config names; POPE is an external benchmark
    "data/v2/eval_coco_exist.jsonl": "data/v2/test_exist.jsonl",
    "data/v2/eval_coco_position.jsonl": "data/v2/test_position.jsonl",
    "data/v2/eval_coco_relation.jsonl": "data/v2/test_relation.jsonl",
    "data/v2/eval_coco_size.jsonl": "data/v2/test_size.jsonl",
    "data/v2/eval_gqa.jsonl": "data/v2/test_gqa.jsonl",
    "data/v2/eval_vqa_choice.jsonl": "data/v2/test_vqa_choice.jsonl",
    "data/v2/eval_vqa_yesno.jsonl": "data/v2/test_vqa_yesno.jsonl",
    "data/v2/eval_pope.jsonl": "data/v2/bench_pope.jsonl",
    "data/v2/dev_coco_exist.jsonl": None,
    "data/v2/coco_exist.jsonl": None,
    "data/v2/MANIFEST.json": None,
    "data/lv_bench/aokvqa.jsonl": None,            # stays: image paths inside point at lv_bench/images/
    "data/lv_bench/pope.unseen.jsonl": None,
    "data/pope.jsonl": None,                       # the old 300-question POPE of round 1
    # results of rounds 3 / 4, with every suffix
    "runs/v3-data2/eval/eval_gqa.json": "runs/v3-data2/eval/test_gqa.json",
    "runs/v3-data2/eval/eval_gqa.details.jsonl": "runs/v3-data2/eval/test_gqa.details.jsonl",
    "runs/v4-relation-tb/eval/eval_coco_size.mismatched.json": "runs/v4-relation-tb/eval/test_size.mismatched.json",
    "runs/v4-relation-tb/eval/eval_vqa_choice.reversed.details.jsonl":
        "runs/v4-relation-tb/eval/test_vqa_choice.reversed.details.jsonl",
    "runs/v3-data2/eval/eval_pope.mismatched.details.jsonl": "runs/v3-data2/eval/bench_pope.mismatched.details.jsonl",
    "runs/v3-data2/eval/dev_mix_half.json": None,
    # round 2: the v2eval_ prefix goes
    "runs/v2-align-lora/eval/v2eval_eval_coco_relation.reversed.json": "runs/v2-align-lora/eval/test_relation.reversed.json",
    "runs/v2-align-lora/eval/v2eval_eval_pope.details.jsonl": "runs/v2-align-lora/eval/bench_pope.details.jsonl",
    "runs/v2-align-lora/eval/v2eval_val_mix.mismatched.json": "runs/v2-align-lora/eval/val_mix.mismatched.json",
    "runs/v2-align-lora/eval/pope.json": None,
    "runs/v2-align-lora/eval/val_cocoqa.details.jsonl": None,
    "runs/v1-release-0.1/eval/gqa_by_type.txt": None,
    # laya-vision's benchmark
    "runs/v3-data2/eval/lv_aokvqa.json": "runs/v3-data2/eval/bench_lv_aokvqa.json",
    "runs/v3-data2/eval/lv_pope.details.jsonl": "runs/v3-data2/eval/bench_lv_pope.details.jsonl",
    "runs/v2-align-lora/eval/lv_vqav2_yesno.json": "runs/v2-align-lora/eval/bench_lv_vqav2_yesno.json",
    # compare/: every part of the name
    "runs/v3-data2/eval/compare/lv_aokvqa.unseen.json": "runs/v3-data2/eval/compare/bench_lv_aokvqa.unseen.json",
    "runs/v3-data2/eval/compare/v2_vs_v3.eval_coco_exist.axis.json":
        "runs/v3-data2/eval/compare/v2_vs_v3.test_exist.axis.json",
    "runs/v3-data2/eval/compare/v2_vs_v3.lv_scienceqa.json": "runs/v3-data2/eval/compare/v2_vs_v3.bench_lv_scienceqa.json",
    # only data/v<N>/ and eval/ directories
    "runs/v3-data2/stage2/eval_gqa.json": None,
    "docs/eval_gqa.md": None,
}


@pytest.mark.parametrize("old", sorted(CASES))
def test_each_file_gets_its_new_name(old):
    assert rename.rename_target(old) == CASES[old]


def test_new_names_are_left_alone():
    for new in CASES.values():
        if new:
            assert rename.rename_target(new) is None


def test_two_files_landing_on_one_name_stop_the_plan():
    with pytest.raises(SystemExit, match="would become"):
        rename.plan_renames(["runs/r/eval/v2eval_eval_gqa.json", "runs/r/eval/eval_gqa.json"])


def test_references_in_configs_and_docs_are_rewritten_once():
    yaml = ("evaluate:\n  prefix: v2eval_      # results as v2eval_<set>.json\n  sets:\n"
            "    eval_coco_exist: data/v2/eval_coco_exist.jsonl\n    eval_pope: data/v2/eval_pope.jsonl\n"
            "---\n  prefix: lv_\n  sets:\n    aokvqa: data/lv_bench/aokvqa.jsonl\n")
    new = rename.rewrite_text(yaml, yaml=True)
    assert new == ("evaluate:\n  sets:\n    test_exist: data/v2/test_exist.jsonl\n    bench_pope: data/v2/bench_pope.jsonl\n"
                   "---\n  prefix: bench_lv_\n  sets:\n    aokvqa: data/lv_bench/aokvqa.jsonl\n")
    doc = ("`runs/v2-align-lora/eval/v2eval_*.details.jsonl`, `runs/v3-data2/eval/lv_aokvqa.details.jsonl`, "
           "`compare/v2_vs_v3.lv_*.json`; | eval_vqa_choice | ...; pools coco_exist.jsonl, dev_coco_exist; "
           "scripts/data/lv_bench.py")
    assert rename.rewrite_text(doc) == (
        "`runs/v2-align-lora/eval/*.details.jsonl`, `runs/v3-data2/eval/bench_lv_aokvqa.details.jsonl`, "
        "`compare/v2_vs_v3.bench_lv_*.json`; | test_vqa_choice | ...; pools coco_exist.jsonl, dev_coco_exist; "
        "scripts/data/lv_bench.py")
    for text, is_yaml in ((new, True), (rename.rewrite_text(doc), False)):
        assert rename.rewrite_text(text, yaml=is_yaml) == text


def _tree(tmp_path, files):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _plan(tmp_path):
    return rename.plan(str(tmp_path / "data"), str(tmp_path / "runs"), str(tmp_path / "repo"))


def test_apply_then_a_second_run_changes_nothing(tmp_path):
    manifest = {"text_baselines": {"eval_coco_exist": {"all_files": 0.5}, "eval_pope": {"all_files": 0.5},
                                   "dev_gqa": {"all_files": 0.6}},
                "files": ["data/v2/eval_gqa.jsonl"]}
    _tree(tmp_path, {
        "data/v2/eval_gqa.jsonl": "{}\n", "data/v2/eval_pope.jsonl": "{}\n", "data/v2/dev_gqa.jsonl": "{}\n",
        "data/v2/MANIFEST.json": json.dumps(manifest, indent=1),
        "data/v2/eval_unknown.jsonl": "{}\n",
        "runs/v2-align-lora/eval/v2eval_eval_gqa.details.jsonl": "x\n",
        "runs/v3-data2/eval/compare/v2_vs_v3.eval_gqa.kind.json": "{}",
        "repo/configs/v3.yaml": "sets:\n  eval_gqa: data/v2/eval_gqa.jsonl\n",
        "repo/docs/research/notes.md": "see `eval_gqa`\n",
    })
    ops, warnings = _plan(tmp_path)
    assert {o.kind for o in ops} == {"rename", "rewrite"}
    assert warnings and "eval_unknown.jsonl" in warnings[0]
    rename.apply(ops)
    assert sorted(os.listdir(tmp_path / "data" / "v2")) == ["MANIFEST.json", "bench_pope.jsonl", "dev_gqa.jsonl",
                                                             "eval_unknown.jsonl", "test_gqa.jsonl"]
    assert os.listdir(tmp_path / "runs" / "v2-align-lora" / "eval") == ["test_gqa.details.jsonl"]
    assert os.listdir(tmp_path / "runs" / "v3-data2" / "eval" / "compare") == ["v2_vs_v3.test_gqa.kind.json"]
    m = json.loads((tmp_path / "data" / "v2" / "MANIFEST.json").read_text())
    assert set(m["text_baselines"]) == {"test_exist", "bench_pope", "dev_gqa"}
    assert m["files"] == ["data/v2/test_gqa.jsonl"]
    assert (tmp_path / "repo" / "configs" / "v3.yaml").read_text() == "sets:\n  test_gqa: data/v2/test_gqa.jsonl\n"
    assert _plan(tmp_path)[0] == []


def test_a_target_with_other_content_stops_everything(tmp_path):
    _tree(tmp_path, {"data/v2/eval_gqa.jsonl": "{}\n", "data/v2/test_gqa.jsonl": "{\"other\": 1}\n",
                     "data/v2/eval_pope.jsonl": "{}\n", "repo/docs/a.md": "eval_gqa\n"})
    with pytest.raises(SystemExit, match="different content"):
        _plan(tmp_path)
    assert (tmp_path / "data" / "v2" / "eval_pope.jsonl").exists()


def test_a_target_with_the_same_content_means_a_half_done_rename(tmp_path):
    _tree(tmp_path, {"data/v2/eval_gqa.jsonl": "{}\n", "data/v2/test_gqa.jsonl": "{}\n"})
    ops, _ = _plan(tmp_path)
    assert [(o.kind, os.path.basename(o.path)) for o in ops] == [("drop-duplicate", "eval_gqa.jsonl")]
    rename.apply(ops)
    assert os.listdir(tmp_path / "data" / "v2") == ["test_gqa.jsonl"]

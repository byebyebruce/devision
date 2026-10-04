"""Guards of the from-scratch build: no silently empty pools, no training data built before the benchmark,
and every data file of a config accounted for."""
import json

import pytest
import yaml
from check_configs import main as check_configs
from v2_common import BENCHMARK_FILES, check, require_benchmarks
from v2_gqa import TRAIN_SHARDS


def test_an_empty_file_fails_acceptance():
    assert check([], "mix:gqa", "equal") == ["mix:gqa: empty"]


def test_training_data_waits_for_the_benchmark(tmp_path):
    with pytest.raises(SystemExit, match="lv_bench"):
        require_benchmarks(str(tmp_path))
    for f in BENCHMARK_FILES:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("{}\n")
    require_benchmarks(str(tmp_path))   # all there: no error


def test_gqa_pictures_come_from_a_fixed_shard_list():
    assert TRAIN_SHARDS == ["train-00000-of-00021.parquet", "train-00001-of-00021.parquet",
                            "train-00002-of-00021.parquet"]


def test_a_config_whose_data_is_missing_or_empty_is_reported(tmp_path, capsys):
    root = tmp_path / "data"
    (root / "v2").mkdir(parents=True)
    (root / "v2" / "train.jsonl").write_text(json.dumps({"id": "x"}) + "\n")
    (root / "v2" / "dev.jsonl").write_text("")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"name": "x", "stages": [{"name": "s", "kind": "train", "init": "runs/a",
                                   "data": "data/v2/train.jsonl", "val": "data/v2/dev.jsonl"}],
                                   "evaluate": {"sets": {"t": "data/v2/test.jsonl"}}}))
    with pytest.raises(SystemExit):
        check_configs(["--root", str(root), str(cfg)])
    out = capsys.readouterr().out
    assert "dev.jsonl" in out and "test.jsonl" in out and "train.jsonl" not in out

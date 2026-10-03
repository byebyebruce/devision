# TODO

只记录已经确定要做的具体事项；方向、目标和备选想法不记在这里。完成后删除。

## 评测集改名：在主仓库执行一次

第 4 轮（`v4-relation-tb`）评测结束后、第 5 轮开始前，在主仓库运行 `uv run python scripts/rename_eval_sets.py --root data --runs runs`（先看 dry run），再加 `--apply`。之后对 `configs/v2-align-lora-testsets.yaml`、`v2-align-lora-lvbench.yaml`、`v3-data2.yaml`、`v3-data2-lvbench.yaml`、`v4-relation-tb.yaml` 各跑一次 `uv run devision-pipeline <配置> --dry-run`，确认全部显示 "already done, skipped"。仓库外的 `scripts/data/v2_build.py` 输出的文件名同步改成 `test_*` / `bench_pope`。

# TODO

## 下阶段目标：发布 v0.4（能力在全部评测集上整体超过 v0.3）

负责人 2026-10-21 定：走优质训练数据（自造或处理公开数据）补齐短板；教师蒸馏暂缓。方案见 `docs/experiments/2026-10-21-v04-plan.md`。每小时自动汇报进度（cc-connect cron `e6f6d043`）。

1. ~~补全评测~~（2026-10-21 完成：What'sUp、CountBench、CoGenT B、COCO val2014 计数、MMBench-en dev、CV-Bench、MME、NaturalBench、ScienceQA test 当开发集；v0.3 / r17 补评；冻结清单与规则见 `docs/experiments/2026-10-21-v04-eval-freeze.md`，判定脚本 `scripts/v04/accept.py`）。未做：SEED-Bench（27 GB，含视频）、LVIS（只有官方直连地址，待负责人决定）、`test_bind_gqa` / `test_count_fresh` 的本机重建。
2. ~~错误分析~~（What'sUp 几乎总选 on / in front of；计数从不答 5、10；ScienceQA 自然科学弱在磁铁、溶液浓度、粒子运动等示意图题）。
3. ~~data-n2~~（`scripts/data/n2/`，`data/n2/train_mix.MIX.json`：785k 题）。
4. ~~r19 / r20 / r21~~（训练日志 r19-n2、r20-science、r21-soft 三节）。
5. ~~发布 v0.4~~（2026-10-22：r21-soft，Hugging Face tag `v0.4`、ModelScope、GitHub tag `model-v0.4`；推理包不需要新版本，PyPI 0.2.1 可直接加载）。

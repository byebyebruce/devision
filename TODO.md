# TODO

## 下阶段目标：发布 v0.4（能力在全部评测集上整体超过 v0.3）

负责人 2026-10-21 定：走优质训练数据（自造或处理公开数据）补齐短板；教师蒸馏暂缓。方案见 `docs/experiments/2026-10-21-v04-plan.md`。每小时自动汇报进度（cc-connect cron `e6f6d043`）。

1. 补全评测：数据 B 生成计数、What'sUp、绑定、CoGenT B；下载并转换 MMBench-en dev、SEED-Bench 图片、CV-Bench、NaturalBench、MME；v0.3、r17 补评；冻结评测集与验收规则。
2. 错误分析（开发集）：失败簇 → 补什么数据。
3. data-n2：示意图 / 科学、空间、计数、存在（LVIS）、常识（V7W 全量、A-OKVQA、OK-VQA）、对比对；每块带测试与 MANIFEST，过质量门。
4. r19：从 v0.3 一遍 n1（降采样）+ 存在题 + data-n2；评测、逐题配对、lv_report、训练日志。
5. 需要时补洞（r20 / 权重插值）。
6. 发布 v0.4：按冻结的验收规则；CPU 延迟；`scripts/release/hf.py build|verify|publish`，PyPI 按 `docs/pypi-release.md`。

# TODO

## 下阶段目标：发布 v0.4（能力在全部评测集上整体超过 v0.3）

负责人 2026-10-21 定。方案见 `docs/experiments/2026-10-21-v04-plan.md`（强教师软标签 + 补全评测 + 视觉读出）。

1. r18-smooth 训完、评测，作为备选候选与参照。
2. 补全评测：用数据 B 生成计数、What'sUp、绑定、CoGenT B；下载并转换外部评测（MMBench-en dev、SEED-Bench 图片、CV-Bench、NaturalBench、MME）；v0.3 全部补评；冻结评测集与验收规则。
3. 教师打分脚本（Qwen3-VL，读选项 logit，带测试）；下载教师权重。
4. 教师门槛：各开发集约 500 题，教师 vs v0.3，逐集决定是否用教师。
5. 教师给训练池打分（80–100 万题）。
6. r19-distill：从 v0.3 一遍软目标训练；评测、逐题配对、lv_report、训练日志。
7. 多层 SigLIP 特征短程对照，通过则并入最后一轮。
8. 发布 v0.4：按冻结的验收规则；CPU 延迟单独测；`scripts/release/hf.py build|verify|publish`，PyPI 按 `docs/pypi-release.md`。

# TODO

## 下阶段目标：v0.5，真实需求场景的能力突破

负责人 2026-10-22 定：以真实场景的外部指标衡量，不再以自造或合成测试集为主。目标和口径见 `docs/experiments/2026-10-22-v05-goal.md`。主指标是 ImageJevBench（Benchmark Heaven），目标是上榜，并且 Capability 超过所有 1B 以下的模型。

1. exp78（分辨率 256 对比 384）跑完，按事先写下的规则判定（`docs/experiments/2026-10-22-exp78-resolution-plan.md`）。
2. 基线：v0.4 跑 ImageJevBench 公开 300 题，以及 VizWiz、RealWorldQA、SEED-Bench 图片部分。
3. 按 ImageJevBench 的五个用途分项定 v0.5 的具体数值目标，交负责人确认。

## 已完成：v0.4（2026-10-22 发布）

r21-soft；Hugging Face / ModelScope 的 `v0.4` 标签，GitHub 的 `model-v0.4` 标签。PyPI 0.2.1 可直接加载，不需要发新包。细节见训练日志。

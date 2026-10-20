# TODO

只记录已经确定要做的具体事项；方向、目标和备选想法不记在这里。完成后删除。

2026-10-20 负责人决定：不再使用旧 Mac、不复现封存结果，原来的 EXP76 V 阶段、EXP76 完整运行、v0.4-k 补测都撤下；在本机 CUDA 上从零重训，目标 v0.4 整体超过 v0.3。方案见 `docs/experiments/2026-10-20-r16-cuda-scratch-plan.md`。

## 1. 数据 A（本机重建）

- 下载 COCO train2014（HF `GAIA-URJC/COCO_2014` 的 `train2014.zip`），解压到 `data/coco/train2014/`。
- `UV_NO_SYNC=1 bash scripts/data/build_all.sh data`，最后的配置核对要通过。

## 2. v0.3 基线

- 在数据 A 的全部评测集（`test_*`、`bench_pope`、`bench_lv_*`，含对照）上评 `lukbit/devision`，结果放 `runs/v03-baseline/eval/`，作为所有比较的参照。

## 3. r16（数据 A）

- bf16 与 `amp: off` 的短程对照（同数据、同 seed，各约 2,000 步），定精度设置。
- 从 align 开始，按方案第 1 步的链路训练；每段评测、写训练日志、与 v0.3 逐题配对、跑 `scripts/lv_report.py`。

## 4. 数据 B 与 r17

- 后台下载并构建 `ext/`、`r13` 来源、exp14、r13b / r14 / r14b 混合；合成图脚本的 macOS 字体路径改成可配置。
- r17：v12-ext 扩展包 → r14b 混合 + 解冻 SigLIP 顶部 4 层；开训前实测显存。加入方案里列的数据修正。

## 5. 发布 v0.4

- 验收按方案的判据；CPU 延迟单独测；`scripts/release/hf.py build|verify|publish`，PyPI 包按 `docs/pypi-release.md`。

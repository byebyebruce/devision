# TODO

只记录已经确定要做的具体事项；方向、目标和备选想法不记在这里。完成后删除。

2026-10-20 负责人决定：不再使用旧 Mac、不复现封存结果（原来的 EXP76 V 阶段、EXP76 完整运行、v0.4-k 补测都撤下）；在本机 CUDA 上从 v0.3 接着训，目标 v0.4 整体超过 v0.3。方案见 `docs/experiments/2026-10-20-r16-cuda-plan.md`。

## 1. 数据

- 数据 A：`build_all.sh` 跑完，最后的配置核对通过。
- 数据 B：`scripts/data/prefetch_b.py` 下完文件；按方案的顺序构建 v11、ext、r13 来源、exp14、r13b、r14、r14b。
- r16 混合：r13 新来源（存在题按类别配平、计数 0..10 选项、科学题保留看图题）+ 60–70% r14b 回放；核对不看图规则接近随机。

## 2. v0.3 基线

- `configs/v03-baseline.yaml` 全部集合评完（`runs/v03/eval/`）；数据 B 就绪后加 `test_bind_gqa`、`bench_countbench`、`bench_whatsup`、`test_count_fresh`、CoGenT B。

## 3. r16

- r16 训练；评测、训练日志、与 v0.3 逐题配对（`devision-compare`）、`scripts/lv_report.py`。

## 4. 发布 v0.4

- 验收按方案的判据；CPU 延迟单独测；`scripts/release/hf.py build|verify|publish`，PyPI 包按 `docs/pypi-release.md`。

# TODO

只记录已经确定要做的具体事项；方向、目标和备选想法不记在这里。完成后删除。

2026-10-20 负责人决定：不再使用旧 Mac、不复现封存结果（原来的 EXP76 V 阶段、EXP76 完整运行、v0.4-k 补测都撤下）；在本机 CUDA 上从 v0.3 接着训、用新构造的训练集 data-n1，目标 v0.4 整体超过 v0.3。方案见 `docs/experiments/2026-10-20-r16-cuda-plan.md`。

## 1. 数据

- 数据 A：`build_all.sh` 跑完（评测集和回放用），最后的配置核对通过。
- data-n1：GQA 21 个分片下完；`n1_vqav2.py`、`n1_gqa.py`、`n1_mix.py` 依次运行，看 MANIFEST 的只看题目基线。

## 2. v0.3 基线

- `configs/v03-baseline.yaml` 全部集合评完（`runs/v03/eval/`）；数据 B 就绪后加 `test_bind_gqa`、`bench_countbench`、`bench_whatsup`、`test_count_fresh`、CoGenT B。

## 3. r16

- r16（v0.3 + data-n1）训练；评测、训练日志、与 v0.3 逐题配对（`devision-compare`）、`scripts/lv_report.py`。

## 4. 发布 v0.4

- 验收按方案的判据；CPU 延迟单独测；`scripts/release/hf.py build|verify|publish`，PyPI 包按 `docs/pypi-release.md`。

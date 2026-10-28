# exp128：r22 数据 + v0.4 尺寸题回放（预写，2026-10-28 17:10，exp127 训练中）

**背景**：r22-guiego 32 集里 31 集对 v0.4 不退，唯一显著退步是 **test_size −2.59 [−4.10, −1.18]**（84.4 vs 87.0；NLL 0.507 → 0.359、ECE 0.234 → 0.033，校准反而更好）。查训练混合：v0.4 的 `data/n2/soft_mix.jsonl` 有 coco-size 4,331 题，exp116 起所有 v05 混合（含 exp124 / r22、exp127）只剩 **129 题（3%）**，而其他 v0.4 来源都保留约 35%（coco-exist 在回放前也只有 3,076，是 exp124 修 POPE 的同一类问题）。

**假设**：尺寸退步来自尺寸题被欠采样，而不是 GUI / 第一人称数据的干扰；把 soft_mix 里其余 4,202 道 coco-size 加回去即可恢复。

**唯一变化**（相对 r22）：训练数据 164,040 → 168,242（`scripts/data/v05/exp128_mix.py`，`data/v05/exp128/MANIFEST.json`；soft_mix 已剔除评测图，复核 coco-size 与 test_size 图片重叠 0）。seed 22、init、超参、监测集、32 个评测集与 r22 完全相同（`configs/exp128-v4-size-replay.yaml`）。

**判据**（`devision-compare` 按图 95%）：
1. **修复**：test_size 对 v0.4 区间含 0 或为正，且对 r22 点估计 > 0。
2. **不伤别处**：r22 判据 1 的三个新用途对 v0.4 仍显著提升；其余 31 集对 v0.4 无显著退步。
- 1、2 都过 → exp128 配方（若 exp127 扩量也过，则 exp127 数据 + 尺寸回放）作为 `r23-*` 正式轮次。
- 1 不过 → 尺寸退步不是欠采样所致；在 GPU 空闲时补评 exp124 / exp126 的 test_size 判断是否种子，再查 GUI / ego 对尺寸的干扰。

**预算**：训练约 60 分钟 + 评测约 90 分钟。`scripts/v05/chain_exp128.sh 647323` 17:08 挂在 exp127 链之后（日志 `runs/v05-chain/chain_exp128.log`、`runs/exp128-v4-size-replay.out`），dry-run 通过；预计 exp127 约 20:10 评完，exp128 约 22:45 评完，评完自动跑 lv_report（`--prev runs/r22-guiego/eval`）。

# exp115：v0.4 + 图表 + UI + 计数补量，去掉新增科学（预写，2026-10-27 21:05）

**背景**：exp114（exp81 + 科学 16,343 + ScreenQA 2,048）科学没涨（`bench_lv_scienceqa` +0.57 [−0.48, +1.62]），CountBench 显著退步（50.1→44.4，−5.70 [−9.57, −1.83]）；图表 dev 58.6%（exp81 52.7%）、UI dev 54.7→66.8%。计数类（tallyqa + coco-count + clevr）在 exp114 只占 4.4%（v0.4 的 soft_mix 为 7.7%）。

**假设**：CountBench 退步来自计数信号被稀释。去掉 exp114 新增的科学题、补 3,000 道 COCO 计数题（`data/n2/count_train.jsonl`，train2014，与 exp81 不重复），计数占比回到约 10%，可以保住图表、UI 增益并消除 CountBench 退步。这一版作为后续加入 GUI 操作 / 第一人称数据（`data/v05/gui`、`ego`，构建中）的多用途底座。

**唯一变化**（相对 exp114）：训练数据 66,391 → 53,048（exp81 48,000 + ScreenQA 2,048 + COCO 计数 3,000，`scripts/data/v05/exp115_mix.py`，seed 115）；init、学习率、视觉解冻、CE、1 epoch、评测集全部不变。训练图片 id 与全部评测集交集 0（本次复核；A-OKVQA 文件名为哈希，无法按 COCO id 核对，`.unseen` 子集另看）。

**归因对照**：接续脚本先在 exp81-chart 上评 CountBench（`runs/exp81-chart/eval/bench_countbench.json`），区分退步来自图表数据还是科学数据。

**判据**（与 v0.4 `runs/r21-soft/eval`、exp114 逐题配对，`devision-compare` 按图配对 95% 区间）：
1. CountBench 相对 v0.4 区间上界 ≥ 0（无显著退步）；
2. 图表 dev ≥ exp81 −2pp（≥ 50.7%），UI dev ≥ v0.4（54.7%）；
3. MMBench / MME / CV-Bench / NaturalBench / RealWorld / What'sUp / POPE / lv 四集相对 v0.4 无显著退步（区间上界不在 0 以下）。
全过 → 作为 GUI / ego 数据实验的底座（exp116 = exp115 + GUI 动作题）；1 不过 → 计数退步不是稀释造成，查图表数据对计数的干扰；2/3 不过 → 记录、回到 exp81 底座。

**预算**：训练约 15 分钟 + 评测约 35 分钟，单卡，nohup 接续（`scripts/v05/chain_exp115.sh`，exp114 评测结束自动启动）。不拟合发布温度。

## 结果（2026-10-27 22:20）

训练 + 评测完整跑完（`runs/exp115-v4-chart-ui-count/{train,eval}`，SwanLab `29zzs54n`）。相对 v0.4 逐题配对（`devision-compare`，按图 95% 区间）：

| 集合 | v0.4 | exp115 | 差 [95%] |
|---|---|---|---|
| CountBench | 50.10 | 47.45 | −2.65 [−6.72, +1.22] |
| lv ScienceQA | 82.07 | 80.35 | **−1.72 [−2.96, −0.48]** |
| dev_sqa | 84.63 | 82.55 | **−2.08 [−3.31, −0.88]** |
| MMBench | 62.67 | 62.11 | −0.56 [−1.40, +0.26] |
| MME | 64.74 | 65.33 | +0.59 [−0.38, +1.60] |
| CV-Bench | 55.12 | 54.81 | −0.30 [−1.40, +0.87] |
| NaturalBench | 62.40 | 62.93 | +0.53 [−0.09, +1.16] |
| What'sUp | 46.71 | 45.37 | −1.34 [−3.29, +0.73] |
| POPE（= lv POPE） | 86.53 | 86.93 | +0.40 [−0.06, +0.86] |
| lv VQAv2 是非 | 75.59 | 75.75 | +0.16 [−0.48, +0.85] |
| lv A-OKVQA | 63.36 | 63.80 | +0.44 [−1.23, +1.93] |
| RealWorld | — | 46.34 | v0.4 无逐题明细，基线由 `configs/r21-soft-guiego.yaml` 补评（exp116 接续脚本） |

图表 dev 58.9%（exp81 52.7%），UI dev 70.7%（v0.4 54.7%），protect 79.8%。

**归因对照**：exp81-chart 单独在 CountBench 上相对 v0.4 −6.72 [−10.79, −2.65]——计数退步来自图表数据，不是 exp114 的科学数据；exp115 补 3,000 道计数题后相对 exp114 +3.05 [−1.02, +7.13]，相对 v0.4 不再显著。

**裁决**：判据 1 过、判据 2 过、**判据 3 不过**（ScienceQA 显著退步 −1.72；去掉科学题后科学回落，exp114 带科学时为 +0.57）。按预写"2/3 不过 → 记录、回到 exp81 底座"；但 exp81 自身 CountBench −6.72 显著退步，同样不满足判据 3，回到 exp81 并不更好。实际处理：exp116（GUI + ego，已预写并接续）仍在 exp115 底座上做，以隔离 GUI / ego 的增量（exp116 预案已写明底座未过时只与 v0.4 比旧能力）；ScienceQA 的修复（把 exp114 的科学题加回）作为下一步单独实验，不和 GUI / ego 混在一次改动里。

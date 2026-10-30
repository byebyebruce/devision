# exp124：exp122 + 存在类是非题回放（预写，2026-10-28 09:05，exp123 训练中）

**背景**：exp122（[文档](2026-10-28-exp122-v4-spatial-balance.md)）配平后 What'sUp / CV-Bench 判据过，但对 v0.4 POPE −0.71 [−1.36, −0.06] 显著、lv VQAv2 是非 −0.74（上界 +0.04）。v0.4 混合含 40,000 道 coco-exist，exp122 混合只剩 3,076（GUI / ego / 图表扩量稀释）。

**改动**（`scripts/data/v05/exp124_mix.py`，seed 124，`data/v05/exp124/MANIFEST.json`）：exp122 训练文件不改，从 `data/n2/soft_mix.jsonl`（v0.4 训练混合，已剔除评测图）中未在混合里的行加入 coco-exist 6,000（true / false 各半）+ VQAv2 是非 3,000。共 164,040 题（约 20.5k 步）；init、学习率、视觉解冻、CE、1 epoch、评测集全部同 exp122。

**对照**：exp122（只差 9k 回放，逐题配对）、v0.4；exp123 结果出来后另与之配对。

**判据**（`devision-compare` 按图 95%）：
1. **POPE 修复**：bench_pope 对 v0.4 区间含 0 或为正，且对 exp122 点估计 > 0；lv VQAv2 是非对 v0.4 不显著退步。
2. **新用途保住**：test_gui_shop_state、test_gui_state、test_ego 对 exp122 无显著退步（GUI 下界 > −2pp）。
3. **其他旧能力**：What'sUp / CV-Bench / ScienceQA / dev_sqa / MMBench / MME / CountBench / RealWorld / NaturalBench / EgoThink 对 v0.4 无显著退步。
- 1–3 都过 → exp124 配方（按 exp123 的 GUI 占比结论调整 GUI 量）定 `r22-*` 正式轮次。
- 1 不过 → POPE 退步不是存在题占比造成的，查 GUI / ego 题里的 yes 偏置（yes ratio），不再加回放量。
- 2 不过 → 回放减半。

**预算**：训练约 55 分钟 + 评测约 55 分钟。`scripts/v05/chain_exp124.sh 610369 604058` 09:02 挂在 exp123 流水线之后（日志 `runs/v05-chain/chain_exp124.log`、`runs/exp124-v4-exist-replay.out`），dry-run 通过，预计 12:00 前后评完。不拟合发布温度。

## 部分结果（2026-10-28 11:05，评测 5/18 集）

训练 10:48 完成，评测中。对 exp122 逐题配对（`devision-compare`，按图 95%）：

| 集合 | exp122 | exp124 | 差 [95%] |
|---|---|---|---|
| test_gui_shop_state | 69.4 | 69.6 | +0.26（含 0） |
| test_gui_state | 52.4 | 49.9 | **−2.47 [−4.29, −0.58]**（aitw −1.80 含 0；mind2web −3.06 [−5.75, −0.37]） |
| test_gui | 77.3 | 75.3 | −1.95 [−3.64, −0.26]（两个来源各自含 0） |
| test_ego | 68.7 | 68.8 | +0.12（含 0） |
| bench_egothink | 77.2 | 77.0 | −0.17 |

**判据 2 不过**（test_gui_state 下界 −4.29 < −2pp）。退步主要在网页 Mind2Web（256 输入下本就无看图增益，偏文字先验）；注意 exp122 / exp124 种子不同（122 / 124），有种子噪声的成分，没有做同种子复现。按预写规则「2 不过 → 回放减半」：exp125（[文档](2026-10-28-exp125-v4-exist-replay-half.md)）已挂在 exp124 之后。判据 1（POPE）待评完再判；若 1 不过，exp125 的意义只剩剂量对照，下一轮决定是否中止。

## 裁决（2026-10-28 12:05，18/18 集评完）

其余 13 集（`runs/exp124-v4-exist-replay/eval/compare/*.txt`，`devision-compare --by source`，按图 95%）：

| 集合 | v0.4 | exp122 | exp124 | 对 exp122 | 对 v0.4 |
|---|---|---|---|---|---|
| bench_pope | 86.5 | 85.8 | 86.1 | +0.30 [−0.28, +0.90] | −0.41 [−0.97, +0.16] |
| bench_lv_vqav2_yesno | 75.6 | 74.9 | 75.6 | **+0.80 [+0.10, +1.52]** | +0.06 [−0.74, +0.83] |
| bench_lv_aokvqa | 63.4 | 62.0 | 62.6 | +0.53 | −0.79 [−2.64, +1.14] |
| bench_lv_scienceqa | 82.1 | 84.0 | 82.7 | **−1.24 [−2.24, −0.14]** | +0.67 [−0.62, +2.00] |
| dev_sqa | 84.6 | 85.0 | 84.8 | −0.20 | +0.20 |
| bench_whatsup | 46.7 | 50.9 | 49.8 | −1.10 [−3.05, +0.98] | **+3.05 [+0.61, +5.49]** |
| bench_cvbench | 55.1 | 56.2 | 56.7 | +0.53 | **+1.63 [+0.30, +3.03]** |
| bench_mmbench | 62.7 | 62.2 | 62.6 | +0.37 | −0.09 |
| bench_mme | 64.7 | 64.9 | 64.4 | −0.42 | −0.29 |
| bench_countbench | 50.1 | 51.5 | 51.5 | 0.00 | +1.43 |
| bench_realworld | 46.7 | 45.5 | 47.5 | +2.04 | +0.85 |
| bench_naturalbench | 62.4 | 62.8 | 62.9 | +0.11 | +0.46 |
| bench_egothink | 74.7 | 77.2 | 77.0 | −0.17 | +2.32 |

POPE 误报 503 → 440（v0.4 411），yes ratio 0.470 → 0.459（v0.4 0.457）：回放确实压下了 exp122 多出来的 "yes"。

- **判据 1 过**：POPE 对 v0.4 区间含 0、对 exp122 点估计 +0.30；lv VQAv2 是非对 v0.4 +0.06。
- **判据 2 不过**（见上节，test_gui_state −2.47、test_gui −1.95）。
- **判据 3 过**：10 个旧集合对 v0.4 无显著退步（What'sUp、CV-Bench 显著提升）。另有 ScienceQA 对 exp122 −1.24 显著（对 v0.4 仍 +0.67），记下。
- **结论**：存在题回放能修 POPE，代价是 GUI −2 左右——但 exp122 / exp124 种子不同，这个 −2 可能是种子噪声。exp125（半剂量）按计划进行；按 exp125 预写规则中"GUI 退步另以同种子复现确认"一条，exp126（exp122 数据 + seed 124，[文档](2026-10-28-exp126-v4-seed-control.md)）已挂在 exp125 之后，测种子噪声的大小。

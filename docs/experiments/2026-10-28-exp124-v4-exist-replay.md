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

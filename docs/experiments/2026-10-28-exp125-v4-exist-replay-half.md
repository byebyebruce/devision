# exp125：exp122 + 存在类是非题回放减半（预写，2026-10-28 11:05，exp124 评测中）

**背景**：exp124（exp122 + 9k 存在 / 是非回放，[文档](2026-10-28-exp124-v4-exist-replay.md)）评测前 5 集中 test_gui_state 对 exp122 −2.47 [−4.29, −0.58]，判据 2 不过；按其预写规则回放减半。POPE（判据 1）在 exp124 评完后才知道。

**改动**：`scripts/data/v05/exp124_mix.py --seed 125 --exist 3000 --vqav2 1500 --out v05/exp125`（脚本只加了 `--out` 参数）。exp122 训练文件不改，加 coco-exist 1,500 真 + 1,500 假 + VQAv2 是非 1,500，共 159,540 题（`data/v05/exp125/MANIFEST.json`，sha256 5c17d53f…）。init、学习率、视觉解冻、CE、1 epoch、评测集同 exp124，seed 125。

**对照**：exp122、exp124（逐题配对，剂量 0 / 4.5k / 9k）、v0.4。

**判据**（`devision-compare` 按图 95%）：
1. POPE：对 v0.4 区间含 0 或为正；lv VQAv2 是非对 v0.4 不显著退步。
2. GUI / ego：test_gui_shop_state、test_gui_state、test_ego 对 exp122 下界 > −2pp。
3. 其余旧集合对 v0.4 无显著退步（同 exp124 判据 3）。
- 1–3 都过 → exp125 配方定 `r22-*`。
- 1 不过、exp124 的 1 过 → POPE 需要 9k 剂量，GUI 退步另以同种子复现确认是否为种子噪声后再定。
- exp124 的 1 也不过 → 回放不是 POPE 的原因，exp125 只作剂量记录；转查 GUI / ego 数据的 yes 偏置。

**预算**：训练约 55 分钟 + 评测约 45 分钟。`scripts/v05/chain_exp125.sh 617022 612385` 11:02 挂在 exp124 流水线之后（日志 `runs/v05-chain/chain_exp125.log`、`runs/exp125-v4-exist-replay-half.out`），dry-run 通过，预计 13:15 前后评完。不拟合发布温度。

**进度（12:05）**：训练 10,900 / 19,943 步（loss 0.70–0.73，与前几轮同期一致）。exp124 已评完：判据 1（POPE）过、2 不过、3 过；因此本实验落在"exp124 的 1 过"分支，同种子对照 exp126 已挂在本实验之后。

## 结果（2026-10-28 13:15，18/18 集评完）

`runs/exp125-v4-exist-replay-half/eval/compare/*.txt`（`devision-compare --by source`，按图 95%，差值 = exp125 − 对照，pp）：

| 集合 | exp125 | 对 exp122 | 对 exp124 | 对 v0.4 |
|---|---|---|---|---|
| bench_pope | 85.8 | −0.07 [−0.69, +0.56] | −0.37 [−0.97, +0.22] | **−0.78 [−1.41, −0.17]** |
| lv VQAv2 是非 | 75.0 | +0.14 | −0.65 [−1.46, +0.12] | −0.59 [−1.41, +0.21] |
| test_gui_state | 50.3 | **−2.08 [−4.16, −0.19]** | +0.39 | +25.00 |
| test_gui_shop_state | 67.3 | −2.07 [−4.78, +0.52] | −2.33 [−4.65, +0.13] | +45.87 |
| test_gui | 76.0 | −1.23 [−2.92, +0.39] | +0.71 | +37.47 |
| test_ego | 68.8 | +0.10 | −0.02 | +34.11 |
| ScienceQA（lv） | 82.2 | **−1.81 [−2.86, −0.76]** | −0.57 | +0.10 |
| MME | 65.9 | +1.01 [+0.00, +2.06] | **+1.43** | **+1.14 [+0.08, +2.23]** |
| What'sUp | 52.1 | +1.22 | **+2.32** | **+5.37** |
| RealWorld | 48.4 | **+2.90** | +0.85 | +1.70 |
| EgoThink | 78.1 | +0.99 | +1.16 | **+3.48 [+0.17, +6.62]** |
| CV-Bench / MMBench / NaturalBench / CountBench / dev_sqa / A-OKVQA | — | 均含 0 | 均含 0 | 均含 0 |

POPE 误报 433（exp124 440、v0.4 411），recall 0.812。

**裁决**：判据 1 不过（POPE 对 v0.4 −0.78 显著，4.5k 剂量不够）；判据 2 不过（test_gui_state、test_gui_shop_state 下界 < −2pp）；判据 3 过（其余旧集合对 v0.4 无显著退步，MME / What'sUp / EgoThink 显著提升）。按预写规则落在"1 不过、exp124 的 1 过"分支 → **POPE 需要 9k 剂量**。

**观察**：exp125（seed 125）与 exp124（seed 124）对 exp122（seed 122）的 GUI / ScienceQA 退步方向、幅度几乎相同（test_gui_state −2.08 vs −2.47、ScienceQA −1.81 vs −1.24），而 exp125 对 exp124 都不显著 → 回放剂量减半没有收回 GUI，exp122 这个种子偏高的解释更可信。由 exp126（exp122 数据 + seed 124，13:09 已开训）确认。

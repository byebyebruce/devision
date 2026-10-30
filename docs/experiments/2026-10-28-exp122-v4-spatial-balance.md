# exp122：exp121 + n2-spatial 剩余全部（关系词重新配平）（预写，2026-10-28 06:00，exp121 评测前）

**背景**：exp120 加入 4,000 道 gold 只为 in front of / behind 的深度题后，What'sUp 对 exp119 −12.93 [−17.07, −8.78]，left_of / right_of 准确率 ~50% → 1.5%（模型不再选 left / right）。CV-Bench 3D 同时 +2.3 显著。原因是选项先验失衡，不是深度能力本身（见 [exp120](2026-10-28-exp120-v4-depth-ego.md) 补齐一节）。exp121（exp120 + 手机 GUI 24k）以 exp120 训练文件为底，预计继承该退步。

**改动**（`scripts/data/v05/exp122_mix.py`，seed 122，`data/v05/exp122/MANIFEST.json`）：以 exp121 训练文件为底（不改动），加入 `data/n2/spatial.jsonl` 中尚未在混合里的全部 10,151 题（depth 的 left 2,379 / right 2,374 / front 349 / behind 337；vertical 的 left / right / on / under 各约 1,180）。混合后 n2-spatial 共 14,599 题，各类别比例与源文件一致（depth 四类各约 2,440，vertical 四类各约 1,215）。共 155,040 题（约 19.4k 步）；init、学习率、视觉解冻、CE、1 epoch、评测集全部同 exp121。

**对照**：exp121（直接底座，逐题配对）、exp119（深度回放之前）、v0.4；所有集合配错图 / 倒序对照。

**判据**（`devision-compare` 按图 95%）：
1. **左右恢复**：What'sUp 相对 exp121 下界 > 0，且相对 v0.4 不显著退步（区间含 0 或为正）；left_of / right_of 子项各 ≥ 40%。
2. **深度保住**：CV-Bench 相对 v0.4 不显著退步，3D-depth 点估计不低于 v0.4。
3. **其他不退**：GUI 三集、test_ego / EgoThink、ScienceQA / dev_sqa、MMBench 及其余 bench 相对 exp121 无显著退步。
- 1–3 都过 → exp122 配方即 `r22-*` 正式轮次配方（GUI 量按 exp121 的判据决定是否保留 24k 扩量；若 exp121 判据 1 不过，r22 从 exp122 混合中去掉那 24k）。
- 1 不过 → 深度题改成两选项（front / behind only）再试，不加步数。
- 2 不过 → 深度回放减半、配平保留。

**预算**：训练约 50 分钟 + 评测约 55 分钟。`scripts/v05/chain_exp122.sh 594252 587690` 06:02 挂在 exp121 流水线之后（日志 `runs/v05-chain/chain_exp122.log`、`runs/exp122-v4-spatial-balance.out`），dry-run 通过，预计 08:45 前后评完。不拟合发布温度。

## 部分结果（2026-10-28 08:00，训练完成，18 集已评 8 集）

`devision-compare` 按图 95%，B = exp122：

| 集合 | exp122 | 对 exp121 | 对 v0.4 |
|---|---|---|---|
| test_gui_state | 52.4 | +2.34 [+0.45, +4.22] | +27.08 |
| test_gui_shop_state（未见网站） | 69.4 | +2.20 [−0.26, +4.52] | +47.93 |
| test_gui | 77.3 | +1.43 [−0.13, +3.12] | +38.70 |
| test_ego | 68.7 | **−1.11 [−2.01, −0.21]** | +34.00 |
| bench_egothink | 77.2 | +1.32 [−0.99, +3.81] | +2.48 [−0.66, +5.63] |
| bench_lv_scienceqa | 84.0 | +0.67 [−0.43, +1.76] | +1.91 [+0.67, +3.15] |
| dev_sqa | 85.0 | +0.99 [−0.35, +2.45] | +0.40 [−1.04, +1.99] |
| bench_realworld | 45.5 | +0.51 [−2.56, +3.57] | −1.19 [−4.08, +1.53] |

判据 3 目前有 test_ego 一项小幅显著退（−1.1，仍比 exp120 的 68.1 高 0.6）；判据 1（What'sUp 左右）、2（CV-Bench）与 MMBench / MME 等待剩余集合，08:45 前后补齐后裁决。

## 补齐与裁决（2026-10-28 09:05，18/18）

`devision-compare` 按图 95%（`runs/exp122-v4-spatial-balance/eval/compare/`），B = exp122：

| 集合 | exp122 | 对 exp121 | 对 v0.4 |
|---|---|---|---|
| bench_whatsup | 50.9 | **+18.41 [+14.02, +22.56]** | **+4.15 [+1.71, +6.59]** |
| bench_cvbench | 56.2 | −1.29 [−2.65, +0.08] | +1.10 [−0.15, +2.35] |
| bench_mmbench | 62.2 | +0.70 [−0.14, +1.51] | −0.47 [−1.40, +0.51] |
| bench_mme | 64.9 | +0.17 [−0.80, +1.22] | +0.13 [−0.93, +1.18] |
| bench_naturalbench | 62.8 | +0.15 | +0.36（均含 0） |
| bench_countbench | 51.5 | +1.02 | +1.43（均含 0） |
| bench_pope / bench_lv_pope | 85.8 | −0.48 [−1.13, +0.18] | **−0.71 [−1.36, −0.06]** |
| bench_lv_vqav2_yesno | 74.9 | −0.47 | −0.74 [−1.54, +0.04] |
| bench_lv_aokvqa | 62.0 | −0.70 | −1.32 [−3.25, +0.53] |

What'sUp 子项（v0.4 / exp121 / exp122）：left_of 42.4 / 2.9 / **49.8**，right_of 50.2 / 3.4 / **53.7**，on 73.8 / 88.3 / 68.0，under 33.0 / 23.3 / 53.4，in-front_of 36.3 / 77.5 / 40.2，behind 45.1 / 57.8 / 38.2。CV-Bench 3D-depth 54.8 / 56.0 / 55.2，3D-distance 47.3 / 51.7 / 52.0。

**裁决**：
- 判据 1 **过**（What'sUp 对 exp121 下界 +14.0，对 v0.4 显著为正，左右各 ≥ 40%）。exp120 的左右崩溃确认是选项先验失衡，配平即可修复。
- 判据 2 **过**（CV-Bench 对 v0.4 含 0，3D-depth 55.2 ≥ 54.8）。
- 判据 3 **不全过**：test_ego 对 exp121 −1.11 [−2.01, −0.21]（仍高于 exp120 的 68.1）；其余对 exp121 均含 0。另外对 v0.4 POPE −0.71 显著（继承自 GUI / ego 扩量后存在类题占比被稀释：v0.4 混合 40k coco-exist，exp122 只有 3,076）。
- → 不直接定 r22。按 exp123 预写规则，选出的配方须对 v0.4 全部旧集合不显著退步，故先做针对性回放 **exp124**（exp122 + 存在类是非题 9k，[文档](2026-10-28-exp124-v4-exist-replay.md)），不加步数重复。GUI 占比（33% vs 23%）仍由 exp123 与 exp122 配对决定。

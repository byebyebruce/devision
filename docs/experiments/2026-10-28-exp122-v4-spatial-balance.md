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

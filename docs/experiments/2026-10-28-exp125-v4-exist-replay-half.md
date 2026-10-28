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

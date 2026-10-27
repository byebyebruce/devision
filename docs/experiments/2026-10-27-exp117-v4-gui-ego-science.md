# exp117：exp116 + 科学题回放（预写，2026-10-27 23:02，exp116 结果出来之前）

**背景**：exp114（含科学 16k）ScienceQA +0.57 [−0.48, +1.62]，未退；exp115 去掉科学后 ScienceQA **−1.72 [−2.96, −0.48]**、dev_sqa −2.08 显著退步，但计数通过补 3,000 道计数题恢复（CountBench −2.65 不再显著）。exp116 在 exp115 上加 GUI / ego，底座的科学退步预计仍在。负责人要求 GPU 不空闲，所以在 exp116 评完之前预写本实验，由 `scripts/v05/chain_exp117.sh` 在 exp116 流水线退出后自动开跑。

**假设**：把 exp114 的科学题（TQA 5,366 + AI2D 5,678 + ScienceQA 4,797 = 15,841，去掉与 exp116 重复的 id）加回 exp116 的混合，ScienceQA 回到不退步，同时计数（exp115 的补充）、GUI / ego 的收益保持。

**唯一变化**（相对 exp116）：训练数据 77,048 → 92,889（`scripts/data/v05/exp117_mix.py`，seed 117，`data/v05/exp117/MANIFEST.json`）；seed 116→117（打乱顺序不同）。init（v0.4）、学习率、视觉解冻、CE、1 epoch、监测集、评测集与 exp116 完全相同（GUI / ego 的 dev、test 用 `data/v05/exp116/` 的冻结副本）。步数约 11.6k（exp116 9.6k）。

**判据**（`devision-compare` 按图配对 95% 区间）：
1. **科学**：`bench_lv_scienceqa` 和 `dev_sqa` 相对 v0.4 无显著退步（区间上界 ≥ 0），相对 exp116 提升（区间下界 > 0 时记为修复成立；exp116 若本身未退，则只要求不退）。
2. **计数**：CountBench 相对 v0.4 无显著退步（exp114 的科学 + 图表组合曾 −5.70，这里要确认补计数后仍守住）。
3. **新用途**：`test_gui` / `test_ego` 相对 exp116 无显著退步（科学题不挤占 GUI / ego）；相对 v0.4 的判据同 exp116 判据 1、2。
4. **旧能力**：同 exp116 判据 3（MMBench / MME / CV-Bench / NaturalBench / RealWorld / What'sUp / POPE / lv 四集相对 v0.4 无显著退步；图表 dev ≥ 50.7%、UI dev ≥ 54.7%）。
- 1–4 全过 → 作为 v0.5 候选配方，下一步扩 GUI（aitw-shopping 构建完成后）/ ego 全量并定正式轮次 `r22-*`。
- 1 不过 → 科学退步不是"缺科学数据"引起，查 exp115 起新增的哪部分（ScreenQA / 计数 / 图表）挤占科学（逐源消融），不再加科学题量。
- 2 不过 → 科学与计数存在冲突，下一步调比例。
- 3 不过 → 降科学比例。
- exp116 若 GUI / ego 判据本身不过，本实验只用来回答科学 / 计数问题，不作为 GUI / ego 的结论。

**预算**：训练约 46 分钟 + 评测约 50 分钟，单卡；接续脚本 `scripts/v05/chain_exp117.sh <exp116 pipeline pid>`，日志 `runs/v05-chain/chain_exp117.log`、`runs/exp117-v4-gui-ego-science.out`。不拟合发布温度。

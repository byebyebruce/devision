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

# exp135：从 v0.4 续训时视觉顶 4 层用低学习率 1e-6（预写，2026-10-29 08:10，exp134 评测中，不算轮次）

**背景**：exp132（视觉顶 4 层 lr_vision 5e-6）新用途强，但 test_vqa_yesno 对 v0.4 −1.60 显著；exp134（同数据同种子、视觉冻结）把 POPE / VQA 是非拉回 v0.4（−0.19 / −0.70，都含 0），但对 exp132 回吐 test_chartqa −9.48、test_gui_shop_state −6.59、test_gui_state −2.86、test_ego −2.64（都显著）。新用途要视觉适配，自然照片要视觉不漂。按 exp134 预写分支「1 过、2 不过 → 降 lr_vision」。

**假设**：视觉顶 4 层用 1e-6（exp132 的 1/5），新用途保住大部分（对 exp132 ≥ −2.0），POPE / VQA 是非仍不显著退步。

**改动**：`configs/exp135-v4-lowlr-vision.yaml` = exp132 只改 `lr_vision: 1.0e-06`（数据 `data/v05/exp132/train.jsonl`、seed 23、`vision_unfreeze_top: 4`），评测 33 集同 exp134。exp132（5e-6）/ exp135（1e-6）/ exp134（0）构成同数据同种子的视觉学习率剂量曲线。

**对照**：v0.4（正式门槛）；exp132、exp134（归因）。

**判据**（`devision-compare` 按图 95%）：
1. bench_pope、test_vqa_yesno 对 v0.4 无显著退步。
2. test_gui_state、test_gui_shop_state、test_ego、test_gui、test_chartqa 对 v0.4 显著提升，且对 exp132 点估计都 ≥ −2.0。
3. 其余集合对 v0.4 无显著退步。
- 1–3 过 → r23 = 本配方换种子（正式轮次，评完 lv_report + 独立校准）。
- 1 过、2 不过 → 两者在视觉学习率上不可兼得；下一步只对截图 / 图表 / 第一人称样本更新视觉（按来源门控视觉梯度），或 r23 候选用 exp134 冻结配方 + 加量新用途数据。
- 1 不过 → 1e-6 已足以让自然照片漂移；转向按来源门控视觉梯度。

**预算**：训练约 65 分钟 + 33 集评测约 2 小时。`scripts/v05/chain_exp135.sh 745311`（日志 `runs/v05-chain/chain_exp135.log`）08:07 挂在 exp134 链之后，exp134 约 09:00 评完，exp135 预计 12:00 前后评完。

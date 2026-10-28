# exp134：从 v0.4 续训时冻结视觉编码器（预写，2026-10-29 04:15，exp132 评测中、exp133 排队，不算轮次）

**背景**：从 v0.4 续训的所有运行（exp114–exp132、r22）都用 `vision_unfreeze_top: 4, lr_vision: 5e-6`，这些运行的 POPE / VQA 是非对 v0.4 一直是 −0.3 到 −1.6。数据层面的修补（回放 exp124/125/128/129、离线蒸馏 exp132）和权重插值（exp130）都没让 POPE 与 test_vqa_yesno 同时不退步（exp132：POPE −0.47 [−1.00, +0.10] 含 0，但 test_vqa_yesno −1.60 [−3.17, −0.20]）。POPE（物体在不在）主要靠视觉特征；视觉层在新用途数据（截图、第一人称、图表）上漂移，可能拖累自然照片。**冻结视觉编码器从没在 v0.4 续训上试过**（v0.4 自己的视觉权重在 checkpoint 里，`--init` 会全部载入，冻结就是保留它们不变）。

**假设**：冻结视觉后 POPE / VQA 是非回到 v0.4 水平；新用途增益主要来自投影层 + 文本侧 LoRA，回吐不多。

**改动**：`configs/exp134-v4-frozen-vision.yaml` = exp132 去掉 `vision_unfreeze_top` / `lr_vision`（数据 `data/v05/exp132/train.jsonl`、seed 23、其余超参不变），评测加 test_chartqa。对 exp132 的差就是冻结视觉这一个变量。

**对照**：v0.4（正式门槛）；exp132（同数据同种子，归因）。

**判据**（`devision-compare` 按图 95%）：
1. 旧能力：bench_pope 与 test_vqa_yesno 对 v0.4 无显著退步。
2. 新用途：test_gui_state、test_gui_shop_state、test_ego、test_gui、test_chartqa 对 v0.4 显著提升，且对 exp132 的点估计都不低于 −2.0。
3. 其余全部集合对 v0.4 无显著退步。
- 1–3 过 → r23 候选配方改为冻结视觉（种子另定），再看 exp133 切块是否叠加。
- 1 过、2 不过 → 视觉漂移是 POPE 退步的来源，但新用途要视觉适配；下一步试只解冻顶 1–2 层或 lr_vision 降到 1e-6。
- 1 不过 → POPE 漂移不来自视觉层，记录；下一步看文本侧（LoRA 秩 / 学习率）。

**预算**：训练约 60 分钟（视觉冻结，比 exp132 略快）+ 33 集带对照评测约 2.5 小时。`scripts/v05/chain_exp134.sh <exp133 链 pid>` 挂在 exp133 之后，预计 2026-10-29 15:00 前后评完并自动跑 lv_report。

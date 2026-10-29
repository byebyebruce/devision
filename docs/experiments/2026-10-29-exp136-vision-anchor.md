# exp136：视觉特征锚定——自然照片的 SigLIP 特征拉向 v0.4，截图 / 图表照常适配（预写，2026-10-29 10:20，exp135 评测中，不算轮次）

**背景**：同数据同种子（`data/v05/exp132/train.jsonl`、seed 23）的视觉学习率剂量曲线：
- exp132（视觉顶 4 层 5e-6）：新用途最强，test_vqa_yesno 对 v0.4 **−1.60** 显著；
- exp134（冻结）：33 集对 v0.4 **全部无显著退步**，但对 exp132 回吐 ChartQA −9.48、shop −6.59、gui_state −2.86、ego −2.64；
- exp135（1e-6，中期 14 / 33 集）：对 exp132 仍回吐 shop −6.46、ChartQA −2.30、gui_state −2.21 → 判据 2 不过。

实测视觉漂移（`||f−f0||²/||f0||²`，对 v0.4 的 SigLIP 输出，每类 24 张图，CPU）：exp132 在 POPE / VQA 照片上 0.084 / 0.082，ChartQA / 购物截图 0.376 / 0.309；exp135 是 0.008 / 0.008 和 0.059 / 0.054。新用途要视觉大幅移动（~0.3），而照片上的 ~0.08 漂移伴随旧能力退步。统一调学习率无法同时满足；需要按图片域区分的约束。exp132 的输出层蒸馏（gold 混 v0.4 答案）对 VQA 是非无效，说明约束要加在特征上。

**假设**：在 13 个旧自然照片来源（同 exp132 蒸馏的来源）的样本上加特征锚定损失 `10 · ||f(x)−f0(x)||²/||f0(x)||²`（f0 = 载入时的 v0.4 视觉塔，冻结副本），视觉仍按 5e-6 在截图 / 图表 / 第一人称上适配。照片漂移降到 exp135 量级，新用途保持 exp132 水平。

**改动**：`TrainConfig.vision_anchor_weight / vision_anchor_sources`（`src/devision/train/rlcd.py`，测试 `tests/train/test_vision_unfreeze.py`）；训练日志多一项 `train/vision_drift`（锚定样本上的相对漂移）。`configs/exp136-v4-vision-anchor.yaml` = exp132 + 锚定（权重 10，来源 vqav2, coco-exist, gqa, n2-spatial, coco-size, coco-count-val, v7w, tallyqa, coco-position, coco-relation, aokvqa, vsr, vg-relation），评测 33 集同 exp134 / exp135。权重 10 的依据：exp132 的照片漂移 0.08 时惩罚约 0.8 × 锚定样本占比 ≈ 0.25，与 CE（~0.6）同量级。

**对照**：v0.4（正式门槛）；exp132（同数据同种子，只差锚定，归因）；exp134 / exp135（剂量曲线）。

**判据**（`devision-compare` 按图 95%）：
1. bench_pope、test_vqa_yesno 对 v0.4 无显著退步。
2. test_gui_state、test_gui_shop_state、test_ego、test_gui、test_chartqa 对 v0.4 显著提升，且对 exp132 点估计都 ≥ −2.0。
3. 其余集合对 v0.4 无显著退步。
- 1–3 过 → r23 = 本配方换种子（正式轮次，评完 lv_report + 独立校准）。
- 1 过、2 不过 → 锚定也压住了新用途需要的特征（共享层）；r23 候选改用 exp134 冻结配方并加量新用途数据。
- 1 不过 → 特征锚定不足以挡住照片上的退步；检查训练日志里 vision_drift 是否真的被压低（未压低 → 加大权重再试一次；已压低 → 退步不来自视觉特征漂移，记录）。

**预算**：训练约 65–70 分钟（锚定样本多一次不求导的视觉塔前向）+ 33 集评测约 2–2.5 小时。`scripts/v05/chain_exp136.sh 762397`（日志 `runs/v05-chain/chain_exp136.log`）10:15 挂在 exp135 链之后，exp135 约 12:00 评完，exp136 预计 15:00 前后评完。

## 2026-10-29 11:05：停止（未完成，无结论）

10:40 与第 23 轮 r23-v05 同时在 GPU 上开训（两条接续脚本都挂在 exp135 之后），两者都被拖慢到 0.7–1.0 秒/步。负责人 2026-10-29 已决定放行 POPE 小幅退步，本实验要解决的视觉漂移不再阻塞 v0.5，且正式轮次优先，故在第 1,200 步手动停止（chain、pipeline、训练进程），不再接续；目录保留，无评测结果。

# exp132：离线蒸馏向 v0.4（旧自然照片来源的 gold 混入 v0.4 答案）

2026-10-29 00:30 预写，开训前不改判据。

**背景**：从 v0.4 微调出来的每一次运行 POPE 对 v0.4 都在 −0.3 到 −1.0（r22 −0.34、exp125 −0.78、exp126 −0.98、exp127 −0.80、exp128 −0.76、exp129 −0.66 [−1.21, −0.09]）。exp129 加了 vqav2 是非 5,307 + coco-exist 3,000 回放仍显著退步，说明「多放旧题」不够。exp130（权重插值）是后处理路线；本实验是训练路线：题目不变，只把旧来源题的目标从 gold 改成 `0.5·gold + 0.5·v0.4 的答案`，让学生在 v0.4 原本就会的地方被拉回 v0.4 的函数（LwF 式蒸馏，但离线做，不在训练时加教师前向，显存与 exp129 相同）。exp18 的在线蒸馏是另一个问题（计数），判「无效」的形式与本实验不同。

**改动**：`scripts/data/v05/exp132_distill.py`
- `subset`：从 `data/v05/exp129r/train.jsonl` 取 13 个旧自然照片来源（vqav2、coco-exist、gqa、n2-spatial、coco-size、coco-count-val、v7w、tallyqa、coco-position、coco-relation、aokvqa、vsr、vg-relation）共 60,014 题到 `teacher_input.jsonl`；
- v0.4（`runs/r21-soft/train`，`decide` 实际输出，即带它自己的温度）在这些题上跑一遍 `devision-eval`（约 11 ms/题，约 12 分钟）；
- `mix`：这些题的目标改为 `0.5·gold + 0.5·v0.4`（按 id + qid + 选项集合对齐；n2-spatial 有同 id 不同选项的行），其余 145,152 题（新用途、图表、科学等）不变，写 `data/v05/exp132/train.jsonl` + MANIFEST。
- `configs/exp132-v4-offline-distill.yaml` 与 exp129 只差 `name` 和 `data`；**seed 23 同 exp129**，题目和顺序不变，所以对 exp129 的差就是目标的差。

**对照**：v0.4（正式门槛）；exp129（同数据同种子，归因用）。

**判据**（`devision-compare` 按图 95%）：
1. 旧能力：bench_pope 与 test_vqa_yesno 对 v0.4 无显著退步（95% 区间含 0 或为正）。
2. 新用途：test_gui_state、test_gui_shop_state、test_ego 对 v0.4 显著提升，且 test_ego、test_gui_state 对 exp129 的点估计不低于 −1.0。
3. 其余全部集合对 v0.4 无显著退步。
- 1–3 过 → r23 用 exp132 配方（种子另定），并与 exp130 的插值结果比较取简单者。
- 1 不过但 POPE 对 exp129 点估计 ≥ +0.3 → 方向对、力度不够，下一步 w = 0.8 或扩到全部旧来源。
- 1 不过且对 exp129 无改善 → 离线蒸馏对 POPE 漂移无效，不再重复。
- 2 不过 → 蒸馏拖累新用途，记录后放弃。

**预算**：教师一遍约 12 分钟 + 训练约 65 分钟 + 32 集评测约 120 分钟。`scripts/v05/chain_exp132.sh 703880`（00:35 挂在 exp131 链之后，日志 `runs/v05-chain/chain_exp132.log`），预计 2026-10-29 07:00–08:00 评完并自动跑 lv_report。`mix` 用均匀假教师试过：60,014 题改写，205,166 行全部通过 `read_decision_jsonl`。

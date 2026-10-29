# 训练日志

每一轮训练一节（轮次 = YAML 的 `name`，目录 `runs/<name>/`），写清改了什么、结果、和上一轮比涨了多少。过程细节和失败的尝试在 `experiments/2026-09-30-rlcd-plateau.md`。

**当前采用：`r21-soft`**（`runs/r21-soft/train`，模型 v0.4，2026-10-22 发布到 Hugging Face / ModelScope）。负责人 2026-10-22 决定发布（"发布这一版"）。相对 v0.3：23 个判定集合 22 个提升、无显著退步（镜像关系 +21.2、What'sUp +18.0、CountBench +14.5、V7W +7.2、VSR +7.0、ScienceQA +3.1、POPE +1.7）；事先写下的校准规则 4 未通过（size / MME / 存在 NLL 超门槛，主要是欠自信）。细节见 r21-soft 一节及 `release/v0.4/adoption.md`。


**v0.5开发方向**：图表方向保留exp81-chart作为后续起点；exp82-ui未通过，不取代它。exp83/84/85未通过；exp86原生SmolVLM未过筛；exp87 D3原生决策UI筛选通过，exp88图表/照片增益通过但位置/大小退化，exp89未见明显顺序偏置；exp94/96恢复pilot未通过，exp97组合诊断不扩量；exp98真实表单绑定开发筛选通过，exp99科学图示代理明显落后v0.4，exp100计数代理未达扩大门槛，尚缺表面检查域与正式全用途证据。exp110–112 D3 末层科学适配三次未过门槛，exp113 排除分辨率，D3 末层适配在科学图示上停止（2026-10-27），候选回到 v0.4 架构。均未正式发布。








## 2026-10-29 11:05：r23-v05 已开训（lr_vision 5e-6）；与之撞车的 exp136 已停

- 10:40 exp135 评完，按预写规则选 lr_vision：exp135 对 exp132 在 test_gui_shop_state **−6.46**（最差一项 < −2）→ r23 用 **5e-6**（exp132 原值）。
- 10:40 r23 开训，但 10:00 那一轮挂的 exp136（视觉锚定）也同时开训，两者都拖慢到 0.7–1.0 秒/步；POPE 已放行、正式轮次优先，11:05 停掉 exp136（第 1,200 步，无结论，见其文档末节）。r23 单独占卡后预计约 13:00 训完（25,646 步）、15:30 前后评完。

## 2026-10-29 10:20：负责人决定放行 POPE 小幅退步、不换架构 → 第 23 轮 r23-v05 正式训练已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。

- **决定（负责人 2026-10-29）**：POPE 对 v0.4 约 −0.5～−0.8 的小幅退步可以接受（模型卡写明）；v0.5 不换架构。
- **r23-v05**：exp132 配方 + 新种子 2323；lr_vision 在 exp135（1e-6）评完后按预写规则自动选（新用途最差一项不低于 exp132 2 点即用 1e-6）。`scripts/v05/chain_r23.sh` 挂在 exp135 链之后，预计 12:30 开训、18:00 前后评完；判据与发布准备见 [r23 计划](experiments/2026-10-29-r23-v05-plan.md)。

## 2026-10-29 10:20：exp134 终裁（33 集对 v0.4 零显著退步，首个）；exp135 中期判据 2 不过；exp136（视觉特征锚定）实现并接续

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。

- **exp134（冻结视觉）33 集评完**：对 v0.4 **没有一个集合显著退步**（从 v0.4 续训以来第一次），显著提升 shop +42.4、gui +36.7、ego +34.4、gui_state +24.6、ChartQA +16.3、What'sUp +3.9、CV-Bench +1.6；但对 exp132 新用途回吐（判据 2 不过）。保留为零退步的保底配方。计数、ScienceQA 无提升。
- **exp135（lr_vision 1e-6）14 / 33 集**：对 v0.4 ChartQA **+23.5**、shop +42.5、gui +37.7、ego +35.4、gui_state +25.2；对 exp132 shop **−6.46**、ChartQA **−2.30**、gui_state **−2.21** 显著 → 判据 2 不过；POPE / VQA 是非待评（约 12:00）。
- **视觉漂移实测**（相对 v0.4 SigLIP 输出）：exp132 照片 0.08、图表 / 购物截图 0.31–0.38；exp135 照片 0.008、截图 0.05–0.06。学习率统一调不出「照片不动、截图大动」。
- **exp136（新）**：新增 `vision_anchor_weight / vision_anchor_sources`（附测试），在 13 个旧自然照片来源上把 SigLIP 特征拉向 v0.4（权重 10），其余同 exp132。`scripts/v05/chain_exp136.sh` 挂在 exp135 之后，预计 15:00 前后评完。见 [exp136](experiments/2026-10-29-exp136-vision-anchor.md)。

## 2026-10-29 08:10：exp134（冻结视觉）中期：POPE / VQA 是非修好，但新用途回吐（ChartQA −9.5）→ 判据 2 不过；exp135（lr_vision 1e-6）接续

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。

- **exp134 对 v0.4**（24 / 33 集，按图 95%）：POPE −0.19 [−0.62, +0.26]、test_vqa_yesno −0.70 [−1.80, +0.50] 都含 0（exp132 −0.47 / **−1.60**）→ 判据 1 过；What'sUp **+3.9**、CV-Bench **+1.6**；新用途 shop +42.4、gui +36.7、ego +34.4、gui_state +24.6、ChartQA +16.3。
- **对 exp132（同数据同种子，只差视觉冻结）**：ChartQA **−9.48**、shop **−6.59**、gui_state **−2.86**、ego **−2.64** 显著 → 判据 2 不过。旧能力退步来自视觉漂移，新用途又要视觉适配。
- **exp135（新）**：exp132 只把 lr_vision 5e-6 → 1e-6，构成 0 / 1e-6 / 5e-6 剂量曲线；`scripts/v05/chain_exp135.sh` 挂在 exp134 之后，预计 12:00 前后评完。见 [exp134](experiments/2026-10-29-exp134-frozen-vision.md)、[exp135](experiments/2026-10-29-exp135-lowlr-vision.md)。

## 2026-10-29 06:10：exp133 切块判据 1 不过（只 ChartQA +2.6 显著，Mind2Web / FUNSD 不动）→ r23 不加切块；exp134 接续

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。

- **exp133 tiles 2 对 tiles 0**（同数据同 seed，按图 95%）：test_chartqa **+2.62 [+0.67, +4.61]** 显著；Mind2Web 子集 gui_state +0.24、gui −0.24，test_funsd −2.70，都含 0 → 读小字四项只一项显著，**判据 1 不过**。守护侧已见 MMBench −0.86 显著退步。切块把网页字高从 ~4.8 px 提到 ~9.6 px，Mind2Web 仍停在 36–37%，瓶颈不在像素。
- 两臂对 v0.4：chartqa 30.7 → 53–56、test_gui 38.6 → 75、test_ego 34.7 → 62（同为 exp129 系数据的效果，不是切块）。
- tiles 0 臂守护集还剩 7 集（约 06:40 评完），链随后补比较与 CPU 延迟，然后自动启动 exp134（冻结视觉），预计 10:30–11:00 评完。见 [exp133](experiments/2026-10-29-exp133-tiles.md)。

## 2026-10-29 04:15：exp132 裁决（不过：test_vqa_yesno 对 v0.4 −1.60 显著）；exp134（冻结视觉）预写并挂在 exp133 之后

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。

- **exp132（离线蒸馏向 v0.4，同 exp129 数据与 seed）31 / 32 集（剩 test_count_coco）**，对 v0.4（按图 95%）：POPE −0.47 [−1.00, +0.10]（不再显著；exp129 −0.66 显著），**test_vqa_yesno −1.60 [−3.17, −0.20] 显著** → 判据 1 不过；新用途 shop +49.0、gui +38.4、ego +37.0、gui_state +27.4，What'sUp +4.3、EgoThink +4.1；其余 18 集都含 0。对 exp129：test_v7w **+2.0** 显著修好，POPE +0.19、VQA 是非 −0.60（都不显著）→ 按预写「对 POPE 漂移无效，不再重复」。
- **exp134（新）**：exp132 原样，只把视觉编码器冻结在 v0.4 权重（此前 exp114–exp132 续训全部解冻视觉顶 4 层，从没试过冻结）。判据：POPE、VQA 是非对 v0.4 不显著退步，新用途对 exp132 点估计 ≥ −2.0。`scripts/v05/chain_exp134.sh` 挂在 exp133 链之后，预计 15:00 前后评完。见 [exp134](experiments/2026-10-29-exp134-frozen-vision.md)。
- GPU 排程：exp132 剩 test_count_coco（约 04:30 完）→ exp133 切块两臂（约 11:00–12:00）→ exp134。

## 2026-10-29 02:15：exp129 / exp130 裁决（都不过：POPE 对 v0.4 仍显著退步）；ChartQA 官方 test 图表 +27.7 确认；exp133 切块对照预写并挂链

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。exp129–exp133 为实验。

- **exp129 32 集评完**（对 v0.4，按图 95%）：新用途 shopping **+48.7**、test_gui **+38.7**、test_ego **+37.3**、test_gui_state **+26.3**、What'sUp **+4.6**、EgoThink **+3.6**、CV-Bench **+1.8**；**POPE −0.66、test_v7w −1.70 显著** → 判据 2 不过；test_vqa_yesno −1.0、test_size +0.8 等其余含 0。
- **exp131（ChartQA 官方 test，2,215 题未见图）**：v0.4 30.7 → **exp129 58.4，+27.67 [+25.2, +30.2]**、exp130 53.7，+23.0 → 图表能力在未见测试上显著提升，作为 v0.5 图表证据；后续候选都加评 test_chartqa。
- **exp130（v0.4 → exp129 插值，开发集选 α = 0.75）**：对 exp129 修好 test_vqa_yesno **+1.10**、test_v7w **+1.20**、A-OKVQA **+0.97**，但 POPE 只 +0.14，对 v0.4 仍 **−0.51 [−1.02, −0.03]**；新用途回吐 3–4pp（ego −3.84、gui −3.25、shop −3.36）；ECE 中位数 0.077 > exp129 0.062 → 判据 1、3 不过，r23 不加插值。
- **exp132**（离线蒸馏向 v0.4，回答 POPE 漂移）02:00 开训，02:07 在 1.4k / 25.6k 步，显存 10.3 GB；预计 03:20 训完、05:30 前后评完。
- **exp133（新，切块）**：字高检查通过——256 下 Mind2Web / FUNSD / ChartQA 文字几乎全 < 6 px，2×2 切块后 < 6 px 降到 5% / 25% / 7%（手机截图 72% 短边 < 512 不切块，本实验不回答手机 UI）。从 v0.4 起、同一 53,762 题大图小字混合、seed 133、micro batch 4，tiles 0 / 2 两臂只差切块，评 18 集（ChartQA test、Mind2Web 子集、FUNSD、守护集）+ CPU 延迟；`scripts/v05/chain_exp133.sh` 挂在 exp132 之后，预计 11:00–12:00 完成。见 [exp133](experiments/2026-10-29-exp133-tiles.md)。

**下一步**：exp132 评完裁决 POPE（过则以其配方写 r23）；exp133 判定切块是否进入 r23。

## 2026-10-29 00:40：exp129 评测 18 / 32：判据 2 不过（POPE 对 v0.4 −0.66 显著，回放没修好）；exp132（离线蒸馏向 v0.4）预写并挂链

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。exp129–exp132 为实验。

- **exp129 对 v0.4**（按图 95%）：新用途 shopping **+48.7**、test_gui **+38.7**、test_ego **+37.3**、test_gui_state **+26.3**；What'sUp **+4.6**、EgoThink **+3.6**、CV-Bench **+1.8** 显著；MME +0.4、MMBench 0.0、NaturalBench +0.1、RealWorld +0.5、CountBench +0.6、ScienceQA +1.0、lv VQAv2 是非 −0.2、A-OKVQA −0.4 含 0；**POPE −0.66 [−1.21, −0.09] 显著** → 判据 2 不过，不定 r23。对 r22 POPE −0.31（含 0）。yes ratio 不变（0.456 / 0.457），不是是非偏向。
- POPE 漂移六次运行都在（r22 −0.34 … exp129 −0.66），与回放量、种子无关；预写的「回到 exp128 配方」分支失效（exp128 也 −0.76）。另：未重拟合温度时 POPE NLL 0.557 / ECE 0.121（v0.4 0.363 / 0.048），候选必须重拟合温度再比校准。
- **exp132（新，预写）**：题目与 seed 同 exp129，只把 13 个旧自然照片来源 60,014 题的目标改成 `0.5·gold + 0.5·v0.4 答案`（离线蒸馏，不加显存），判据：POPE / test_vqa_yesno 对 v0.4 不显著退步且新用途保持。`scripts/v05/chain_exp132.sh` 挂在 exp131 之后（exp129 评完 → exp130 插值 → exp131 ChartQA test → exp132），预计 07:00–08:00 评完。见 [exp132](experiments/2026-10-29-exp132-v4-offline-distill.md)。

**下一步**：exp129 余下 14 集评完补表；exp130 选中 α 后看 POPE 是否回到 v0.4；exp130 / exp132 任一过判据即写 r23。

## 2026-10-28 23:25：exp129 评测 11 / 32：新用途判据 1、3 已过（test_ego 对 r22 +3.97 显著）；新建图表未见测试集 test_chartqa（exp131，挂在 exp130 之后）

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。exp129–exp131 为实验。

- **exp129 已评集合**（按图 95%，对 v0.4 / 对 r22）：test_ego 72.0，**+37.30** / **+3.97**；test_gui 77.3，**+38.70** / +1.04；test_gui_state 51.6，**+26.30** / +0.65（aitw-general 组对 r22 +2.08，含 0，点估计 ≥ 0）；shopping 70.2，**+48.71** / +1.55；EgoThink **+3.64** / +1.66；MMBench 0.00、RealWorld +0.51、ScienceQA +1.00（对 v0.4 含 0）。判据 1（新用途显著提升）、3（种子稳定）已过；判据 2 关键的 POPE / test_vqa_yesno / test_size 尚未评。评测比预计慢（同机其他 CPU 负载），预计次日 02:00–03:00 评完，之后 exp130 → exp131 依次自动接续。
- **exp131（新）**：32 集里没有图表测试，图表只有 chart_dev（val，监测用）。从 ChartQA 官方 test 划分（此前未下载）按 exp81 协议生成 `test_chartqa` 2,215 题（像素与 train / val 重叠的 21 图剔除），评 v0.4 / exp129 / exp130 选中的 α 并配对。见 [exp131](experiments/2026-10-28-exp131-chartqa-test.md)。

**下一步**：exp129 评完裁决判据 2；文档 / 小字仍缺未见测试集，下一步用同样方式补（DocVQA / InfographicVQA val 转选择题）。

## 2026-10-28 22:05：exp129 训完 25,646 步、32 集评测中；exp130（v0.4 → exp129 权重插值）预写并挂链

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。exp129、exp130 为实验。

- **exp129**（exp127 + 尺寸 / 是非 / 存在回放，seed 23）约 22:00 训完，显存 11.5 / 12.2 GB 正常；末段 loss 0.57–0.60。训练监测集（非判据）：val 59.4、gui 79.3、gui_state 55.1、gui_shop 71.5、ego 74.0、protect 80.2、chart 57.5、ui 69.1。32 集评测 22:03 开始，预计 23:45 前后评完并自动跑 lv_report。尚无判据结论。
- **exp130 预写**：POPE 对 v0.4 在五次运行中一直 −0.3 到 −1.0，数据回放没能稳定修好；改试权重插值 θ_v0.4 + α(θ_exp129 − θ_v0.4)，α ∈ {0.9, 0.75, 0.6}，温度重拟合，**只在开发集上**选 α（新用途保留 ≥ 85% 收益且 coco-exist + VQAv2 是非准确率合计比 exp129 高 ≥ 0.5pp，取最小 α），选中才全量评测 32 集 + lv_report。`scripts/v05/chain_exp130.sh` 挂在 exp129 链之后，预计次日 02:00 完成。见 [exp130](experiments/2026-10-28-exp130-v4-interp.md)。

**下一步**：exp129 评完按判据 1–3 与 v0.4 配对裁决；exp130 结果决定 r23 是否加插值后处理。

## 2026-10-28 21:05：exp128（r22 + 尺寸回放）评完 32 集：test_size 修复（对 r22 +3.5 显著），但 POPE 对 v0.4 −0.76 显著 → 判据 2 不过；exp129 训练 5.2k / 25.6k 步

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。exp128、exp129 为实验。

- **exp128 对 v0.4 / 对 r22**（按图 95%，`runs/exp128-v4-size-replay/eval/compare/`）：**test_size 87.9，+0.92（含 0）/ +3.51 [+2.16, +4.90]** → 判据 1 过，尺寸退步确是欠采样。新用途保持（test_gui_state +25.9、shopping +46.3、test_gui +37.5、test_ego +34.9 对 v0.4；test_ego 对 r22 **+1.59**）；CV-Bench **+1.59**、What'sUp **+2.56** 对 v0.4 显著；**POPE −0.76 [−1.42, −0.11] 对 v0.4 显著**（对 r22 −0.41 含 0）；test_vqa_yesno −1.10、MME 0.00、MMBench +0.12、ScienceQA +0.86 等其余 27 集含 0。lv_report：对 laya-vision VQAv2 是非 +3.4、A-OKVQA 未见过 +3.5 显著，ScienceQA +0.5，POPE 三档 89.0 / 86.2 / 82.1 均高于其公开分。
- **裁决**：判据 1 过、2 不过 → 不定 r23；尺寸回放保留进后续配方。POPE 对 v0.4 在 −0.3 到 −1.0 之间随种子 / 混合波动（r22 −0.34、exp125 −0.78、exp126 −0.98、exp128 −0.76），正是 exp129 是非 / 存在回放要解决的问题。exp129 后备分支（判据 3 不过时回到 exp128 配方）也须带这批回放。见 [exp128](experiments/2026-10-28-exp128-v4-size-replay.md)。
- **exp129**（exp127 + 尺寸 / 是非 / 存在回放，seed 23）20:47 开训，21:03 在 5,200 / 25,646 步（loss 0.77–0.80），显存 11.5 / 12.2 GB；预计 22:15 训完、23:45 前后评完并自动跑 lv_report。

**下一步**：exp129 评完按判据 1–3 与 v0.4 配对裁决，重点看 POPE 和 test_vqa_yesno；过则写 r23。exp129 评完前预写下一个 GPU 任务并挂链。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 20:05：exp128（r22 + 尺寸回放）训完、评测 6 / 32：三个新用途对 v0.4 仍大幅提升，test_ego 对 r22 +1.6 显著；test_size 尚未评

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。exp128、exp129 为实验。

- **exp128**（r22 数据 + coco-size 4,202，seed 22）19:50 前后训完 21k 步，显存峰值 10.6 / 12.2 GB。已评 6 集（对 r22 / 对 v0.4，按图 95%）：test_gui_state 51.2，+0.26 / **+25.91**；test_gui_shop_state 67.7，−0.90 / **+46.25**；test_gui 76.0，−0.19 / **+37.47**；test_ego 69.6，**+1.59** / **+34.91**；EgoThink 0.00 / +1.99；ScienceQA −0.14 / +0.86（后两项含 0）。尺寸回放未挤掉 GUI / 第一人称。判据 1（test_size）未评，预计 21:45 前后 32 集评完并自动跑 lv_report。
- **exp129**（exp127 数据 + 尺寸 / 是非 / 存在回放，seed 23）由 `scripts/v05/chain_exp129.sh` 在 exp128 链退出后自动开训，预计次日 00:30–01:00 评完；GPU 到那时一直有任务。

**下一步**：exp128 评完按判据 1、2 裁决；exp129 评完与 v0.4、exp127 配对，过则写 r23。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 19:15：exp127 评完 32 集：第一人称 / 手机 GUI 扩量有效（ego +3.8、gui +2.7），但 VQAv2 是非 −3.0、POPE 对 v0.4 −0.8 显著 → 判据 3 不过；exp129 开训前加是非题回放

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。exp127 为实验（不算轮次）。

- **exp127**（r22 数据 + EPIC-VISOR 19.6k + AITW-general 状态题 9k，seed 22）对 r22 / 对 v0.4（按图 95%）：test_ego 71.8，**+3.79** / +37.11；test_gui 79.0，**+2.73** / +40.39；test_gui_state 51.9，+0.91 / **+26.56**；shopping 70.5，+1.94 / **+49.10**；What'sUp +1.83 / **+4.76**；**test_vqa_yesno 69.4，−3.00 / −2.80（均显著）**；**POPE −0.46 / −0.80（对 v0.4 显著）**；test_size +0.08 / −2.50（r22 已知欠采样）；其余 22 集对两者含 0（MME +1.05 / +0.51、ScienceQA −0.05 / +0.95、RealWorld、MMBench、CV-Bench、EgoThink 等）。
- **裁决**：判据 1、2 过，判据 3 不过 → 不定 r23。不是是非偏向（新增 touching 题配平，预测 true 比例 0.51 → 0.48、true 召回 −6pp），是自然照片是非题占比被稀释。
- **exp129 开训前修订**：数据改为 exp127 + coco-size 4,202 + vqav2 是非 5,307 + coco-exist 3,000（`data/v05/exp129r/`，205,166 题，与评测集图片重叠 0），seed 23，判据不变。exp128（r22 + 尺寸回放）19:15 训练 6.8k / 21k 步，loss 0.70，显存 10.6 / 12.2 GB；约 21:30 评完后 exp129 自动启动，约次日 00:30 评完。见 [exp127](experiments/2026-10-28-exp127-v4-gui-ego-scale.md)、[exp129](experiments/2026-10-28-exp129-v4-scale-size-seed23.md)。

**下一步**：exp128 评完裁决尺寸回放；exp129 过则写 r23。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 17:10：r22-guiego 评完 32 集：新用途大幅提升、30 个旧集合不退，但 test_size −2.6 显著 → 按预写判据不采用；定位为尺寸题欠采样（129 / 4,331），exp128 尺寸回放挂在 exp127 之后

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。r22 为正式轮次（判据 2 不过，不采用），exp127（训练中）、exp128（预写）为实验。

- **r22-guiego 对 v0.4**（按图 95%）：test_gui_state **+25.65**、test_gui_shop_state **+47.16**、test_gui **+37.66**、test_ego **+33.32**、What'sUp **+2.93**、CV-Bench **+1.48**（均显著）；**test_size −2.59 [−4.10, −1.18] 显著退步**（84.4 vs 87.0，NLL 0.507 → 0.359、ECE 0.234 → 0.033）；其余 25 集含 0（POPE −0.34、MME −0.55、MMBench −0.58、ScienceQA +1.00、CountBench +0.81、RealWorld −0.51、EgoThink +1.99）。对 exp124 同数据异种子 18 集全部含 0。lv_report：对 laya-vision VQAv2 是非 +3.7、A-OKVQA +3.9 显著，ScienceQA +0.7（上一版 −0.3），POPE 三档均高于其公开分。
- **裁决**：判据 1、3 过，判据 2 不过 → 不采用（相对上一轮 v0.4 的增长见上；不是正式五用途 / ImageJevBench 成绩）。原因：exp116 起 v05 混合的 coco-size 只剩 129 题（v0.4 4,331；其他 v0.4 来源保留约 35%）。
- **exp128 预写**：r22 数据 + soft_mix 其余 4,202 道 coco-size（168,242 题，与 test_size 图片重叠 0），seed 22 与 r22 配对，32 集全评 + lv_report；`scripts/v05/chain_exp128.sh` 挂在 exp127 链之后，预计 22:45 评完。exp127 17:00 在 8,500 / 24,083 步（loss 0.74–0.75，显存 8.9 / 12.2 GB），预计 20:10 评完。exp124 / exp126 的 test_size CPU 补评因拖慢训练而停止，由 exp128 同种子配对回答。见 [r22](experiments/2026-10-28-r22-guiego.md)、[exp128](experiments/2026-10-28-exp128-v4-size-replay.md)。

**下一步**：exp127 评完按其判据裁决（其判据 3 的旧集合包含 test_size，同样会受欠采样影响，裁决时要和 exp128 分开看）；exp128 过则 r23 = 选定数据 + 尺寸回放。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 16:05：r22-guiego 训完、评测 9 / 32：三个新用途对 v0.4 显著提升（GUI 状态 +25.7、shopping +47.2、第一人称 +33.3），已评旧集合不退；exp127（第一人称 / 手机 GUI 扩量）挂在 r22 之后

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。r22 为正式轮次（评测中），exp127 为实验（未开训）。

- **r22-guiego**（exp124 数据 + seed 22）15:39 训完 20,505 步。已评 9 集（对 v0.4 / 对 exp124，按图 95%）：test_gui_state 51.0，**+25.65** / +1.04；test_gui_shop_state 68.6，**+47.16** / −1.03；test_gui 76.2，**+37.66** / +0.91；test_ego 68.0，**+33.32** / −0.80；EgoThink +1.99 / −0.33；ScienceQA +1.00 / +0.33；dev_sqa −0.30 / −0.50；RealWorld −0.51 / −1.36（后六项均含 0）。判据 1 过；对 exp124 全部含 0（配方稳定）；判据 2 待其余 23 集，预计 17:30 前后评完并自动跑 lv_report。见 [r22](experiments/2026-10-28-r22-guiego.md)。
- **exp127 预写**：r22 数据 + EPIC-VISOR 剩余 19.6k + AITW-general 屏幕状态 9k（192,657 题），seed 22 与 r22 同种子配对；`scripts/v05/chain_exp127.sh` 在 r22 链退出后自动开训，预计 20:10 前后评完。见 [exp127](experiments/2026-10-28-exp127-v4-gui-ego-scale.md)。

**下一步**：r22 评完按判据 2 裁决，与 v0.4 逐题配对、读 lv_report；过则 r22 进入候选冻结后的独立校准 / 测试。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据；Mind2Web 网页在 256 输入下无看图增益。

## 2026-10-28 15:10：exp126（种子对照）评完：ScienceQA 退步是种子，回放的 GUI 代价不显著（−1.1，下界 −2.8）→ r22 继续；r22 训练 6.2k / 20.5k 步

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。exp126 为实验，r22 为正式轮次（训练中）。

- **exp126**（exp122 数据 + seed 124；对 exp122 / exp124 / v0.4）：test_gui_state 51.0，−1.36 / +1.10 / +25.7；test_gui_shop_state 69.1，−0.26 / −0.52 / +47.7；test_ego 68.1，−0.61 / −0.73 / +33.4；**ScienceQA −1.57 显著** / −0.33 / +0.33；**POPE** −0.27 / −0.57 / **−0.98 显著**；MMBench **+1.30 / +0.93** 显著；What'sUp、CV-Bench、EgoThink 对 v0.4 显著提升；其余含 0。
- **裁决**：判据 1 部分成立（ScienceQA 退步同种子复现 → 种子；GUI 同向但含 0）；判据 2 严格不过（exp124 − exp126 test_gui_state −1.10 [−2.79, +0.65]）；判据 3 不成立 → 按预写分支 **r22 不停**。回放修 POPE（−0.98 → −0.41），代价是 GUI 约 −1pp 点估计（Mind2Web 为主）。见 [exp126](experiments/2026-10-28-exp126-v4-seed-control.md)。
- **r22-guiego** 14:43 开训，15:01 在 6,200 / 20,505 步（loss 0.74–0.84），显存 9.1 GB；预计 15:45 训完、17:30 前后评完（32 集 + lv_report）。

**下一步**：r22 评完按其判据与 v0.4 逐题配对、lv_report；r22 评测期间预写候选冻结后的独立校准 / 测试流程。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 14:05：exp126（种子对照）训完 19,380 步、评测中；第 22 轮 r22-guiego 预写并挂在 exp126 之后

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。exp126 为实验，r22 为正式轮次（尚未开训）。

- **exp126**（exp122 数据 + seed 124）：14:00 训完 19,380 / 19,380 步，末段 loss 0.65–0.70（与 exp124 同期一致），val 准确率 62.1、ECE 0.049，显存 11.7 / 12.2 GB 正常；18 集评测中，预计 15:00 前后评完。尚无对照结论。
- **r22-guiego 预写**：exp124 训练文件（9k 存在回放）+ seed 22，评测 = exp124 的 18 集 + v0.4 的 14 个旧集合，评完自动跑 lv_report。`scripts/v05/chain_r22.sh` 在 exp126 退出后自动开训；若 exp126 判出 9k 回放有真实 GUI 代价（其判据 3），下一轮停掉 r22 改训 COCO-only 回放。见 [r22](experiments/2026-10-28-r22-guiego.md)。

**下一步**：exp126 评完按判据裁决并核对 r22 分支；r22 评完与 v0.4 逐题配对、lv_report。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 13:20：exp125（回放减半）评完：POPE 对 v0.4 −0.8 显著（半剂量不够），GUI 对 exp122 仍 −2 → 9k 剂量；exp126 种子对照 13:09 开训

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp125**（exp122 + 4.5k 回放，seed 125；对 exp122 / exp124 / v0.4）：**POPE 85.8，−0.07 / −0.37 / −0.78 [−1.41, −0.17]**；test_gui_state 50.3，**−2.08 [−4.16, −0.19]** / +0.39 / +25.0；test_gui_shop_state 67.3，−2.07 / −2.33 / +45.9；test_ego 68.8，+0.10 / −0.02 / +34.1；ScienceQA **−1.81 显著** / −0.57 / +0.10；MME +1.01 / **+1.43** / **+1.14**；What'sUp **+5.37**、EgoThink **+3.48** 对 v0.4 显著；其余含 0。
- **裁决**：判据 1、2 不过，3 过 → 走预写分支"POPE 需要 9k 剂量"。exp124、exp125 两个不同种子对 exp122 的 GUI / ScienceQA 退步几乎一样，而两者之间不显著 → 减半回放收不回 GUI，更像 exp122 种子偏高。见 [exp125](experiments/2026-10-28-exp125-v4-exist-replay-half.md)。
- **exp126**（exp122 数据 + seed 124）13:09 自动开训，预计 15:10 前后评完；与 exp124 同种子配对量回放的 GUI 代价。

**下一步**：exp126 评完后按其判据定 `r22-*` 配方（预期 9k 回放），并把 r22 预写挂在 exp126 之后。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 12:05：exp124 评完：存在题回放修好 POPE（对 v0.4 −0.4 含 0），旧集合全守住，但 GUI −2 显著；exp126（同种子对照）已挂在 exp125 之后

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp124 其余 13 集**（对上一实验 exp122 / 对 v0.4）：**POPE 86.1，+0.30 / −0.41 [−0.97, +0.16]**（误报 503→440，yes ratio 0.470→0.459）；lv VQAv2 是非 **+0.80 显著** / +0.06；What'sUp 49.8，−1.10 / **+3.05 显著**；CV-Bench +0.53 / **+1.63 显著**；ScienceQA **−1.24 [−2.24, −0.14]** / +0.67；MMBench、MME、CountBench、RealWorld、NaturalBench、A-OKVQA、dev_sqa、EgoThink 均含 0。
- **裁决**：判据 1（POPE）过、3（旧能力对 v0.4）过、2（GUI 对 exp122 下界 > −2pp）不过（test_gui_state −2.47、test_gui −1.95）。exp124 与 exp122 种子不同，GUI / ScienceQA 的退步可能是种子噪声。见 [exp124](experiments/2026-10-28-exp124-v4-exist-replay.md)。
- **exp125**（半剂量回放）12:05 在 10,900 / 19,943 步（loss 0.70–0.73），预计 13:15 前后评完。
- **exp126 预写并挂接**：exp122 数据 + seed 124（= exp124 去掉回放），与 exp124 同种子配对量回放的真实 GUI 代价、与 exp122 同数据配对量种子噪声；不新造数据。`scripts/v05/chain_exp126.sh` 在 exp125 退出后自动开跑，预计 15:10 前后评完。见 [exp126](experiments/2026-10-28-exp126-v4-seed-control.md)。

**下一步**：exp125 判半剂量；exp126 定回放剂量后写 `r22-*`。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 11:05：exp124（存在题回放 9k）评测中 5/18：test_gui_state 对 exp122 −2.5 显著，判据 2 不过；exp125（回放减半）已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp124**（对上一实验 exp122）：test_gui_shop_state 69.6，+0.26（含 0）；**test_gui_state 49.9，−2.47 [−4.29, −0.58]**（主要在网页 Mind2Web −3.06；手机 AITW −1.80 含 0）；test_gui 75.3，−1.95 [−3.64, −0.26]；test_ego 68.8，+0.12；EgoThink −0.17。两者种子不同，未排除种子噪声。POPE（判据 1）等 13 集评测中，预计 11:30 前后评完。
- **裁决（部分）**：判据 2 不过 → 按预写规则回放减半。见 [exp124](experiments/2026-10-28-exp124-v4-exist-replay.md)。
- **exp125 预写并挂接**：exp122 训练文件 + coco-exist 3k（真假各半）+ VQAv2 是非 1.5k（159,540 题，seed 125）；`scripts/v05/chain_exp125.sh` 在 exp124 退出后自动开跑，预计 13:15 前后评完。见 [exp125](experiments/2026-10-28-exp125-v4-exist-replay-half.md)。

**下一步**：exp124 补齐后判 POPE；exp125 与 exp122 / exp124 配对定回放剂量，过则定 `r22-*`。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 10:10：exp123（GUI 23% + 配平）评完：去掉 24k GUI 使未见网站 −8.8，MME / MMBench 不变 → r22 用 33% GUI；POPE 仍 −0.8，等 exp124

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp123**（对配对的 exp122 / 对 v0.4）：未见网站 `test_gui_shop_state` **60.6，−8.79 [−11.89, −5.56]** / +39.2；test_gui_state −6.17、test_gui −4.87（均显著）；test_ego 68.6，−0.07；**MME −0.13 [−1.10, +0.80]、MMBench +0.49 [−0.30, +1.28]**（对 v0.4 均 0.0）；What'sUp 51.5，+0.61 / **+4.76 显著**（对 exp120 +18.9）；CV-Bench +0.99 / **+2.08 显著**；**POPE −0.11 / −0.82 [−1.50, −0.12]**；其余（ScienceQA、dev_sqa、CountBench、EgoThink、RealWorld、NaturalBench、A-OKVQA、VQAv2 是非）对 v0.4 均含 0。
- **裁决**：判据 1 过；判据 2 → GUI 扩量不是 MME / MMBench 退步来源，**r22 采用 exp122 的 33% GUI**；判据 3 只剩 POPE（与 GUI 量无关，归因存在题稀释）。见 [exp123](experiments/2026-10-28-exp123-v4-spatial-gui23.md)。
- **exp124**（exp122 + 存在题回放 9k）09:51 开训，10:00 在 3,300 / 20,505 步（loss 0.87–0.93，与前几轮同期一致），预计 11:10 训完、12:05 前后评完。

**下一步**：exp124 判 POPE 修复；过则以其配方定 `r22-*`，11:00 那一轮须把 r22（或 exp124 不过时的下一个实验）预写并挂在 exp124 之后。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 09:05：exp122 补齐：What'sUp 左右恢复（+18.4），CV-Bench 守住，但 POPE 对 v0.4 −0.7 显著；exp124（存在题回放）已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp122 剩余 10 集**（对上一实验 exp121 / 对 v0.4）：**What'sUp 50.9，+18.41 [+14.02, +22.56] / +4.15 [+1.71, +6.59]**（left_of 2.9→49.8、right_of 3.4→53.7）；CV-Bench 56.2，−1.29（含 0）/ +1.10（含 0，3D-depth 55.2 vs v0.4 54.8）；MMBench +0.70 / −0.47、MME +0.17 / +0.13、NaturalBench、CountBench 均含 0；**POPE 85.8，−0.48（含 0）/ −0.71 [−1.36, −0.06]**；lv VQAv2 是非 −0.74 [−1.54, +0.04]、A-OKVQA −1.32（含 0）对 v0.4。
- **裁决**：判据 1（左右恢复）、2（深度保住）过；判据 3 有 test_ego 对 exp121 −1.1 一项小幅显著；对 v0.4 还有 POPE −0.7 显著 → 不直接定 r22。见 [exp122](experiments/2026-10-28-exp122-v4-spatial-balance.md)。
- **exp123** 09:00 训练 14,300 / 16,380 步（loss 0.69–0.72，正常），预计 10:05 前后评完。
- **exp124 预写并挂接**：exp122 训练文件 + v0.4 混合中的 coco-exist 6k（真假各半）+ VQAv2 是非 3k（164,040 题）；判据：POPE 对 v0.4 不显著退步、GUI / ego 对 exp122 不退、其余旧集合对 v0.4 不退。`scripts/v05/chain_exp124.sh` 在 exp123 退出后自动开跑，预计 12:00 前后评完。见 [exp124](experiments/2026-10-28-exp124-v4-exist-replay.md)。

**下一步**：exp123 与 exp122 配对定 GUI 占比；exp124 判存在题回放；两者合并后定 `r22-*`。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 08:05：exp122（关系词配平）训完，8/18 集已评：GUI 小涨、ScienceQA 对 v0.4 +1.9 显著、test_ego −1.1；exp123 已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp122**（对上一实验 exp121 / 对 v0.4）：未见网站 `test_gui_shop_state` **69.4，+2.20 [−0.26, +4.52]** / +47.9；`test_gui_state` 52.4，+2.34 [+0.45, +4.22]；test_gui 77.3，+1.43（含 0）；**test_ego 68.7，−1.11 [−2.01, −0.21]**；EgoThink +1.32 / +2.48（含 0）；ScienceQA 84.0，+0.67 / **对 v0.4 +1.91 [+0.67, +3.15]**；dev_sqa +0.99 / +0.40；RealWorld +0.51 / −1.19（含 0）。
- 判据 1（What'sUp 左右恢复）、2（CV-Bench）及 MMBench / MME / CountBench / POPE 等 10 集评测中，预计 08:45 补齐后裁决。见 [exp122](experiments/2026-10-28-exp122-v4-spatial-balance.md)。
- exp123（exp122 去掉 24k GUI）由 `scripts/v05/chain_exp123.sh` 在 exp122 退出后自动开跑，预计 10:30 前后评完；GPU 到那时都不空。

**下一步**：exp122 补齐并裁决；exp123 与 exp122 配对定 GUI 占比；10:00 那一轮前须预写 `r22-*` 并挂在 exp123 之后。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 07:10：exp121（手机 GUI 再扩 24k）：未见网站 GUI +8.7，但 MME / MMBench 小幅显著退，判据 3 不过；exp123（GUI 23% + 配平）已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp121**（对上一实验 exp120 / 对 v0.4）：未见网站 `test_gui_shop_state` **67.2，+8.66 [+5.56, +11.76]** / +45.7（配错图 29.3）；`test_gui_state` 的 AITW 部分 +6.37 [+3.60, +9.42]；Mind2Web 不变（−0.49 / 0.00，仍无看图增益）；test_ego 69.8，+1.69；ScienceQA +0.48 / 对 v0.4 +1.24 显著；CV-Bench 对 v0.4 +2.39 显著。
- **退步**：MME 对 exp120 −1.14 [−2.19, −0.08]；MMBench 对 v0.4 −1.16 [−2.07, −0.26]；What'sUp 对 v0.4 −14.3（继承 exp120）。判据 1、2 过，**3 不过**，按预写规则 GUI 回到 23%。见 [exp121](experiments/2026-10-28-exp121-v4-gui-scale.md)。
- **exp122** 训练中（07:05 在 7,600 / 19,380 步，loss 0.71–0.78），预计 08:40 前后评完。
- **exp123 预写并挂接**：exp120 训练文件 + 与 exp122 相同的 n2-spatial 10,151 题（= exp122 去掉 24k GUI），131,040 题；与 exp122 逐题配对判定 GUI 扩量是否是 MME / MMBench 退步的来源，选出 r22 配方。`scripts/v05/chain_exp123.sh` 在 exp122 退出后自动开跑，预计 10:30 前后评完。见 [exp123](experiments/2026-10-28-exp123-v4-spatial-gui23.md)。

**下一步**：exp122 判关系词配平；exp123 与 exp122 配对定 GUI 占比，选出的配方进 `r22-*`。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据；网页（Mind2Web）在 256 输入下不可读。

## 2026-10-28 06:00：exp120 补齐：What'sUp 左右关系崩溃（−12.9），判据 3 不过；exp122（关系词配平）已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp120 剩余 6 集**（对上一实验 exp119 / 对 v0.4）：**What'sUp 32.6，−12.93 [−17.07, −8.78] / −14.15 [−18.05, −10.12]**——left_of / right_of 从约 50% 掉到 1.5%（in-front_of 32→78、behind 28→59）；CountBench +3.05 / +2.04（含 0）；POPE −0.56 [−1.13, −0.01] / −0.29（含 0）；lv VQAv2 是非 +0.33、A-OKVQA −0.09（含 0）。
- **归因**：新增 4,000 道 n2-spatial 深度题的选项是 left / right / front / behind 四选一，gold 只会是 front / behind，混合里以 left / right 为 gold 的只有 213 题 → 模型学到"不选 left / right"的选项先验。CV-Bench 3D 的提升（选项是物体名）仍成立。**exp120 判据 3 不过**，05:05 节"判据 3 只差 MMBench 一项小幅"的结论作废；exp121 以 exp120 为底，预计继承此退步。详见 [exp120](experiments/2026-10-28-exp120-v4-depth-ego.md)。
- **exp121** 06:00 训完 18,112 步（末段 loss 0.63–0.70，acc ≈ 0.74），评测中，预计 07:00 前后结束。
- **exp122 预写并挂接**：exp121 训练文件 + n2-spatial 剩余全部 10,151 题（混合内各关系类别比例与源文件一致），155,040 题；判据：What'sUp 对 exp121 下界 > 0 且对 v0.4 不显著退步、左右子项各 ≥ 40%，CV-Bench 不退，其余对 exp121 不退。`scripts/v05/chain_exp122.sh` 在 exp121 退出后自动开跑，预计 08:45 前后评完。见 [exp122](experiments/2026-10-28-exp122-v4-spatial-balance.md)。

**下一步**：exp121 判 GUI 扩量（What'sUp 预计同样崩溃，与 GUI 判据分开看）；exp122 判关系词配平，过则 exp122 配方定 `r22-*`。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 05:05：exp120（深度回放 + ego 扩量）部分结果：CV-Bench 3D 修复、ego +3.6，MMBench 小幅退；exp121 已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp120**（exp119 + n2-spatial 深度 4k + EPIC ego 12k，120,889 题），18 集已评 12 集。相对上一实验 exp119：**CV-Bench +2.31 [+1.02, +3.56]**（3D-depth +3.17、3D-distance +4.83；对 v0.4 +2.65 [+1.29, +4.02]，exp117/118 以来的 3D 退步修复）；**test_ego 64.6→68.1，+3.56 [+2.56, +4.52]**（对 v0.4 +33.4），EgoThink +0.00；GUI 三集均含 0（shop 58.5 −1.29、state 47.1 +0.13、原题型 −1.36）；ScienceQA +0.91、dev_sqa +0.59、MME +1.01、NaturalBench +0.07、RealWorld −1.02 均不显著退步；**MMBench −0.95 [−1.77, −0.12]**（对 v0.4 −0.72 含 0）。→ 判据 1、2 过，判据 3 有 MMBench 一项小幅不过；剩余 CountBench / What'sUp / POPE / lv 三集评测中。详见 [exp120](experiments/2026-10-28-exp120-v4-depth-ego.md)。
- **exp121**（exp120 + 手机 GUI 改写 24k）由 `scripts/v05/chain_exp121.sh` 在 exp120 评测结束后自动开跑（dry-run 通过，数据 `data/v05/exp121/` 就绪），预计 07:00 前后评完。

**下一步**：补齐 exp120 剩余 6 集；exp121 判 GUI 再扩量后定 `r22-*` 正式配方（含深度回放与 ego 扩量，MMBench 对 v0.4 继续盯）；Mind2Web 走切块 / 高分辨率小实验。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 04:05：exp119（手机 GUI 扩量 + 科学回放）判据全过，成为 r22 候选；exp120 训练中，exp121（GUI 再扩 24k）已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp119**（104,889 题）：未见网站的手机购物操作 `test_gui_shop_state` **59.8**（配错图 31.1；v0.4 21.4、exp117 36.4、exp118 49.6），相对上一实验 exp118 **+10.21 [+6.72, +13.70]**；`test_gui_state` AITW 58.9 vs 配错图 32.3；Mind2Web 原题型对 exp117 +0.00（判据 2 过）。旧能力：对 exp117 无显著退步，CV-Bench +1.71 [+0.45, +2.92]（3D-depth +3.83），**对 v0.4 15 个旧集合全部区间含 0**（ScienceQA −0.14、dev_sqa −0.05、CV-Bench +0.34、CountBench −1.02、What'sUp −1.22）；新用途对 v0.4：test_ego +29.86、test_gui +35.39、shop +38.37。→ 判据 1–3 全过，**exp119 配方为 `r22-*` 候选**。详见 [exp119](experiments/2026-10-28-exp119-v4-gui-phone-science.md)。
- **exp120**（+深度回放 4k + ego 12k）04:00 训练 5,300/15,112 步，loss 0.73–0.86，正常，预计 05:15 前后评完。
- **exp121 预写并挂接**：exp120 训练文件 + AITW-general 改写 10k + AITW-shopping 改写 14k（未用 episode，GUI 占比 23%→33%，144,889 题）；判据：GUI 对 exp120 下界 > 0、Mind2Web 不退 > −3pp、旧能力对 exp120 与 v0.4 无显著退步。`scripts/v05/chain_exp121.sh` 在 exp120 退出后自动开跑，预计 06:50 评完。见 [exp121](experiments/2026-10-28-exp121-v4-gui-scale.md)。

**下一步**：exp120 判深度 / ego 扩量，exp121 判 GUI 再扩量，合并后定 `r22-*` 正式轮次（与 v0.4 全面逐题配对、lv_report、独立校准）。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据；网页（Mind2Web）在 256 输入下不可读，需切块 / 分辨率实验。

## 2026-10-28 03:05：exp118 补完判据 3（CV-Bench 仍退）；exp119 训完评测中；exp120（深度回放 + ego 扩量）已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp118 判据 3**（对上一实验 exp116 按图配对）：**CV-Bench −2.05 [−3.26, −0.87]** 显著退步（与 exp117 的 3D 退步同向）；NaturalBench +1.25 [+0.66, +1.86]；其余 8 集含 0。GUI-shopping 基线（`test_gui_shop_state`，774 题、未见网站）：v0.4 21.4、exp117 36.4、exp118 49.6。
- **exp119** 13,112 步 03:00 训完（末段 loss 0.64–0.77，acc ≈ 0.70，正常），评测进行中，预计 03:55 结束。
- **exp120 预写并挂接**：exp119 训练文件 + n2-spatial 深度题（in front of / behind）4,000 + EPIC-VISOR 12,000（共 120,889 题）；判据：CV-Bench 对 exp119 下界 > 0 或对 v0.4 不再显著退步、test_ego 对 exp119 下界 > 0、GUI / 科学 / 其余不退。`scripts/v05/chain_exp120.sh` 在 exp119 退出后自动开跑，预计 05:30 前后评完。见 [exp120](experiments/2026-10-28-exp120-v4-depth-ego.md)。

**下一步**：exp119 评完按预写判据裁决（手机 GUI 扩量、Mind2Web 不退、科学守住）；exp120 判深度回放能否修 CV-Bench 3D；两者都过则定 `r22-*`。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 02:10：exp118（要看屏幕的 GUI 题）初步结果：手机屏幕看图成立，网页在 256 下不可读；exp119 已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp118**（exp116 + GUI 改写题：去历史操作、干扰项换同 episode 其他步的 gold）GUI / ego / 科学已评完，其余 10 集评测中（预计 02:50 结束）。`test_gui_state`：45.5（配错图 35.1，+10.4pp）；**相对上一实验 exp116 +6.49 [+3.83, +9.09]**；分来源：**AITW 手机 56.6 vs 配错图 34.8（+21.8pp），对 exp116 +14.27 [+9.97, +18.56]**；**Mind2Web 网页 35.6 vs 35.3，对 exp116 −0.37 [−3.79, +3.06]** → 按预写判据 1 不过（Mind2Web < 5pp）。原题型 test_gui 80.6→66.3（判据 2 不过，Mind2Web 丢了历史线索）。ego 对 exp117 +0.73 [−0.10, +1.58]。科学对 exp117 −2.3 / −2.5（exp118 底座是 exp116，无科学回放，与 exp116 一致，预期内）。
- **裁决**：手机 GUI 题型有效；Mind2Web 是 1024×800 桌面页，256 letterbox 下文字约 3px，走"分辨率 / 切块"分支，不加步数。详见 [exp118](experiments/2026-10-28-exp118-gui-screen-state.md)。
- **exp119 预写并挂接**：exp117 混合（科学回放）+ AITW-general 改写 6k + 新增 AITW-shopping 改写 12k（未见网站分组 test 774 题）+ Mind2Web 保持原题型，104,889 题；`scripts/v05/chain_exp119.sh` 在 exp118 退出后先评 v0.4 / exp117 / exp118 的 shopping 基线，再训练评测，预计 04:30 前后结束。见 [exp119](experiments/2026-10-28-exp119-v4-gui-phone-science.md)。

**下一步**：exp118 剩余集合评完补判据 3；exp119 判手机 GUI 扩量 + 科学是否同时守住，过则作为 `r22-*` 候选；Mind2Web 另做切块 / 高分辨率小实验。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 01:15：exp117（exp116 + 科学回放）结果：科学退步修复，CV-Bench 3D 小幅退步；exp118 已自动开跑

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp117**（exp116 + exp114 科学题 15,841，92,889 题 / 11,612 步，对 v0.4 按图配对）：ScienceQA 82.1→82.5（+0.48 [−0.62, +1.67]）、dev_sqa 84.6→85.0（+0.40）；**相对上一实验 exp116：ScienceQA +2.19 [+0.95, +3.48]、dev_sqa +3.82 [+2.37, +5.31]** → 科学退步修复。GUI 80.1（对 exp116 −0.52，含 0；配错图 79.2，仍是文字先验）、ego 64.9（−0.51，含 0）、EgoThink 77.0（对 exp116 +2.32 [+0.17, +4.47]）、CountBench +1.43、MME 对 exp116 +1.26 [+0.17, +2.32]；图表 dev 56.3、UI dev 68.8（均过门槛，比 exp116 低 2–3 点）。**不过的一项**：CV-Bench −1.36 [−2.62, −0.08]（3D-depth 对 v0.4 −3.83、3D-distance 对 exp116 −4.00；二选一、v0.4 自身近随机），其余 13 集无显著退步。lv_report 已生成。裁决：作为合并配方底座，不是发布候选。详见 [exp117](experiments/2026-10-27-exp117-v4-gui-ego-science.md)。
- **exp118**（要看屏幕的 GUI 题型）01:07 由 `scripts/v05/chain_exp118.sh` 自动启动：先评 v0.4 / exp116 的 `test_gui_state` 基线，再训练评测，预计 03:00 前后结束。

**下一步**：exp118 判 GUI 是否看屏幕；过则合并 exp117 科学回放 + exp118 GUI 题型 + aitw-shopping / ego 扩量，复查 CV-Bench 3D，定 `r22-*`；3D 退步若仍在，加深度 / 距离真实照片数据。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-28 00:10：exp116（+GUI 操作 + 第一人称）结果：ego 成立，GUI 是文字先验；exp117 训练中，exp118 已挂接

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp）。

- **exp116**（exp115 + GUI 12k + EPIC ego 12k，对 v0.4 按图配对）：test_ego 34.7→**65.4**（+30.73 [+29.28, +32.26]），配错图低 13–25pp（action 72.4 vs 47.9），EgoThink +0.00 [−3.15, +3.15] → **ego 判据通过**。test_gui 38.6→**80.6**（+42.08 [+39.22, +45.00]），但**配错图 78.9%**（Mind2Web 配错图 78.9 > 真图 78.2）→ 模型靠 goal + 历史操作推下一步，不看屏幕，**GUI 判据不过**。旧能力：ScienceQA −1.72 [−3.05, −0.38]、dev_sqa −3.42 [−5.22, −1.70] 仍显著（承自 exp115，对 exp115 不显著）；CountBench 对 exp115 +4.28 [+1.22, +7.33] 已恢复；其余集合均含 0；图表 dev 58.5、UI dev 71.9。相对上一实验 exp115：GUI +38.1、ego +31.6，科学不变。详见 [exp116](experiments/2026-10-27-exp116-v4-gui-ego.md)。
- **exp117**（exp116 + 科学 15.8k）00:05 训练 3,500/11,612 步，loss 0.75–0.83，正常，预计 00:30 训完、01:20 评完。
- **exp118 预写并挂接**：GUI 题改成要看屏幕才能答（去掉历史操作，干扰项换成同 episode 其他步骤的 gold 操作；词重叠基线 0.43→0.26），其余与 exp116 相同；`scripts/v05/chain_exp118.sh` 在 exp117 退出后先评 v0.4 / exp116 的改写 test，再训练评测。判据：改写 test 上真图 − 配错图 ≥ 10pp 且对 exp116 下界 > 0。见 [exp118](experiments/2026-10-28-exp118-gui-screen-state.md)。

**下一步**：exp117 评完按预写判据裁决科学 / 计数；exp118 判 GUI 是否真的看屏幕；两者都过则合并配方、加 aitw-shopping / ego 扩量，定 `r22-*`。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-27 22:20：exp114 / exp115（v0.4 + 图表 + UI ± 科学 ± 计数）结果，exp116（+GUI 操作 + 第一人称）已接续开跑

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。均为实验（exp），不是正式轮次。

- **exp114**（exp81 图表 + 科学 16k + ScreenQA UI）：图表 dev 58.6%、UI dev 66.8%（v0.4 54.7%）；ScienceQA +0.57 [−0.48, +1.62] 未达"下界 > 0"；CountBench **−5.70 [−9.57, −1.83]**。未通过。
- **exp115**（去掉科学、补 3,000 计数）：图表 dev 58.9%（exp81 52.7%，+6.2），UI dev 70.7%（v0.4 +16.0）；CountBench −2.65 [−6.72, +1.22]（不再显著；相对 exp114 +3.05）；MMBench / MME / CV-Bench / NaturalBench / What'sUp / POPE / lv 三集均无显著退步；但 **ScienceQA −1.72 [−2.96, −0.48]、dev_sqa −2.08 [−3.31, −0.88]** 显著退步 → 判据 3 不过。归因对照：exp81 单独 CountBench −6.72 [−10.79, −2.65]，计数退步来自图表数据。详见 [exp115](experiments/2026-10-27-exp115-v4-chart-ui-count.md)。
- **exp116**（exp115 + GUI 下一步操作 12k〔Mind2Web 6k + AITW 6k〕+ EPIC 第一人称 12k）：已预写判据（GUI / ego 在未见 App·网站 / 未见参与者的 test 上相对 v0.4 下界 > 0、配错图低 ≥ 5pp、GUI 超过只看文字基线 +5pp；EgoThink 和旧能力无显著退步），22:15 由 `scripts/v05/chain_exp116.sh` 自动接续：先冻结数据（`data/v05/exp116/MANIFEST.json`，GUI test 1,540、ego test 5,864、EgoThink 604），评 v0.4 / exp115 基线（含 v0.4 缺的 RealWorld 明细），再训练 + 评测，预计 23:40 前后结束。见 [exp116](experiments/2026-10-27-exp116-v4-gui-ego.md)。

**23:02 补记**：exp116 训练 7,300/9,631 步，loss 0.69–0.75，正常；exp117（exp116 + exp114 科学题 15,841，92,889 题）已预写并由 `scripts/v05/chain_exp117.sh` 挂在 exp116 流水线之后自动启动，见 [exp117](experiments/2026-10-27-exp117-v4-gui-ego-science.md)。

**下一步**：exp116 评完按预写裁决；ScienceQA 退步单独修（exp117 候选：exp116 混合 + exp114 的科学题，检验能否同时守住科学与计数）。**阻塞**：ImageJevBench v0.3.0 public300 / score / 正式接入仍是外部阻断，本轮无新证据。

## 2026-10-27：exp112 未通过、exp113 排除分辨率，停止 D3 末层科学适配

**当前采用不变：v0.4（`r21-soft`）**，v0.5 未发布。GPU 实查空闲（0%、412MiB），没有训练进程；exp112 已于 10-25 20:14 跑完，结果此前没写入，本轮补裁决。

- **exp112（末十二层适配，256 步）未通过**：科学 72.27→74.22%（+1.95pp，图片配对 95% [−1.17, +5.08]），未达 +3pp 且下界 >0；图表 +2.34 [0, +5.43]，UI 不变，大小 +14.06 [+3.03, +25.81]（64 题 monitoring，只记录），位置不变。相对 exp110（72.66%）+1.56pp，相对 exp111（71.88%）+2.34pp，仍比 v0.4（83.20%）低 9pp。证据 `runs/exp112-d3-science-depth/train/report.json`。
- **错误域切片**：D3 对 v0.4 的科学差距全部集中在三族——地图 / 地理 65.6 vs 96.7（61 题）、磁铁 21.1 vs 84.2（19 题）、天气 / 气候 44.4 vs 100（9 题）；其余 132 题 D3 81.8 vs v0.4 79.5。exp110–112 的训练输入里已有数百道同族题（exp111：highlighted 218、capital 347、magnet 241），三族几乎不动；磁铁、天气题固定偏向 attract / Pair 1 / weather。
- **exp113（零训练，19 秒）**：最小图像像素 65,536→262,144，磁铁 4→3/19，地图 40→40/61（对照不变），概率确有变化（磁铁逐题最大变化 0.461）。按预写判据**排除分辨率**。
- **决定**：D3 末层适配在科学图示上停止（exp110/111/112 三次都没过门槛，不再加步数或加层）；v0.5 候选回到 deVision 自身架构（v0.4 系），D3 只作参照。详见 [exp113](experiments/2026-10-27-exp113-d3-small-image-resolution.md)，就绪矩阵已更新。

**下一步**：在 v0.4 架构上选下一个训练方向（从 exp81-chart 和 exp79 的结论出发，以真实场景集合与 v0.4 逐题配对作判据），先预写再跑。**阻塞**：ImageJevBench v0.3.0 的 public300 / score 语义 / 正式接入仍拿不到（外部），五用途正式比较做不了。


## 2026-10-25 19:20 +08:00：exp110真实科学训练完成未达门槛，exp111覆盖续训进行中

**当前阶段/实际状态**：已恢复训练优先。exp110实际完成256步/2048题并保存全部五份适配权重与优化器；未获得足够科学准确率提升，不采用。exp111从该末步权重和优化器接续，新增1024步/8192题的独立覆盖实验正在运行，19:19:37实查已新增51步，CE .658、梯度 .802，峰值显存2.89GB。Python435954、supervisor435950，前台工具会话81558；实际进程、逐步日志、checkpoint共同核对，非只读状态文件。仍采用v0.4，未发布v0.5。

- **重要，直接观察**：exp110科学256题72.2656→72.6563%，仅+0.3906pp，图片配对95%[-1.5625,+2.3438]pp，未达到+3pp且下界>0的原主要门槛；NLL .610308→.586110。UI128题100%不变，图表128题86.7188→87.5%；大小64题70.3125→65.625%（区间含0），位置64题89.0625→87.5%。不能称科学能力提升或整体替换。证据[runs/exp110-d3-science/train/verification.json](../runs/exp110-d3-science/train/verification.json)、[exp110记录](experiments/2026-10-25-exp110-d3-science-adaptation.md)。
- **已验证**：保存patch重载的科学256题与原末端概率误差<1e-5、语义一致；真图72.6563%、无图60.5469%、倒序73.4375%。原汇总漏qid报错，失败日志与执行源码保留；独立补控工具退出0、CPU复算退出0，不伪称原训练进程exit0。实际完成训练/640对逐题/768条补控，全部checkpoint保留。
- **进行中，有上限的训练假设**：[exp111预案](experiments/2026-10-25-exp111-d3-science-coverage.md)检验首轮仅覆盖每源不足11%是否限制收益；保持lr1e-4、末四层MLP/末层全注意力秩8及回放比例，额外6144道科学题ID与exp110不重复、2048条回放。末四层16模块、532480参数，未改视觉/线性注意力核。从磁盘恢复patch全部张量一致，固定640题续训起点概率最大差0，优化器/输入身份哈希落盘。不是以首轮成功为由晋级；该覆盖试验再失败则停止简单加步数。
- **数据限制**：本轮8192题5905图与保留集映射ID及可访问RGB像素交集0，证据runs/exp111-d3-science-coverage/{isolation_verification.json,train/coverage_verification.json,train/resume_verification.json}。1783个历史图片引用不可访问、近重复和外部预训练曝光仍未证明排除，不宣称完全隔离。原数据准备两次失败、原训练末尾汇总失败和唯一权重全部保留。
- **阻断发布**：科学准确率未达标；正式ImageJevBench v0.3.0认可入口/public300/score语义、五用途、独立校准、完整回归和部署验收仍缺。exp109拿到的是旧版228/456配置，不能冒充目标版本。当前证据不支持按目标正式发布，已提前报告，不降低质量标准。

**下一步/预算**：exp111预计19:40左右训练+固定复验，内部2400秒、外层2700秒，每64步独立权重+优化器，非有限/错误零loss/异常即停；监督进程存活，EXIT_CODE待实际退出采集。20:00原CC-Connect小时任务继续实查并按原门槛裁决；不启动并行GPU任务，不重复旧精度小面板。后续21–23点回归/可行独立校准，23–07点正式验收与材料，07–09点发布缓冲；正式证据缺口不解除则明确未达标。exp81控制权重2073754784 bytes保持保留。实际py_compile、独立隔离/重载/概率/NLL/bootstrap复算与git diff --check通过；未重装环境、改许可、创建递归定时任务或对外发布。


## 2026-10-25 18:04:18 +08:00：exp109评分入口版本核对、进入发布收口

**当前阶段/实际状态**：18:00训练冻结，仍采用v0.4，未发布v0.5。实读AGENTS/CLAUDE/目标/spec及最新实验日志；RTX5070实查0%/201MiB、无训练评测进程。exp81控制权重实查2,073,754,784 bytes；exp106 verification在。exp86原repro_after.log实读token1–49279<49280、image id49190一致、seq64，历史故障修复不重复。所有失败目录与唯一权重保留。CC-Connect原064e9c68仍为唯一启用小时任务，未创建/补触发或递归调度。

- **本轮完成，直接观察**：[exp109](experiments/2026-10-25-exp109-official-score.md)预写后实际下载exit0；固定commit两份公开评分入口/profile及哈希、独立静态验证落盘runs/exp109-official-score。零GPU前向、零训练、无新checkpoint。
- **阻断**：profile明确imagejevbench v0.1.，包装器要求public228/sealed456；不是目标v0.3.0 public300/sealed1200。load_system/score_group数学实现仍需维护者主机scorer.py与参考文件。不能据此宣称score契约完整或正式可执行。建议停止此旧profile检索路线；最小解除为v0.3.0认可的接入、合法public300与完整题型语义。证据原件official-profiles.json:25–30、official_score.py:70–89及verification.json。
- **发布判断**：新增[就绪矩阵](experiments/2026-10-25-v05-release-readiness.md)，分别列明正式榜单、五用途、校准、回归、性能和发布材料。exp106退化仍在，当前不支持达标发布，距截止约15小时。已授权发布有效，但不提交不达标候选。

**下一步/阻塞**：23:00前完成可复用研究材料与验收缺口收口；只推进能解除具体缺口的工作，不重复小面板或失败训练，不对失败候选拟合发布温度。周一08:00缓冲、09:00截止；正式执行接入和合格全域候选为关键阻塞。实际完成下载、哈希/AST/版本题数断言；未运行全仓测试、模型评测或正式榜单，未改环境/许可、未对外发布。


## 2026-10-25 17:05:25 +08:00：exp108官方全题型与候选适配器边界核对完成

**当前阶段/实际状态**：正式接入契约收口，仍采用v0.4、未发布v0.5。实读AGENTS/CLAUDE/目标/spec和最新日志，GPU实查0%/144MiB、无训练评测任务；exp106完整verification仍在。exp81权重实查2,073,754,784 bytes；exp86原repro_after.log确认词表映射及64图片token一致，历史故障已有修复，不重复。保留所有失败目录和唯一权重。CC-Connect仅原064e9c68每小时任务启用，未新建/补触发任务。

- **本轮完成，直接观察**：预写并执行[exp108](experiments/2026-10-25-exp108-official-contract.md)，下载exit0，5份同commit公开原件及URL/哈希归档，独立复核和静态契约断言通过；零训练、无GPU前向或新checkpoint。
- **阻断**：官方页面源码78行明确600 choice/300 Noul/300 score；找到的native_image适配器1–10行却明确proposed、未获正式认可且只处理choice，不能解除全题型接入缺口。其嵌套state.image.data也不是本项目图像传输契约，未来需显式适配并检查真正看图。当前不据未认可源码改产品API。证据`runs/exp108-official-contract/{manifest,verification}.json`及原始源码；验证标准是正式认可的运行路径、完整题型和概率评分语义。
- **能力阻断仍在**：exp106科学图示对v0.4 −11.328pp、95%区间[−17.969,−4.688]pp；正式public300、五用途显著提升、独立校准及完整回归未完成。距周一09:00约16小时，当前不支持达标发布；发布授权有效，未提交不达标候选。

**下一步/倒排**：检查固定commit的official_score.py公开依赖以定位score定义和完整执行条件，20分钟CPU预算；18:00冻结训练，23:00完成发布就绪证据矩阵与最小缺口，周一08:00缓冲、09:00截止。不给失败候选拟合虚假发布温度，不占卡空跑。实际py_compile、5文件哈希/契约断言及git diff --check通过；未运行全仓测试、正式榜单或训练，未改环境/许可、未对外发布。


## 2026-10-25 16:05:51 +08:00：exp107正式接入证据归档与接口契约验证完成

**当前阶段/实际状态**：v0.5正式评测接入收口；采用仍为v0.4、未发布v0.5。实读AGENTS/CLAUDE/目标/spec及最新实验训练记录，实查RTX5070 0%/144MiB，无训练评测进程；exp106完整逐题和report仍在，无活跃任务需干预。exp81控制权重实际2,073,754,784 bytes。CC-Connect原064e9c68启用、proj-codex、每小时整点/new_per_run/50分钟，仅1项，info last_run=15:07:14；本轮未新增/补触发任务。零训练、无新checkpoint，全部失败目录和唯一权重保留。

- **已完成，直接观察**：预写并执行[exp107](experiments/2026-10-25-exp107-official-access.md)，实际exit0。固定发布者commit `4365ca7ec2ee12ea952079185cbf68f37bbf7ff1`，完整tree 43,352条/truncated=false；官方results与示例manifest原始响应、URL、哈希持久化于`runs/exp107-official-access/`。官方聚合source hash/built_at未变，示例明确8题public_examples_only，本次未取得public300；没有把路径搜索无结果写成题集不存在。官方[data card](https://benchmarkheaven.com/jev-models/data)仍仅提供系统级结果；[提交页](https://benchmarkheaven.com/submit)普通队列免费、须审核、时间由维护者决定，本轮未提交。
- **阻断，直接观察**：`src/devision/infer.py:204`拒绝score，现有实际decide契约测试16 passed/21 deselected、exit0（含score拒绝）；说明当前规格行为正常，但完整正式评测兼容尚未实现。最小解除材料为明确score输入/输出及评分语义、合法public300归档或正式可执行接入。不能靠ECE替代官方Calibration。
- **重要，历史结果未改变**：exp106科学图示相对v0.4 −11.328pp，图片配对95%区间[−17.969,−4.688]pp；不整体替换、不扩大exp94/96已失败配方。exp86原首错169占位/64视觉特征已有seq64修复和完整结果，tokenizer/config一致，不原样重复故障。

**下一步/倒排**：下一轮优先从本轮固定tree定位官方score契约/评分及执行代码，30分钟CPU预算，形成明确可测适配再实现；不给失败候选做发布温度。18:00训练冻结，23:00完成可行回归/独立校准与缺口材料，周一08:00缓冲、09:00截止。距截止约17小时，正式public300、五用途显著改善、独立校准与完整回归仍缺，当前证据不支持按期达标发布；发布授权有效，未上传未达标版本。

**改动/验证**：新增exp107预案/脚本/原始归档与结果，追加接入调研。实际元数据实验及16项现有接口测试完成；没有模型训练、GPU前向、全仓测试或正式榜单执行，未改依赖/许可。


## 2026-10-25 15:07:01 +08:00：exp105完整复核、exp106 FP32科学图示退化确认

**当前阶段与实际状态**：小型预训练骨干基线回归；仍采用v0.4，未发布v0.5。读取项目约定/目标/spec、最新实验/训练日志，实际GPU空闲且无训练评测后预写并串行运行exp106。768次CUDA前向完成、49.86秒、峰值3.55GB；结束GPU0%/144MiB、无本实验进程，零训练、无新checkpoint。外层shell未写出EXIT_CODE，原返回码未知，不以完整report冒充exit0；独立复算实际exit0。原launcher和日志保留，后续用前台工具会话/独立supervisor采集返回值。exp81控制权重实际2,073,754,784 bytes，全部失败目录及唯一权重保留。CC-Connect实查仅原064e9c68每小时启用，未创建、补触发或递归调度。

- **已完成，直接观察**：exp105两设备各EXIT_CODE=0和24条明细；report_exp105.py复算通过，四域跨设备最大概率差9.8733e-6、重复差0、语义一致。CPU4线程p50/p95=4653.84/11031.03ms，CUDA205.30/506.52ms、峰值3.74GB；12题无预热monitoring，非正式速度admission。证据`runs/exp105-d3-crossdomain-device/verification.json`及[exp105追加](experiments/2026-10-25-exp105-d3-crossdomain-device.md)。
- **重要，退化未消除（直接观察）**：固定exp99全部256图三臂，只改FP32，真图184/256=71.875%、无图153/256、倒序185/256；对v0.4 213/256差−11.328pp，图片配对95%区间[−17.969,−4.688]pp。看图增益+12.109pp [5.859,18.359]pp，倒序语义改变6.25%，原门槛失败。对BF16仅1题语义改变但概率最大差.115891，不能继承其校准。建议与决定：不整体替换v0.4，不扩大失败训练，不再重复本面板精度诊断；验证依据为完整三臂及v0.4逐题标签/概率/NLL/配对复算通过，仍非正式五用途或独立测试。证据[exp106](experiments/2026-10-25-exp106-d3-fp32-scienceqa.md)、`runs/exp106-d3-fp32-scienceqa/{report,verification,fp32-comparison}.json`及1024条明细。
- **exp86故障核实**：本轮读取PIL2原始masked_scatter首错和seq64 repro_after.log：token1–49279<49280、image id49190一致、processor64。历史169占位/64特征不匹配已有修复及512题产物，未重复故障、未重装环境；pad警告不是已证实根因。
- **阻断发布，提前报告**：正式public300合法归档/执行接入缺口沿用历史核查，本轮未重复联网、不声称题集不存在；五用途显著提高、score契约、独立校准及完整外部回归未完成，且科学图示退化在FP32下仍有证据。距周一09:00约18小时，现有证据不支持达标发布，也无法保证维护者按时接受上榜。已获发布授权，未上传未达标候选。

**下一步/倒排**：优先正式ImageJevBench题集/执行接入的版本化材料核对与缺口收口，不再继续小面板精度复跑；新训练需有不同监督/可训层假设及预写门槛。18:00训练冻结、23:00可行回归/独立校准与缺口材料、周一08:00材料缓冲、09:00截止。最终候选未达能力门槛前不把开发集当独立校准，不生成虚假达标模型卡。

**改动与验证范围**：新增exp106预案、运行及复算脚本，追加exp105结果；实际py_compile、768次CUDA前向、完整1024条逐题与bootstrap复算、git diff --check通过。未运行全仓测试、未更改依赖或许可、未外发发布、未干预他人任务。


## 2026-10-25 14:05 +08:00：exp105非UI跨设备FP32验证已启动

**当前阶段/实际状态**：仍采用v0.4，未发布v0.5。实读AGENTS、CLAUDE、目标/spec及最新exp103/104与日志，核对GPU0%/144MiB且无训练评测；预写[exp105](experiments/2026-10-25-exp105-d3-crossdomain-device.md)后启动CPU→CUDA串行固定面板。总timeout PID421869、CPU Python421875，14:04:51已有首条真实图表调用落盘（2.702秒），CPU217%、RSS约4.5GB；GPU仍空闲。不能据进程存活声称通过，最终结果待核实。日志`runs/exp105-d3-crossdomain-device/supervisor.log`，身份及逐题在cpu/cuda-0子目录；总预算1800秒、子阶段850秒、内部800秒、异常即停，独立EXIT_CODE。零训练、无新checkpoint。

- **直接观察**：exp103 verification确认UI跨设备最大差2.767e-5、重复差0；exp104 verification确认587题FP32准确率61.16%、对v0.4 +14.48pp。支持继续FP32部署验证，不代表五用途或概率跨精度等价。exp105固定四域各3题、每设备两轮，按id哈希选题而非按误差，判据为跨设备≤.005、重复≤1e-5且语义一致。独立复算入口`scripts/v05/report_exp105.py`已编译，待完整两设备exit0后执行。
- **exp86故障原证据**：本轮实读PIL2首个masked_scatter断言及seq64 repro_after.log；token1–49279<49280、image id49190一致、processor64，原169占位/64特征不匹配已有修复与512题结果。未重复已修复故障。exp81控制权重实际2,073,754,784 bytes，全部失败目录与唯一权重保留。
- **阻断发布**：public300合法完整归档/正式执行入口缺口沿用历史核查，本轮未重复联网，不声称题集不存在；五用途显著改善、score、独立校准、完整回归仍未完成，科学图示/大小位置退化仍在。距周一09:00约19小时，当前证据不支持达标发布，正式接受日期也无法保证。发布授权有效，未上传不达标候选。

**下一步/倒排**：核对exp105两阶段退出、完整48条及逐域概率，运行独立复算；失败保留现场并按域定位。18:00训练冻结，23:00完成可行回归/校准与缺口材料，周一08:00材料缓冲、09:00截止。最终模型/精度未冻结前不把开发集当独立校准。CC-Connect原064e9c68每小时enabled、仅1项，未创建或补触发任务。实际py_compile及git diff --check通过，未运行全仓测试，未重装环境、变更依赖/许可、触碰他人进程或外发发布。


## 2026-10-25 13:09:04 +08:00：exp103完整复核、exp104 FP32真实照片回归完成

**当前阶段/实际状态**：小型图文预训练D3基线的部署精度与真实照片回归；仍采用v0.4，v0.5未发布。完整读取项目约定/目标/spec、最新实验与日志，实查GPU空闲、无训练评测后预写exp104；首轮输出适配遗漏exit1，保留目录，修正noul后独立retry1实际587题exit0。结束GPU0%/144MiB、无本实验进程，零训练/无新checkpoint。exp81控制权重实查2,073,754,784 bytes，所有唯一权重和失败目录保留。CC-Connect原064e9c68每小时启用，仅1项，未新建或递归调度。

- **已完成，直接观察**：exp103两设备各67调用、子阶段EXIT_CODE均0，完整逐题独立复算；32题两轮跨设备最大概率差.00002767、同设备重复差0、语义一致，原数值门槛通过。CPU4线程p50/p95=11.435/12.030秒，CUDA493.54/515.52ms、峰值3.74GB；仅UI本地system_one/非HTTP，不能外推全用途。根目录无总EXIT_CODE，不虚构launcher退出状态。证据`runs/exp103-d3-fp32-deployment/verification.json`及两设备原始产物；[exp103追加](experiments/2026-10-25-exp103-d3-fp32-deployment.md)。
- **已完成，直接观察**：exp104真实照片587题/585图，FP32 61.1584%、NLL .799531，对历史BF16 −.1704pp、NLL +.001113，通过预写部署筛选；对v0.4 +14.4804pp，图片分组95%区间[9.3697,19.6581]pp。仅1题语义改变，但最大概率差.097842，须按最终路径独立校准，不能称概率等价或直接继承BF16校准。229.35秒、峰值3.74GB，概率/标签/NLL和历史同图gold核对、5000次bootstrap通过。证据`runs/exp104-d3-fp32-realworld-retry1/{verification.json,report.json,details.jsonl,IDENTITY.json,EXIT_CODE}`；[预写与结果](experiments/2026-10-25-exp104-d3-fp32-realworld.md)。该集已用于路线筛选，monitoring；预训练曝光未知，不是正式五用途。
- **exp86故障实查**：本轮读取原repro_after.log，token1–49279<49280、image token49190一致、processor64；原PIL2首错169占位/64特征masked_scatter不匹配，已有完整修复评测。pad token警告仍有但不是已证实首错根因；未重复旧失败或已通过的CUDA诊断。
- **阻断发布，提前报告**：正式public300合法归档/执行接入缺口沿用此前核查，本轮未重新联网，不声称题集不存在；五用途显著改善、score、独立校准、完整外部回归仍未完成，科学图示/大小位置已有退化，D3不整体替换v0.4。当前证据不支持达标发布，距周一09:00不足20小时，正式被接受上榜无法保证；发布授权有效，未上传未达标候选。

**下一步/倒排**：FP32保留为可验证计算路径，下一轮优先非UI跨设备固定面板（图表/文档/照片和noul，约10分钟），验证数值结论适用范围；停止扩大exp94/96失败配方。18:00冻结训练，23:00完成可行回归/校准和缺口材料，周一08:00发布材料缓冲、09:00截止。仅在最终模型/精度冻结且有独立数据时做正式校准，不用本轮开发集代替。

**改动/验证范围**：新增exp104预案、运行与独立复算脚本，追加exp103实际结果；实际py_compile、CUDA587调用、离线配对与概率/NLL复算、git diff --check通过。首轮noul适配失败完整保留；未运行全仓测试，未重装环境、改依赖或许可、对外上传发布。


## 2026-10-25 12:05:11 +08:00：exp103 FP32完整32题重复性与端到端成本复验进行中

**当前阶段/运行状态**：已完整读取AGENTS、CLAUDE、v0.5目标与spec、最新exp102及训练日志，核对实际GPU0%/144MiB且无训练评测后，预写[exp103](experiments/2026-10-25-exp103-d3-fp32-deployment.md)，在现有uv环境启动真实CPU→CUDA串行评测。总timeout PID416682，CPU Python PID416687；模型已完成加载，12:05实查已落盘5条真实调用（含3次warmup），CPU约383%占用，GPU仍0%/144MiB；尚无最终结果，不把进程存活视为验收通过。总预算1800秒、每设备内部800秒，逐题flush，错误即停，阶段退出码保留。日志`runs/exp103-d3-fp32-deployment/supervisor.log`，逐题和身份在其cpu/cuda-0子目录。零训练，无新checkpoint。

- **重要，直接观察**：exp86原始repro_after.log确认token1–49279<49280、image token49190一致、processor image_seq_len64；历史PIL2原因是169占位/64特征不匹配，已有修复完整产物；本轮不重复故障或已失败配方，全部失败目录保留。exp81控制权重实际2,073,754,784 bytes，未改动。CC-Connect仅既有064e9c68每小时启用，本轮未创建或补触发任务。
- **待验证**：exp103固定exp101全部32题、原生system_one、加载BF16权重转FP32、CPU4线程，各设备3次warmup和两轮32题。标准是跨设备概率最大差≤0.005、同设备重复≤1e-5并语义一致；记录全部延迟/概率，不挑较好轮。FP32四题成功尚不能外推全用途；本轮未完成校准或产品集成。py_compile已通过。
- **阻断发布/路线决定**：当前仍采用v0.4。D3保留UI/文档/图表研究基线，但科学图示/旧大小位置已有退化，不支持整体替换，不扩大exp94/96失败训练。正式public300合法归档/执行接入缺口、五用途显著改善、score契约、独立校准和完整回归仍缺，现有证据不支持正式达标发布，周一09:00期限高风险；沿用已有接入核查证据，本轮未新联网、不声称题集不存在。

**下一步/倒排**：本轮评测结束按完整32题及重复结果决定可用部署精度和CPU成本，下一小时核对退出码、逐题完整性与GPU再做分析，异常保留目录。18:00训练冻结、23:00完成可行回归/校准和缺口材料，周一08:00发布材料缓冲、09:00截止；只有证据支持才使用已授予发布权限，不以局部UI或数值通过冒充目标完成。未改依赖/许可，未重装环境，未上传发布或触碰他人进程。


## 2026-10-25 11:07 +08:00：exp102固定输入精度诊断完成，FP32值得部署复验

**当前阶段与实际状态**：小型预训练D3部署数值诊断；仍采用v0.4，无达标v0.5候选。读取项目约定、目标/spec与最新实验，实查GPU0%/144MiB且无训练评测后预写执行exp102；20条固定输入前向与8次原生调用核对、118.19秒、实际exit0。结束GPU0%/144MiB、无本实验进程。exp81控制权重实查2,073,754,784 bytes，全部失败目录/唯一权重保留。CC-Connect原064e9c68 enabled、每小时整点/new_per_run/50分钟，last_run10:10:26，无last_error，未创建/补触发任务。

- **重要，直接观察**：exp101差值最大的固定4题、2张截图，CPU生成一次输入张量并保存跨设备复用。BF16最大概率差0.01394290>0.005，FP32降至0.00002772，BF16 CUDA math仍0.01543504；原生接口与固定张量前向最大误差2.10e-7。支持精度相关数值路径是主要因素，单改注意力后端不足，尚未定位具体层/算子。证据[exp102](experiments/2026-10-25-exp102-d3-device-numerics.md)、`runs/exp102-d3-device-numerics/{report.json,verification.json,details.jsonl,inputs.pt,EXIT_CODE}`。验证标准为固定输入下症状复现及各单变量比较，20条概率/候选映射/差值离线复算通过。
- **重要，未验证部分**：本轮BF16与exp101同题最大差0.02597889，未精确复现原0.03625573，运行间/冷暖状态重复性仍待区分。选择后4题不能证明全用途等价；FP32仅为候选计算路径，尚未测其端到端成本和完整概率稳定性，独立校准也未完成。下一项固定全部32题加入重复调用/冷暖与FP32成本（约10分钟），完成条件是明确重复性及可验证部署精度，不原样扩量BF16或声称已修复全部原因。
- **exp86故障复核**：实际读取after原日志：token1–49279<49280、image id49190一致、processor64；历史PIL2为169占位/64特征不匹配且已有完整修复CUDA产物。本轮未重复已通过复现、未重装环境。
- **阻断发布，提前报告**：正式public300合法归档/执行接入缺口沿用已核查证据，本轮未重复联网；五用途显著改善、score产品契约、独立校准、完整外部回归尚未完成，D3科学图示/旧大小位置已有退化。现有证据不支持正式达标发布，周一09:00期限高风险；发布授权有效，未提交不达标候选。

**倒排与下一步**：12:00依据能力与成本证据决定路线，不扩大exp94/96失败配方；18:00训练冻结、23:00可行回归/独立校准/性能、周一08:00材料与09:00截止。主指标接入与现有退化未解除前，不以局部UI、FP32四题成功替代目标。

**改动与验证范围**：新增exp102预案/结果和诊断脚本；实际py_compile、20条前向、8次原生核对、独立离线复算及真实退出/GPU检查完成。零训练、无新checkpoint，未运行全仓测试，未改环境/许可或外部上传发布。

## 2026-10-25 10:10:18 +08:00：exp101 CPU/CUDA端到端成本完成，设备概率筛选未通过

**当前阶段**：小型图文预训练D3基线部署成本验证；仍采用v0.4，无达标v0.5候选。实查空闲后预写并CPU→CUDA串行完成各35调用，实际退出码均0；结束GPU0%/144MiB，无对应进程。exp81控制权重实查2,073,754,784 bytes；失败目录及唯一权重保留。CC-Connect原任务064e9c68实查启用、每小时整点，未新增/补触发递归任务。

- **重要，直接测量**：32题UI热调用，CPU4线程p50/p95=11.00/11.35秒，CUDA=235.30/253.18ms、峰值1.99GB。原生backbone BF16/readout float32、D3_FAST=0，参考算子环境；非HTTP、非全用途或正式榜单计时。证据[exp101预案与结果](experiments/2026-10-25-exp101-d3-deployment-cost.md)、`runs/exp101-d3-deployment-cost/{cpu,cuda-0}/report.json`及逐题/身份/退出码。CPU代价明显，未安装新依赖。
- **重要，数值筛选失败**：32/32语义答案一致，概率最大差0.03626，超过预写0.005；原因待验证，不能声称概率跨设备等价。建议下一轮取最大差题，固定相同输入张量、单变量精度/算子诊断（预计10–20分钟），完成条件是定位差异来源并选择可验证部署路径，不原样扩量。
- **重要，部署契约差异**：固定revision原生`d3_runtime.py:316–343`使用熵式confidence，与项目Jev公式不同；筛选概率/准确率不受其影响，但正式集成需重建并验证confidence。尚未改产品API；score正式schema未验收。
- **exp86故障核对**：原始repro_after.log确认token范围1–49279<49280、image id两端49190，processor64；历史PIL2失败为169占位/64特征不匹配且已有完整修复CUDA产物。未重复故障/重装环境。
- **阻断发布，提前报告**：正式public300归档/可执行接入缺口仍在（沿用历史核查，本轮未重复联网）；五用途、独立校准、完整外部回归、score与部署验收未完成，且科学图示/旧大小位置已有退化。当前证据不支持正式达标发布，周一09:00目标高风险。发布授权继续有效。

**倒排与下一步**：12:00据能力/成本证据确定路线，优先上述数值诊断与正式接入材料；18:00冻结训练，23:00完成可行回归/独立校准/性能，周一08:00材料、09:00截止。只在新监督或可训层假设成立时启动新训练，不扩大exp94/96失败配方。

**改动与验证**：新增exp101预案、成本脚本、两设备实际70次调用和离线配对；py_compile通过，未运行全仓测试、未训练或生成checkpoint、未改依赖/许可、未上传发布。

## 2026-10-25 09:06:00 +08:00：exp100真实计数全量开发对照完成，尚不支持D3整体替换

**当前阶段与实际状态**：小型图文预训练骨干基线与真实数据覆盖；仍采用v0.4，无合格v0.5候选、未发布。已读取AGENTS/CLAUDE、目标/spec、最新实验和训练日志；实查起始GPU空闲、无训练评测后预写并串行执行exp100。真实1473次CUDA前向、94.76秒、峰值1.99GB、exit0，结束GPU0%/144MiB且无对应任务。exp81控制权重实查2,073,754,784 bytes，所有唯一权重及失败目录保留。CC-Connect仅既有064e9c68启用、每小时整点，未创建、补触发或递归定时任务。

- **重要，计数扩大门槛未通过（直接观察）**：全部491道CountBench派生choice，D3真图268/491=54.58%、无图26/491=5.30%、倒序264/491=53.77%，v0.4历史同题246/491=50.10%。差+4.48pp，图片配对bootstrap95%区间[−1.43,+10.39]pp；真图-无图+49.29pp [44.81,53.77]pp，语义改变15.07%。具视觉依赖，但不满足预写相对v0.4至少+5pp且下界>0，不扩大本域直接验证或重复失败恢复训练。证据[exp100](experiments/2026-10-25-exp100-d3-countbench.md)、`runs/exp100-d3-countbench-retry1/`的report/verification/IDENTITY/四臂明细/EXIT_CODE。1964条标签、概率与NLL独立复算，491张RGB像素与历史baseline哈希全部一致。monitoring、未知预训练曝光、非原生开放计数、未独立校准，不代表正式检查用途达标。
- **exp86故障优先复核（直接读取原始证据）**：实际`repro_after.log`词表49280、token1–49279、image token49190两端一致；历史PIL2首错为169占位/64视觉特征masked_scatter长度不匹配，seq64已有完整CUDA产物。pad token警告尚非已确认根因；未原样重复已修复故障或重装环境。本轮CPU选样脚本误替换哈希名导致首轮exit1，已修正、独立retry1成功，两个目录/日志均保留，未掩盖失败。
- **探索性错误分析（非新主判据）**：实际查看ScienceQA val-2532地图；固定国家高亮问题12题D3对5/v0.4对11，其余244题180/202，不能把全部科学集差距归因为地图或选项映射。证据`runs/exp99-d3-scienceqa/country-question-analysis.json`与exp99追加；仅抽看一图，不称全量标签审计。
- **阻断发布，提前报告**：正式ImageJevBench public300合法归档/执行接入仍缺，沿用历史调查、本轮未重复联网探测；五用途显著收益、score接入、独立校准、完整外部回归与CPU/CUDA部署验收未完成。现有D3还存在科学图示及旧大小/位置退化，当前证据不支持整体替换或正式达标发布。授权有效但不发布未达标候选；距周一09:00约24小时，无法保证按期被正式接受上榜。
- **下一步与倒排**：下轮优先做D3完整输入/结构化输出部署及CPU成本测量，为12:00路线决定补成本证据（预计10–20分钟，GPU部分串行）；仅在新的真实监督/可训层假设和保护门槛成立后启动新训练，不扩大exp94/96原配方。18:00冻结训练，23:00完成可行的回归/独立校准/性能，周一08:00材料、09:00截止。正式题集接入仍为最小外部缺失，局部代理不替代目标。

**改动与验证范围**：新增exp100预案/结果与运行、复算脚本，追加exp99错误分层。实际py_compile、1473次CUDA前向、1964条离线复算及491图历史身份检查完成；无新checkpoint，未运行全仓测试、未改依赖/许可、未对外上传或发布。

## 2026-10-25 08:04:44 +08:00：exp99科学图示筛选未通过，D3整体替换风险确认

**当前阶段与实际状态**：小型图文预训练骨干基线与真实数据覆盖，采用仍v0.4；无合格v0.5候选、未发布。先读取项目约定/目标/spec、最新实验日志，实查GPU空闲且无训练评测后预写并串行执行exp99。实际CUDA768次前向、47.38秒、峰值1.82GB、真实exit0；结束GPU0%/144MiB，无对应Python。无训练、无新checkpoint；exp81唯一控制权重2,073,754,784 bytes及所有失败目录保留。CC-Connect原任务064e9c68实查enabled、proj-codex、每小时整点/new_per_run/50分钟，last_run07:08:55；未新建、补触发或递归调度。

- **重要，图示代理筛选未通过（直接观察）**：预写[exp99](experiments/2026-10-25-exp99-d3-scienceqa.md)，对scienceqa.unseen按图片ID固定哈希选256题，保留原state_text；D3真图185/256=72.27%、无图153/256=59.77%、倒序185/256，v0.4历史同题213/256=83.20%。D3-v0.4 −10.94pp [−17.97,−4.30]，真图-无图+12.50pp [6.25,18.75]，倒序语义改变5.86%。虽有视觉收益，整体替换门槛失败，停止本域晋级，不扩大已失败恢复配方。证据`runs/exp99-d3-scienceqa/{report.json,verification.json,IDENTITY.json,EXIT_CODE}`与四臂明细。验收标准为原预写双收益与顺序门槛，本轮第一项未过；这是monitoring科学图示代理，不是官方五用途结果。
- **探索性分析，原因待验证**：四选98题D3 70.41%/v0.4 93.88%，是主要差距；三选64题76.56%/73.44%。选项数和内容混杂，不能认定token/映射bug。v0.4独对56、D3独对28；错误清单及分层在`slices.json`/`v04_only_correct.jsonl`，尚未视觉抽查。原生NLL未校准，不作固有概率质量比较。
- **exp86复核（直接读取原始证据）**：after日志token1–49279<49280，image id49190两端一致、占位64；PIL2原故障是169/64的masked_scatter长度不匹配，已有完整CUDA修复产物。pad token警告不等于根因；未重复历史失败或再次空跑已通过复现。
- **下一步与预算**：优先补计数/检查域的固定真实外部对照（数分钟CUDA），并少量视觉核查本轮四选错误，再于12:00做路线决定。18:00训练冻结、23:00完整回归/独立校准/CPU与CUDA性能、周一08:00材料、09:00发布截止不变。新训练必须先有区别于exp94/96的新假设与预写门槛，不能因期限扩大失败配方。
- **阻断发布，提前报告**：正式ImageJevBench public300/接入缺口沿用历史证据，本轮未重新联网探测；五用途显著收益、score、独立校准、完整回归与CPU/CUDA部署验收尚未完成，本轮又确认科学图示代理明显退化。距截止约25小时，按时完整达标风险高；发布授权有效，但现有证据不支持正式达标发布。最小外部缺失仍是合法public300归档/manifest或正式可执行接入，局部UI/文档/图表不能替代。

**改动与验证范围**：新增exp99预案/结果、筛选与独立复算脚本；实际py_compile、768次CUDA、四臂1024条概率/标签/NLL与图片配对bootstrap复算通过，退出码核实0。未运行全仓测试、未重装环境或更改许可，未上传/发布。

## 2026-10-25 07:08:00 +08:00：exp98真实扫描表单绑定筛选通过，完成v0.4同题参照

**当前阶段与实际状态**：小型图文预训练D3基线与真实数据覆盖，仍采用v0.4，v0.5未发布。实查起始GPU空闲且无训练/评测，CPU审计后串行完成D3 333次、v0.4 111次CUDA前向，均真实exit0；结束GPU0%/144MiB、无对应任务。D3耗时35.54秒、峰值1.88GB。无新训练或checkpoint；exp81控制权重实查2,073,754,784 bytes，所有失败目录、唯一权重及exp96补丁保留。CC-Connect仅原任务064e9c68启用、proj-codex、每小时整点/new_per_run/50分钟，实查last_run06:05:51；未创建或补触发任务。

- **已完成，直接观察**：预写[exp98](experiments/2026-10-25-exp98-funsd-binding.md)，从作者固定commit取得FUNSD原包，只解析原train149页，按文档像素哈希固定30页开发。保守唯一实体链接转换得到训练414题/106页、开发111题/29页，原test未解析；与既有v05共42,306个图片路径像素交集0，train/dev交集0。全部525题链接、gold文本、框范围和候选实体映射独立核对通过。证据`data/v05/funsd/derived/{audit.json,manifest.jsonl}`及`runs/exp98-funsd-audit/`。近重复/预训练曝光未知，包内没有数据LICENSE，只本地研究、不再分发；实际视觉抽看两页，未声称完成开训前50题检查。
- **重要正结果，限派生文档开发集（直接观察）**：D3真图109/111=98.20%、无图71/111=63.96%、倒序109/111，语义改变率0；真图减无图+34.23pp，29页聚类bootstrap95%区间[25.66,43.36]pp，通过预写门槛。v0.4真图38/111=34.23%，D3差+63.96pp [55.65,71.56]pp。证据`runs/exp98-d3-funsd-binding/{report.json,verification.json,IDENTITY.json,EXIT_CODE}`及四臂完整逐题（v0.4在`runs/exp98-v04-funsd/`）；复算`scripts/v05/report_exp98.py`。NLL均复核，但温度/分辨率/预训练不同，不把跨模型原始NLL差当固有校准差异，也不称官方FUNSD或五用途得分。
- **方向决定**：该子集已接近满分，且无图文本捷径明显，不立即对其训练或重开exp96。新训练池保留为未来真实文档回放候选，开训前仍需50题视觉审计；下一项优先图示与检查域基线、同类型困难文档负例，不在已解子集增加预算。原test留给将来固定验证，不据本轮结果调它。
- **exp86实际证据核对**：PIL2原日志首错masked_scatter totalElements<=srcSize，既有after日志token1–49279<49280、image id49190一致、占位64；seq64完整512题产物仍在。根因是169占位/64特征长度不匹配，非已确认词表越界；本轮未重复已修复故障或重装环境。
- **阻断发布，提前报告**：本轮重新读取[正式提交页](https://benchmarkheaven.com/submit)，免费队列及所有提交的代码审核由维护者安排，周一09:00被接受无法保证；public300缺归档沿用历史核查，本轮未重探题集。正式主榜、五用途显著收益、score适配、独立校准、完整外部回归及CPU/CUDA部署验收仍未完成，现有证据不支持达标发布。发布授权有效，但不上传未达标候选。12:00路线决定、18:00训练冻结、23:00回归/校准/性能、周一08:00材料、09:00发布缓冲不变，按时完成完整目标风险高。

**改动与验证范围**：新增exp98预案/结果、FUNSD转换/D3基线/独立复算脚本，补充文档数据来源与正式入口核实。实际py_compile、525题结构/链接验证、444次CUDA前向、四臂逐题概率与NLL复算、checkpoint身份哈希完成；未运行全仓测试、未改环境或许可、未对外提交/上传。

## 2026-10-25 06:05:00 +08:00：exp97错误交集与面积语义/LoRA组合诊断完成

**当前阶段与实际状态**：小型预训练D3路线诊断，仍采用v0.4，v0.5未发布。已阅读项目约定/目标/spec、最新实验训练日志并实查GPU和进程；启动前空闲，exp97真实进程退出0，结束后GPU0%/144MiB、无训练评测任务。exp81控制权重实查2,073,754,784 bytes，全部失败目录、唯一权重保留。CC-Connect只有原任务064e9c68，enabled、每小时整点、project=proj-codex、new_per_run；last_run实查05:07:15（本轮尚未返回，不能据此断言06点未执行），未创建或补触发新任务。

- **已完成，直接观察**：预写[exp97](experiments/2026-10-25-exp97-size-error-factorial.md)，CPU复算错误交集后，只读加载exp96 step64补丁，执行128次CUDA前向，12.61秒、峰值显存1.82GB。64道大小题原生45对、面积语义51对、LoRA47对、组合56对；原生温度NLL依次.539521/.478905/.482325/.423011。组合对面积语义+7.8125pp、60图bootstrap95%区间[0,16.129]pp，下界未大于0，**预写晋级门槛未通过，不扩大exp96**。单独LoRA两道修复与语义七道修复重合一道；位置LoRA未改变任何对错身份。组合交互只是假设。证据`runs/exp97-size-error-factorial/{report.json,error-intersections.json,combined.details.jsonl,verification.json,EXIT_CODE}`与`runs/exp97-size-error-factorial.log`；monitoring小切片不能替代真实五用途验收。
- **exp86故障核实**：读取原before/after复现日志，token范围1–49279<49280、processor/config image id49190一致；首错为169占位/64视觉特征不匹配，seq64已修复且有完整CUDA结果。本轮未重复失败实验或已多次通过的复现，无需重装环境。pad token警告不是已确认根因。
- **阻断发布，直接与历史证据分开**：本轮重新读取[正式提交页](https://benchmarkheaven.com/submit)，免费队列与代码审核仍由维护者安排，无法保证周一09:00正式接受；主榜抓取触及大小上限。public300缺归档沿用历史调查，本轮未重复探测。五用途显著收益、score接入、独立校准、完整外部回归及CPU/CUDA部署验收仍未完成；现有证据不支持达标发布。已获版本发布授权，但没有发布未达标权重或材料。
- **下一步与期限**：停止本面板继续提示搜索和末层q/v扩量，优先准备真实文档字段绑定的小预算数据/对照，同时补图示、检查域覆盖；正式300需合法题集归档及manifest或官方可执行入口。12:00路线决定、18:00训练冻结、23:00回归/独立校准/性能、周一08:00材料、09:00发布缓冲不变；当前约27小时，完整目标按时完成风险高，不能用局部收益缩减目标。

**文档与验证范围**：新增exp97预案/结果、诊断脚本，追加正式入口核实；实际py_compile、CUDA四臂前向、面积语义逐题重现误差0、64条概率有限/归一化与准确率/NLL独立复算通过。无新训练、无新权重，未运行全仓测试，未更改环境/许可或对外上传。

## 2026-10-25 05:07:06 +08:00：exp96末层LoRA恢复pilot完成，未通过扩量门槛

**当前阶段与实际状态**：D3小型预训练骨干恢复实验，仍采用v0.4，v0.5未发布。检查实际GPU/进程空闲、exp94/95日志和产物后串行执行exp96，64次更新与384题验证完整，report和日志末尾汇总一致，耗时107.35秒、峰值显存2,031,112,192 bytes；结束后实际GPU0%/144MiB，无训练评测进程。注意launcher未写出EXIT_CODE，原shell退出码未知，不以完整report冒充exit0；此证据缺口不改变已完整逐题复算的结果。exp81控制权重实查2,073,754,784 bytes，全部失败目录和唯一权重保留。CC-Connect原任务064e9c68启用、每小时整点/new_per_run/50分钟，未创建新任务。

- **重要，未通过预写规则（直接观察）**：四源256图一遍训练、末层q/v rank4、26,624可训参数、AdamW lr1e-3、batch4累积64更新。大小45/64→47/64（+3.125pp，图片bootstrap95%区间[0,8.065]pp），位置57/64不变，图表111/128、UI128/128不变。大小NLL .53952→.48233，位置 .26000→.26152，图表 .31442→.31428，UI .02090→.01066，均为原生温度未校准。位置未达+3pp，停止扩大本配方；大小区间下界0，不能称显著改善，准确率不变也不能证明完整等效。证据 `runs/exp96-d3-lora/report.json`、`validation.details.jsonl`、`verification.json`及`supervisor.log`；方案[exp96](experiments/2026-10-25-exp96-d3-lora-pilot.md)。这些是monitoring切片，不是正式五用途测试。
- **工程与隔离验证（直接观察）**：exp93全部10份训练/验证源及COCO/VG alias哈希未变，训练inputs哈希匹配；384题原生概率与exp94基线最大误差0，保存补丁实际重载并注入runtime首题概率误差0，全部概率有限/归一化通过。step16/32/48/64补丁及优化器均保留，IDENTITY记录代码commit和实际脚本/配置/输入哈希。未实现自动续训，恢复需按保存的step及固定shuffle重建；补丁依赖原始基座，尚非完整发布checkpoint。近重复与外部预训练曝光仍未知。不同minibatch的CE不用于声称训练前后下降。
- **exp86历史故障核对**：原修复记录表明image id49190两端一致、输入1–49279<49280，PIL2首错为169占位/64视觉特征masked_scatter长度不匹配，seq64已有完整CUDA评测通过。本轮不重复历史失败和多次通过的诊断；本次训练/验证未出现CUDA断言。
- **下一步**：停止扩大本末层q/v配方。下一轮先做现有大小/位置错误交集与面积定义修正的CPU分析（约15分钟），形成可区分的新监督或可训层假设后才开新训练；优先补文档/图示/检查真实数据覆盖和正式基准接入，不继续以局部恢复代替五用途。长任务启动改用可靠等待/退出码采集，保留本轮原launcher证据。12:00路线决定、18:00训练冻结、23:00回归/独立校准/CPU与CUDA性能、周一08:00材料及09:00发布缓冲保持。
- **阻断发布，提前报告**：正式ImageJevBench public300/评测接入仍缺合法归档或正式可执行入口（本轮未联网，沿用历史证据）；五用途显著收益、score接口、独立校准、完整外部回归和CPU部署验收未完成。现有证据不支持达标发布，无法保证周一09:00被接受上榜；不发布未达标候选。发布授权已收到，无需再次确认。

**改动与验证范围**：新增exp96预案、独立YAML和训练脚本，实际py_compile、CUDA训练/384题验证、补丁重载、原生基线比较及离线NLL复算通过；未运行全仓测试、未重装环境或更改许可，未对外上传。

## 2026-10-25 04:04:41 +08:00：exp95末层低秩适配工程探针完成

**当前阶段与实际状态**：D3恢复训练接口准备，仍采用v0.4，v0.5未发布。本轮完整读取AGENTS/CLAUDE、v0.5目标与最新exp94/训练日志，并检查实际GPU、进程及产物：exp94独立验证已exit0，但大小恢复门槛失败，停止扩大冻结读出配方。GPU起始空闲，无活跃训练评测；随后串行执行exp95，真实子进程退出0，结束后GPU0%/144MiB、无训练评测进程。exp81控制权重仍为2,073,754,784 bytes，失败目录与所有唯一权重保留。CC-Connect原任务064e9c68实查启用、每小时整点，未创建新任务。

- **已完成，直接观察**：预写[exp95](experiments/2026-10-25-exp95-d3-encoder-seam.md)后，在固定D3 revision的最后语言层23的q_proj/v_proj接入rank4增量，仅26,624可训参数；取exp93已审计训练图四源各一张。零增量原生概率最大误差0；单次AdamW更新CE .206853783→.206807924，梯度范数 .016906656，耗时4.66秒、峰值显存2,021,862,400 bytes。保存诊断补丁并实际重新读取核对张量相等。证据 `runs/exp95-d3-encoder-seam/{report.json,diagnostic-lora.safetensors,supervisor.log,EXIT_CODE}`。工程接口通过；四张训练图loss略降不证明泛化，补丁不是完整可发布checkpoint，未验证注入新runtime后的输出一致性。
- **exp86历史身份核对**：最新记录及原修复脚本表明PIL2首错为masked_scatter长度断言，169占位/64视觉特征；已有seq64完整CUDA评测。不是已确认词表越界，本轮不重复原失败实验，也未再次运行已多轮通过的CPU复现。保留原始失败证据。
- **下一步**：在exp93固定隔离训练池上预写末层低秩恢复pilot，保持exp94验证图片与原生D3对照，先做有限预算再按大小/位置恢复及图表/UI保持门槛决定扩量；不能把本次训练批下降作为采用依据。周日12:00路线决定、18:00训练冻结、23:00回归/独立校准/CPU性能、周一08:00材料与09:00发布缓冲不变。
- **阻断发布，提前报告**：正式ImageJevBench public300与评测接入、五用途显著收益、独立校准、score接口及完整回归/CPU部署验收仍未完成，现有证据不支持达标发布，无法保证周一09:00被接受上榜。本轮未重新联网核验官方入口，不把历史访问证据冒充本轮结论；未发布未达标候选。

**文档与验证范围**：新增exp95预案和诊断脚本；实际语法编译、CUDA四源前向/单步反向、补丁保存读取检查完成。未运行全仓测试、未改环境或许可。

## 2026-10-25 03:05:43 +08:00：exp94冻结读出恢复pilot未通过，停止扩大本配方

**阶段与实际状态**：已实际完成256题冻结读出100步训练和384题固定验证；独立验证进程exit0、384条明细完整、概率有限与归一化离线检查通过。当前仍采用v0.4，v0.5未发布。启动前GPU空闲；exp81原控制权重实查2,073,754,784 bytes，所有唯一权重、失败目录保留。CC-Connect仅原任务064e9c68，每小时整点启用，未新增任务。

- **重要，恢复门槛未通过（直接观察）**：训练CE .282644→.272172，100步梯度有限，保存20/40/60/80/100步读出补丁；固定末步验证：大小45/64→42/64（−4.6875pp，图片bootstrap95%区间[−11.59,+1.59]pp），位置57/64不变，ChartQA111/128不变，UI128/128不变。未达到大小/位置均+3pp的预写规则，停止扩大此冻结读出配方；大小变化不能称显著退化。证据 `runs/exp94-d3-readout-validation/{report.json,validation.details.jsonl,probability-check.json,supervisor.log,EXIT_CODE}` 与原训练目录五份补丁。图表/UI不变不能视为完整等效证明，验证仅128题切片且未做独立校准。
- **已修复的执行故障**：首轮验证224题后ChartQA noul缺criteria导致KeyError，exit1；原目录全部保留。核对原生d3_format.options为false/true，修正后独立YAML加载已有step100补丁只重做验证，没有重复训练。当前入口为exp92脚本的YAML pilot扩展，通用deVision pipeline/加载仍未支持D3，补丁不是完整发布checkpoint。
- **exp86复核**：本轮实际CPU repro_after.py exit0，输入1–49279<49280、image token49190一致、占位64与配置64一致；历史PIL2为169/64长度不匹配。原pad token警告仍在，未证实导致该故障；未原样重跑CUDA失败实验。
- **下一步与期限**：本配方停止；优先检查大小/位置错误的表示与可训encoder seam，预写低秩适配相对原生D3的对照，再决定是否投入单卡训练，避免扩大仅能降训练loss的读出。周日12:00路线决定、18:00冻结训练、23:00回归/独立校准/性能、周一08:00材料及09:00发布缓冲保持。
- **阻断发布**：正式ImageJevBench public300/评测接入、五用途显著收益、score接口、独立校准、完整回归和CPU部署验收仍缺，现有证据不支持达标发布；不能保证周一09:00被接受上榜。本轮未重新联网查官方入口，沿用已记录的访问证据；未上传或发布候选。

**文档与验证**：新增exp94预案、两份YAML，扩展现有探针入口；实际py_compile、CPU缓存训练、CUDA384题验证及概率完整性检查。未运行全仓测试、未重装环境或改变许可。

## 2026-10-25 02:06:19 +08:00：exp93恢复训练数据审计与冻结特征缓存完成

**当前阶段与运行状态**：D3小型图文预训练骨干的恢复pilot准备，采用仍为v0.4，v0.5未发布。实查RTX5070空闲、无训练/评测进程后执行；实际任务exit0，256条缓存完整，结束后GPU0%/144MiB。exp81控制权重仍为2,073,754,784 bytes；原权重及全部失败目录保留。同一CC-Connect Cron 064e9c68启用、proj-codex、每小时整点/new_per_run/50分钟，info last_run=01:06:09，本轮实际会话执行；未新建任务。

- **已完成，直接观察**：按[exp93预案](experiments/2026-10-25-exp93-d3-cache-audit.md)，审计3776张图片，排除既有大小/位置、图表、UI、照片与公开示例验证图片，检查COCO/VG alias及解码RGB哈希。最终大小/位置/ChartQA/UI各64图，共256条，指定验证像素交集0。冻结D3特征与原生system_one四源首题概率误差最大4.56e-8；耗时40.42秒、峰值显存约1.90GiB。实际重载缓存、输入行数与SHA256校验通过。证据 `runs/exp93-d3-cache-audit-64/{audit.json,report.json,features.safetensors,metadata.json,supervisor.log,EXIT_CODE}`。缓存不是新模型checkpoint，未更新权重、未验证泛化。
- **重要，数据预算修正**：初始128图/源审计失败，大小原始非翻转choice仅128图、排除验证后127，GPU模型加载前assert退出1。完整保留 `runs/exp93-d3-cache-audit/`。再次执行前把四源统一缩为64并写明修订，未放宽隔离、未按能力结果更换判据。近重复与外部基座预训练曝光仍未知，不能称全部训练源已全量审计。
- **exp86复核**：本轮CPU实际重跑repro_after.py通过：输入1–49279<49280，image token49190两端一致，图像占位64与配置64一致；PIL2历史根因是169/64不匹配，未重复失败CUDA实验。pad token警告仍在，未确认与该故障有关。
- **下一步与倒排**：接入D3的YAML训练/加载seam，使用本缓存做最低成本冻结读出恢复pilot，先预写固定验证、更新预算及大小/位置恢复、图表/UI保持门槛；缓存只适用于冻结骨干。周日12:00路线决定、18:00训练冻结、23:00完整回归/独立校准/性能、周一08:00材料与09:00发布缓冲不变。当前尚未完成此pilot，不把准备写成训练已执行。
- **阻断发布，提前报告**：最新既有证据仍缺正式ImageJevBench public300/正式评测接入，五用途显著收益、score适配、独立校准、完整回归与CPU部署验收。正式接受另受维护者审核队列制约，当前不能保证周一09:00达标上榜。本轮未重新访问官方网页，未提交或发布无证据候选。

**文档与验证**：新增exp93预写/失败修订/结果及缓存脚本，完成语法编译、CPU像素与身份审计、CUDA256条特征/四源原生一致性、缓存重载与哈希校验；未运行全仓测试、未重装环境、未改许可。

## 2026-10-25 01:05:32 +08:00：exp92恢复训练接口探针通过

**当前阶段与运行状态**：小型预训练D3骨干的恢复训练接入，当前采用仍为v0.4，未发布v0.5。先核对实际RTX5070空闲、无训练评测进程，再执行16张训练图的单步读出接口探针；持久exec会话退出码0、日志完整，结束后实际GPU0%/144MiB且无任务。exp81控制权重 `runs/exp81-chart/train/model.safetensors` 实查2,073,754,784 bytes，全部失败目录保留。CC-Connect同一任务064e9c68启用，每小时整点、new_per_run、50分钟；info实查last_run=00:05:02，无last_error字段，本轮由该调度会话实际执行，未创建新任务。

- **重要，训练接入路径明确（直接观察）**：原生D3 runtime冻结全骨干，image_logits在inference_mode中，不能直接用于backward。按[exp92预案](experiments/2026-10-25-exp92-d3-training-interface.md)，重用原生图像嵌入与encoder，16张训练图与原生system_one的最大概率差1.49e-7；只对readout单步AdamW，梯度范数7.6772，训练批CE .412572→.410114；safetensors导出重载并注入原生runtime，首题最大概率差1.29e-8。工程门槛通过。证据 `runs/exp92-d3-training-interface/{report.json,supervisor.log,EXIT_CODE,inputs.jsonl,diagnostic-readout.safetensors}`。耗时6.14秒、峰值显存1.85GB；零骨干梯度，无泛化/校准结论，诊断readout不是完整发布checkpoint。
- **exp86故障复核（直接观察）**：本轮uv实际重跑repro_after.py通过：token1–49279<49280，image id49190一致，图像占位/视觉长度64一致；历史PIL2为169与64长度不匹配，不是已确认词表越界。保留原失败日志及目录，不原样重跑。pad token警告尚在，未证明它与原故障有关。
- **重要限制**：本探针按路径/image_id排除exp81 protect及exp91图片，仅验证训练接口；未验证全源COCO/VG映射、像素/近重复隔离、全骨干backward、完整导出、CPU及HTTP一致性。正式pilot前补数据身份审计、YAML训练及加载seam，不能把单步训练批loss降低写成恢复旧能力。
- **下一步与预算**：停止继续提示搜索；最低成本恢复对照使用冻结骨干/可训readout，加入真实ChartQA/UI回放，并固定独立大小/位置、图表/UI验证，预写训练预算和退化停止条件后由YAML执行。现有pipeline只支持项目原骨干align/train，D3训练/加载尚未接入，不能直接拿exp81 YAML开跑。周日12:00路线决定、18:00训练冻结、23:00回归/独立校准/性能、周一08:00材料、09:00发布缓冲不变。
- **阻断发布，提前报告**：正式ImageJevBench public300尚未取得，五用途、score接口、独立校准、完整外部回归与CPU性能仍缺。此次复查[官方提交页](https://benchmarkheaven.com/submit)仍为免费排队及代码审核，维护者决定时间，无法保证周一09:00正式被接受上榜。未提交未达标候选，也未将局部UI/图表收益冒充目标完成。

**文档与验证**：新增exp92预案、结果和探针脚本；实际编译检查、CUDA原生前向一致性、有限非零梯度、单步更新、safetensors重载验证完成。未运行全仓测试，未重装环境、变更许可或覆盖旧权重。

## 2026-10-25 00:04:52 +08:00：exp91独立图片复验完成

**当前状态**：D3候选筛选阶段，仍采用v0.4，未发布v0.5。实查GPU空闲后执行exp91，256次CUDA前向完整、逐题与汇总已核对，结束后无评测进程、GPU0%/144MiB。exp81控制权重2,073,754,784 bytes仍在；无新权重，所有失败目录保留。CC-Connect同一任务064e9c68启用，每小时整点/new_per_run/50分钟，未创建新任务。

- **重要，面积语义收益复现（直接观察）**：排除既有data/v05全部JSONL图片，从旧dev固定大小/位置各64张图。大小70.31%→82.81%，+12.5pp，图片bootstrap95%区间[4.69,21.88]pp，NLL .4951→.4444，通过预写门槛。位置82.81%→79.69%，−3.125pp，区间[−9.375,3.125]pp，不采用位置后缀。证据runs/exp91-d3-semantics-independent/report.json及256条明细；方案[exp91](experiments/2026-10-25-exp91-independent-semantics.md)。仅支持面积定义的局部作用，旧开发身份、外部预训练曝光未知，不证明五用途验收。
- **exp86复核**：实际CPU再跑repro_after.py，token范围1–49279<49280，image id49190一致，图像占位/视觉长度64一致。原PIL2失败目录保留，不重复该失败实验。pad token警告尚在，未证实与原错误有关。
- **执行限制**：准备阶段无系统python已改uv；原监督shell未存活，实际评测timeout子进程正常完成。误补启动被STARTED保护挡在模型加载前，污染日志和EXIT_CODE；原评测退出码未知，完整产物与进程退出有直接证据。详细经过保留于exp91记录，后续改持久化监督和独占日志，勿以此EXIT_CODE判断评测结果。
- **下一步**：检查D3原生训练/损失/导出接口，固定真实数据回放与独立验证图片隔离，预写最小恢复pilot；面积后缀可保留，位置能力仍需恢复。周日12:00路线决策、18:00训练冻结、23:00回归/校准/性能、周一08:00材料、09:00期限。
- **阻断发布**：仍缺正式ImageJevBench public300/接入，五用途、score适配、独立校准、完整回归及部署性能未完成。无法据当前局部收益保证周一达标；不提交冒充完成的版本。

**改动与验证**：新增exp91预写/结果、推理复验与复算脚本；编译检查、实际CUDA256前向和5000次图片配对复算完成；未运行全仓测试，未修改环境/许可。

## 2026-10-24 23:05 +08:00：exp90语义诊断完成，大小题部分恢复

**当前阶段与实际运行**：小型原生图文骨干筛选，当前采用仍v0.4，未训练新权重、未发布v0.5。启动前实查GPU0%/144MiB、无训练评测进程；exp90实际CUDA128前向、逐题完整、汇总和STATUS完成，进程退出后GPU0%/144MiB。外层启动器未留下EXIT_CODE，退出码未知；不以状态文件单独证明完成。exp81控制权重实查2,073,754,784 bytes，失败目录均保留。

- **重要，语义差异仅解释部分退化（直接观察）**：生成器`scripts/data/v2_coco_spatial.py:75`使用annotation area，非bbox面积；抽查六图保留`runs/exp90-d3-semantics/error_sheet.jpg`。预写[exp90](experiments/2026-10-24-exp90-d3-semantics.md)后，只附加统一面积/坐标定义，固定原128题、原序、权重与处理器。size70.31%→79.69%，+9.375pp，图片分组95%区间[1.56,18.03]pp；position89.06%不变，区间[-4.55,4.76]pp。证据`runs/exp90-d3-semantics/report.json`及128条明细。建议在独立图片验证，完成条件为收益在预先固定独立集复现。size仍低于exp81的93.75%，位置退化仍在，不改变旧成绩、不宣称恢复验收通过。
- **exp86故障复核（直接观察）**：本轮实际CPU运行repro_after.py通过：词表49280、输入1–49279、image token49190两端一致、占位/视觉特征64。原错误为masked_scatter长度不匹配，修复已存在且有完整CUDA评测；未重复原始失败尝试。pad token警告仍存在，未证明与原故障有关。
- **调度核验**：CC-Connect任务064e9c68启用、proj-codex、每小时整点/new_per_run/50分钟，info实查last_run22:05:46，无last_error字段。未新建或递归创建任务。
- **下一步与预算**：先固定独立旧开发图片，排除本轮所有诊断图片，做面积措辞复现（预计数分钟CUDA）；据此预写恢复训练pilot。周日12:00路线决策、18:00训练冻结、23:00完整回归/独立校准/CPU与CUDA性能，周一08:00材料、09:00发布期限不变。
- **阻断发布**：正式ImageJevBench public300/正式接入未取得，五用途、score适配、独立校准、完整外部回归与部署验收未完成。当前证据不足以支持达标发布；最小外部缺失仍是合法public300归档/manifest或正式评测接入，局部图表/UI成功不能替代。

**文档与验证**：新增exp90预写及结果、诊断与复算脚本；脚本编译、实际CUDA128前向及逐题配对复算通过；未运行全仓测试、未重装环境或改变许可。外层退出码缺失已披露，后续长任务须用持久化监督进程确保wait与退出码落盘。

## 2026-10-24 22:05 +08:00：exp88完成，exp89排除明显候选顺序偏置

**当前阶段与实际状态**：小型图文预训练骨干基线筛选，当前采用仍v0.4，无新训练权重、未正式发布。起始实查RTX5070为0%/144MiB且无训练Python；exp88已完整退出，六份逐题文件与报告齐全。随后单卡执行exp89，128条完整，完成后GPU恢复0%/144MiB且进程退出；未干预其他任务。exp81唯一控制权重实查2,073,754,784 bytes，全部失败目录保留。

- **重要正结果，限路线筛选（直接观察）**：exp88图表1000题/753图85.2%，对exp81 +32.5pp [28.89,36.20]；照片587题/585图61.33%，对v0.4 +14.65pp [9.54,19.76]。真图相对错图收益分别50.8pp/22.83pp且区间下界>0。证据`runs/exp88-d3-panels-local/report.json`及六份明细，共4838前向。外部预训练曝光未知，不能归因为单一架构变量，未经独立校准。
- **重要退化，暂缓扩大训练（直接观察）**：保护总体81.97%对80.05%，但位置64题下降9.375pp [-17.74,-1.54]、大小64题下降23.4375pp [-35.48,-11.67]，触发原预写门禁。不能用总体改善覆盖分项退化。下一步先抽查错误图像、框面积标签和指代，区分语义/标注差异与能力缺口，再决定恢复训练pilot；完成条件为有逐题视觉证据和独立验证集的预写方案，不直接扩量。
- **已完成诊断（直接观察）**：预写[exp89](experiments/2026-10-24-exp89-d3-protect-order.md)，固定上述128题只反转候选。位置原序89.06%/倒序90.63%、答案改变1.56%；大小70.31%/67.19%、改变6.25%，均低于20%触发线。结果不支持明显顺序偏置解释，不选择较好顺序充当收益。证据`runs/exp89-d3-protect-order/report.json`、128条逐题、IDENTITY、日志；复算`scripts/v05/report_exp89.py`。外层启动器退出码未落盘，明确未知；实际逐题128条、两来源各64、完成汇总存在、进程已退出，离线复算成功。无需为补退出码重复评测。
- **exp86复核（直接观察）**：本轮CPU重新执行`repro_after.py`成功：image token49190两端一致，输入范围1–49279<49280，image_seq_len64、占位64。原PIL2日志首错为masked_scatter_size_check的totalElements<=srcSize，既有修复和全量CUDA证据保留；不再按未修复token越界重跑。模型配置另有pad token警告，但本检查未证明它导致原故障。
- **调度与期限**：实查CC-Connect同一任务064e9c68启用、proj-codex、整点每小时/new_per_run/50分钟，last_run21:14:16，无last_error；未创建递归或重复任务。周日12:00路线决策、18:00训练冻结、23:00回归/独立校准/性能，周一08:00材料与09:00发布缓冲继续有效。
- **阻断发布**：正式ImageJevBench core public300与正式接入尚未取得；五用途显著收益、score适配、独立校准、CPU/CUDA性能和完整回归未完成。当前证据不支持达标发布，已提前明确报告；需要合法public300归档/manifest或正式评测接入。局部UI与图表成功不能替代上述条件。

**验证范围**：本轮CPU处理器复现、CUDA128次实际前向、离线完整性与顺序复算、两个脚本py_compile；未运行全仓测试、未重装环境、未更改许可、未上传或发布。

## 2026-10-24 21:13 +08:00：exp87原生决策UI筛选通过，exp88全域辅助面板进行中

**阶段与实际状态**：小型图文预训练骨干基线筛选，当前采用仍v0.4；无新训练checkpoint、未正式发布。先读AGENTS/CLAUDE、目标和最新实验，并核对GPU/进程/日志/权重。起始GPU0%/144MiB，无训练；exp81唯一控制权重2,073,754,784 bytes存在。exp87完整结束并释放GPU后才启动exp88。21:13直接观察exp88实际Python PID376558、timeout PID376554，GPU45%/2317MiB；chart.real逐题已130条（日志第129题累计86.82%），正在持续写出，不能把部分分数当完整门禁。按现有吞吐预计余下约10–20分钟，2400秒内部预算/2450秒硬退出；日志`runs/exp88-d3-panels-local/supervisor.log`，退出码EXIT_CODE，成功后自动离线报告。

- **exp86故障复核通过（直接观察）**：本轮重新执行`runs/exp86-smol-ui-baseline-seq64/repro_after.py`，image token49190一致、输入范围1–49279小于词表49280、占位64与视觉特征64一致；原始错误是masked_scatter长度不足，不能继续称已确认token越界。已有完整512+256题及提示诊断证据，未原样重跑失败实验，全部失败目录保留。
- **重要正结果，限开发UI（直接观察）**：exp87固定D3-lite revision `03560c26f5199c3c04129a2ccb38960c913635e0`，原生D3.system_one概率读出，实际768条/exit0。真图256/256，轮换错图0/256，倒序255/256；语义答案改变率0.390625%。对exp81 +59.765625pp，64四元组5000次bootstrap95%区间[55.078125,64.453125]pp；P1/P2及顺序诊断通过，决定晋级其他面板。证据`runs/exp87-d3-ui-pil/native_report.json`、IDENTITY及逐题文件，复算`scripts/v05/report_exp87.py`。影响：支持转向原生多模态决策起点继续验证；外部预训练是否见过同图未知、像素预算更高，不作单因素归因或正式五用途结论。原始NLL未校准。
- **重要兼容问题已处理**：D3下载过滤初次遗漏jinja/txt，被原manifest拒绝，保留`runs/exp87-d3-ui`。AutoProcessor还强制加载缺失torchvision的视频组件；未安装依赖，新增静态PIL适配，明确不支持视频，CPU实际图像token/网格匹配后CUDA完整768次通过。PIL与torchvision未做数值等价验证，作为适配因素披露。exp88首个目录又在新版Hub全快照local_files_only检查退出（尚未前向），改为exp87已经验证的本地revision路径，保留原D3 manifest校验，不重复无效下载。全部失败目录保留。代码与方案分别为`scripts/v05/d3_{ui_baseline,image_processor,panels}.py`和[exp87](experiments/2026-10-24-exp87-d3-baseline.md)、[exp88](experiments/2026-10-24-exp88-d3-panels.md)。
- **进行中与下一步**：exp88固定图表1000、旧保护832、真实照片587，各真图/不同图片全局轮换，共4838前向。比较exp81同题与照片v0.4，图片分组bootstrap5000/seed88；预写“图表或照片至少+5pp且下界>0、各面板/旧来源无显著≥3pp退化、图表/照片真图收益下界>0”。通过才预写新骨干训练pilot；若失败先分析，不直接扩大训练。独立校准、完整外部与laya对比、CPU/CUDA性能及deVision接口集成均尚待执行；此轮是外部基线，不冒充完整训练轮次。
- **阻断发布**：正式ImageJevBench public300仍未取得，score适配、五用途显著改善、独立校准与性能尚未完成。最小外部缺失仍为合法core public300归档/manifest或正式评测接入，不用8示例/UI替代。继续周日12:00路线决策、18:00训练冻结、23:00回归校准性能、周一08:00材料与09:00前发布缓冲；关键条件不满足则提前报告未达标，不伪称发布达标。

**调度与验证范围**：CC-Connect原任务064e9c68已启用，project=proj-codex、每小时整点/new_per_run/50分钟，实查last_run=20:04:59且无last_error；未创建递归或重复任务。新增脚本py_compile通过、CPU映射预检与真实CUDA完整exp87通过；exp88仅确认已开始并持续推进，尚未得到完整结果。未运行全仓测试、未重装环境、未更改许可或发布权重。

## 2026-10-24 20:04 +08:00：exp86独立提示诊断完成，停止该提示搜索

**阶段与运行状态**：小型图文预训练骨干筛选；无合格v0.5候选，当前采用仍v0.4。实际GPU起止均0%/144MiB、无其他训练；本轮串行192前向全部完成，两个timeout子进程exit0，已释放GPU。零训练，无新checkpoint；exp81控制权重实查存在（2,073,754,784 bytes），所有失败目录保留。

- **故障复核已通过（直接观察）**：重新执行`uv run --no-sync python runs/exp86-smol-ui-baseline-seq64/repro_after.py`，实际token范围1–49279、词表49280、image token49190、占位64与特征64一致。既有512题日志与倒序256题明细均存在，未重复原始失败运行。无环境重装。
- **重要结果与决策**：预写后在exp85/calib全部32题/8组比较空前缀和固定`Answer: `，检查与ui_dev图片路径零交集。原序/倒序/组内错图准确率：空前缀25.00%/40.625%/28.125%；Answer前缀40.625%/40.625%/28.125%。两顺序平均+7.8125pp，组bootstrap95%区间[-4.6875,20.3125]pp，答案改变率均78.125%。未满足“顺序敏感性下降”的预写必要条件，停止该提示搜索，不扩大Smol微调。证据`runs/exp86-prompt-calib-report.json`及两目录192条明细；复算脚本`scripts/v05/report_exp86_prompt.py`。小样本结果不支持显著改善，校准集现已用于提示选择，后续不得仍标纯calibration_fit。
- **调度直接核验**：CC-Connect同一任务064e9c68启用，真实project=proj-codex，每小时整点、new_per_run、50分钟；info实际last_run=2026-10-24T19:42:28+08:00且无last_error。未新建/递归创建任务。
- **阻断与下一步**：ImageJevBench正式public300仍未取得，正式score适配、五用途显著收益、独立校准与CPU/CUDA部署验收未完成。最小外部缺失材料仍为合法core public300归档与manifest；8例及本UI不替代。下一项优先D3-lite原生决策baseline，先固定revision/审阅代码/核对现有环境兼容性，再按30分钟单卡预算同题筛选；本轮未声称已执行D3。继续按周日12:00路线决策、18:00冻结训练、23:00回归校准性能、周一08:00材料与发布缓冲倒排。若关键门槛或正式接入仍失败，不以期限代替证据正式发布。

**本轮改动与验证**：评测脚本支持显式输入文件、组内轮换及assistant前缀，身份记录前缀/输入路径，汇总记录输入哈希；增加诊断复算脚本。两个脚本py_compile通过，真实CUDA192前向与离线复算通过；未运行全仓测试。方案与结果追加到[exp86记录](experiments/2026-10-24-exp86-smol-baseline.md)。

## 2026-10-24 19:41 +08:00：exp86处理器修复、完整UI基线与倒序诊断

**当前阶段**：小型图文预训练骨干筛选，尚无v0.5合格候选；当前采用仍为v0.4。实查GPU/进程未发现活跃训练后串行执行，未干预他人任务。CC-Connect Cron `064e9c68` 已启用，实际project为`proj-codex`，整点每小时、new_per_run、50分钟超时；本轮为19:37补触发的实际执行，旧last_error在本轮返回前仍可能显示busy，不将其冒充已成功完成调度。未创建递归任务。

- **阻断故障已修复（直接观察）**：PIL2 `supervisor.log`首错为masked_scatter源长度不足。CPU检查证实tokenizer与模型image token均49190，输入范围1–49279在词表49280以内；遗漏原processor_config的image_seq_len64导致当前处理器默认169。显式读取64并增加前向前校验后，真实CUDA完整512题成功。失败/修复复现保存在`runs/exp86-smol-ui-baseline-seq64/repro_{before,after}.{py,log}`，三个原始失败目录保留。未重装依赖、未改缓存权重。验证标准为CPU映射/长度检查及原始协议全量前向，均已通过。
- **重要结果（直接观察）**：`runs/exp86-smol-ui-baseline-seq64/report.json`：真图98/256=38.28%，错图55/256=21.48%；对exp81 40.23%差−1.95pp，64四元组5000次bootstrap区间[−9.77,+5.86]pp；真图减错图+16.80pp [8.59,25.00]。P1未过、P2通过，不能据此换骨干或扩大训练。该模型原始NLL1.626，未校准，不作跨模型概率质量结论。零训练无新checkpoint；exp81唯一控制权重`runs/exp81-chart/train/model.safetensors`保留。
- **已完成（19:42实查退出及256条明细）**：独立目录`runs/exp86-smol-ui-reversed-seq64`，launcher timeout PID371473，600秒硬退出，日志supervisor.log、逐题ui.reversed.details.jsonl。只反转同题候选顺序，检查语义答案敏感性；若变化率>20%或提升≥5pp，先验证读出协议；否则转下一预训练候选筛选。倒序92/256=35.94%，较原序−2.34pp [−10.55,+5.47]，语义答案改变率87.5%，触发预写读出敏感规则；暂不训练，下一轮先比较独立开发上的提示/答案位置协议，不能选择更好顺序充当收益。GPU任务已退出，未占卡等待。实验预写与完整证据见[exp86](experiments/2026-10-24-exp86-smol-baseline.md)。脚本py_compile通过；未运行全仓测试或新训练。
- **发布阻塞**：ImageJevBench core public300仍未取得，正式score题型适配、五用途显著提升、校准及部署性能均未完成。已有8示例不能替代正式验收；因此当前不能承诺2026-10-26 09:00前达标上榜。现有数据筛选后exp82v2仅564训练/84校准/361开发题，exp85四元组可用组136/74，不能把重复呈现当独立数据扩量。

**倒排下一步**：周六优先完成候选读出有效性与真实数据覆盖；周日12:00前固定可训练路线与完整验收输入，18:00前结束训练并冻结候选，18:00–23:00留回归/独立校准/CPU与CUDA性能，周一00:00–08:00留模型卡/本地发布验证和正式上传缓冲。此为预算安排而非完成承诺；若正式题集或关键指标仍阻塞，明确报告未达标，不发布成达标v0.5。发布授权已收到，只有证据支持才执行正式发布。

| 轮次 | 日期 | 配置 | 相对上一轮改了什么 | 状态 |
|---|---|---|---|---|
| v1-release-0.1 | 2026-10-01 | `configs/v1-release-0.1.yaml` | 首个完整流程（对齐 → 2a → 2b），发布为 0.1 | 被 v2 超过 |
| v2-align-lora | 2026-10-02 | `configs/v2-align-lora.yaml` | 对齐阶段 ModernBERT 也加 LoRA（r=16，lr 1e-4），其余不变 | 被 v3 超过 |
| v3-data2 | 2026-10-03 | `configs/v3-data2.yaml` | 从 v2 的对齐结果起，阶段 2 换成 data-v2 训练混合（134,556 题，一个阶段） | 被 v5 取代 |
| v4-relation-tb | 2026-10-03 | `configs/v4-relation-tb.yaml` | 从 v3 接着训 2 个 epoch：上下相对位置题 3,186 道（COCO + 场景图）+ 3 倍回放 | 上下题修好，GQA 退步；未取代 v3 |
| v5-data3 | 2026-10-04 | `configs/v5-data3.yaml` | 从 v3 接着训一遍：新增常识推理与教材插图（A-OKVQA / ScienceQA / AI2D / TQA）+ 上下相对位置题共 32,164 道，回放 30,000 道 | 被 v6 取代（曾为当前最佳：负责人决定取代 v3） |
| v6-flip-lr | 2026-10-04 | `configs/v6-flip-lr.yaml` | 从 v5 接着训一遍：左右镜像成对题 12,000 对 + 按能力保底的回放 24,000 道 | 被 v7 取代（曾为当前最佳：负责人决定取代 v5） |
| v7-sqa | 2026-10-04 | `configs/v7-sqa.yaml` | 从 v6 接着训一遍：data-v4 的 ScienceQA 4,948 道（示意图按像素比对，找回 2,095 道）+ 回放 17,000 + 左右镜像 3,000 对 | 被 v8 取代（曾为当前最佳：负责人决定取代 v6） |
| v8-vsr-v7w | 2026-10-05 | `configs/v8-vsr-v7w.yaml` | 从 v7 接着训一遍：data-v5 的 VSR 6,411 + Visual7W telling 7,500 + v7 的 ScienceQA + 回放 18,500（计数 +1,000、GQA +500）+ 左右镜像 3,000 对 | 被 v9b 取代（曾为当前最佳：负责人决定取代 v7） |
| v9b-lr-replay | 2026-10-05 | `configs/v9b-lr-replay.yaml` | 从 v8 接着训 2 遍：真实照片左右题 2,488 道（R 的原题，原图 + 镜像，几乎全是位置题）+ 回放 2,500（含 VSR / ScienceQA / Visual7W） | 被 v11 取代（曾为当前最佳：负责人决定取代 v8） |
| v10-count | 2026-10-06 | `configs/v10-count.yaml`（全量评测 `configs/v10-count-eval.yaml`） | 从 v9b 接着训 2 遍：v9b 的左右原题 2,488 + 回放 2,488（计数占一半）；示意图路线在阶段 A 停止后的备选 | 未取代 v9b（计数在测试集上没有补回） |
| v11-v9b-recovery | 2026-10-06 | `configs/v11-v9b-recovery.yaml`（筛选 `v11-recovery-screen.yaml`，全量评测 `v11-v9b-recovery-eval.yaml`） | 从 v9b 接着训 1 遍：9,988 道不同的题 = 计数 3,000 + 其他回放 4,500 + v9b 的左右原题 2,488；学习率同 v10 | 被 v12 取代（曾为当前最佳：负责人决定作为取舍取代 v9b） |
| v12-ext | 2026-10-07 | `configs/v12-ext.yaml`（全量评测 `configs/v12-ext-eval.yaml`） | 从 v11 接着训一遍：扩展数据包 353,830 道（13 个新来源 315,342 + 回放 38,488，`data/ext/MIX.json`），架构不变 | 被 r15 取代（曾由负责人作为取舍采用；发布为 v0.2） |
| r14-count | 2026-10-13 | `configs/r14-count.yaml` | 从 v12 接着训一遍：配平 0–10 计数 12,000 + PixMo 存在性 / 窄候选计数 12,000 + 回放 57,876（71%，扩展每源 2,400），81,876 题、10,235 步，结构不变 | 未取代 v12（CountBench +14.5，但镜像关系 −1.4 已证明退化、4 个集合未证明守住、What'sUp 概率警戒；待负责人决定） |
| r14b-mirror | 2026-10-13 | `configs/r14b-mirror.yaml` | r14 配方中扩展回放每源 2,400 → 2,100，换成 v6 左右镜像对 2,000 + VSR 1,000（其余逐题相同） | 第 2,000 步止损（原格式计数面板 −3.3 [−6.3, −0.7]），未训完 |
| r15-vision | 2026-10-14 | `configs/r15-vision.yaml` | r14b 数据 + 解冻 SigLIP 最上面 4 层（5e-6），种子 1，从 exp19 第 2,000 步接着训满 10,235 步 | 被 r21 取代（曾发布为 v0.3；负责人接受取舍；原始 candidate=false 保留） |
| v04-k（草稿） | 2026-10-19 | `configs/v04-full-eval.yaml`（冻结 `runs/v04-k/eval/IDENTITY.json` c14863e2…） | 读数组合：PROTECT-K 决策权重（exp30–protect-k 链）+ EXP71 读数组件 + 确定性 K 规则 + 分题型 / 选项数 / 读数域温度 | 全评文件完整（两份执行证据待补），**未取得发布资格**（dev16 保留门槛未达，RELEASE-GAP-01）；待定 |
| r13-binding | 2026-10-11 | `configs/r13-binding.yaml`（分支 `r13-binding`；全量评测 `configs/r13-binding-eval.yaml`） | 从 v12 接着训一遍：第 13 轮绑定数据 207,109 道（反事实合成 40,040 照片背景 75%、GQA 绑定、VisMin、CoGenT A、PixMo 两类计数、CounterCurate、TallyQA 复杂、VAW + 回放 40,000），加问题条件视觉 token（S2，零初始化） | 未取代 v12（主要条件通过，POPE 已证明退化、概率质量警戒触发；负责人 2026-10-11 决定不采用、不作下一轮起点） |
| r16-n1 | 2026-10-21 | `configs/r16-n1.yaml`（基线 `configs/v03-baseline.yaml`） | 本机 CUDA（RTX 5070）。从 v0.3 接着训一遍新构造的 data-n1：VQAv2 train 全量 313,595（标注者答案作软 gold）+ GQA balanced train 全部 21 个分片 299,398（含开放式查询题）+ v0.3 其他能力回放 125,701；bf16；解冻 SigLIP 顶部 4 层；92,337 步 | 准确率 11 个集合显著提升、无显著退步；POPE / 存在 / 大小的 NLL 变差（过度自信），待修；待负责人决定 |
| r17-exist | 2026-10-21 | `configs/r17-exist.yaml` | 从 r16 接着训一遍：不限物体大小的 COCO 存在题 62,846（`n1_exist.py`，yes = no，四分之一的 yes 是 < 1% 画面的物体）+ r16 混合随机回放 120,000 | 对 v0.3 13 个集合显著提升；POPE 准确率 +1.4 显著、NLL 0.97 → 0.68（仍差于 v0.3 的 0.37）；ScienceQA −1.8 [−3.2, −0.3] 显著退步；未取代 |
| r18-smooth | 2026-10-21 | `configs/r18-smooth.yaml` | 从 v0.3 一遍：data-n1 + r17 的存在题，one-hot gold 做 label smoothing 0.1，ScienceQA / AI2D / TQA 回放 ×2 | 训练中 |
| r19-n2 | 2026-10-21 | `configs/r19-n2.yaml` | 从 v0.3 一遍 data-n2（742k：VQAv2 / GQA 各降到约 20 万，Visual7W 全量、场景图空间描述选择、COCO / TallyQA 计数 0–10、CLEVR、IconQA、合成磁铁 / 溶液示意图，答案能靠文字猜的选择题已去掉）；第一次运行带 MapQA，第 22,000 步止损（`runs/archive/r19-n2-stop-mapqa`） | 验收规则 1、2 通过，3、4 未通过；数据 bug 丢了 46% 的 ScienceQA |
| r20-science | 2026-10-21 | `configs/r20-science.yaml` | 从 r19 短程一遍（123k）：修正后的数据里全部 ScienceQA / AI2D / TQA ×3、IconQA、合成磁铁题、4 万回放 | **v0.4 候选**：验收规则 1–3 通过，规则 4（校准）卡在 POPE / size / MMBench / MME；待负责人决定插值或 r21 |
| r21-soft | 2026-10-22 | `configs/r21-soft.yaml` | 从 r20 短程一遍（104k）：存在题 4 万 + 大小题 4,331 的 gold 软化成 0.9 / 0.1，6 万回放；温度在均衡开发集上拟合 | **当前采用，发布为 v0.4**（负责人 2026-10-22 接受取舍；规则 4 未通过） |

## exp83-ui-tok256（2026-10-24）

从exp81折叠到256视觉token，在exp82相同数据/seed/1,024更新上复测真实UI。终点67.97%，对64token exp82的69.92%差−1.95pp，区间跨0；真图比错图−0.39pp，旧保护NLL超门槛且coco-position/gqa-v2退化。**拒绝采用**，自动进入exp84。短程结果不证明所有256token配方无效；详细局限和区间见2026-10-24-ui-cycle-results.md与runs/exp83/report.json。

## exp82-ui（2026-10-24）

从exp81进行真实ScreenQA派生四选一短程迁移，512道UI题各4次，混图表和旧回放，共8,192呈现/1,024更新。开发256题58.6%→69.9%，但错图67.6%，P2未过；旧保护T=1 NLL0.4951→0.5917，ai2d触发退化，P3未过。**不采用、不扩量**，保留失败权重作证据。详情2026-10-24-exp82-ui-plan.md与runs/exp82/report.json。

## exp81-chart（2026-10-23）

从v0.4以CE扩量24k真实ChartQA派生四选一＋24k旧回放，6,000更新。固定开发1,000题31.7%→52.7%，对pilot45.7% +7.0pp [3.8,10.1]；真图减错图8.2pp [4.9,11.4]，旧保护未报警，照片46.7%点估计持平（不等效证明）。P1/P2/P3/P4通过，**作为图表方向后续起点**；不是全域候选/正式发布。报告runs/exp81/report.json及exp81方案。

## exp80-chart-r / exp80-chart-c（2026-10-23）

同8k真实图表＋8k回放，各2,000更新，从v0.4对照RLCD+CE和CE。开发1,000题31.7%→43.4%/45.7%，两臂均过P1/P2/P3；C-R +2.3pp区间跨0，按预写规则优先简单CE，不宣称目标函数稳定胜负。采用CE扩量验证，未发布。完整结果和汇总脚本修复留证据见exp80方案及runs/exp80/report.json。

## r21-soft（r20 + 存在 / 大小题标签软化；准确率最好，校准换了几处地方没过）

从 r20 一遍 `data/n2/soft_mix.jsonl`（`scripts/data/n2/soft_mix.py`：存在题 40,000 + 大小题 4,331 的 gold 由 0/1 改成 0.9 / 0.1，另 60,000 原样回放），13,042 步；温度在 `data/v5/dev_mix.jsonl`（各来源均衡）上拟合：choice 2.36、noul 1.09、choice:2 3.56。判定 `runs/r21-soft/eval/compare/accept_v04.md`。

- **对 v0.3**：22 / 23 个判定集合点估计提升，无显著退步。新增显著的：POPE +1.7 [+0.9, +2.6]、CoGenT B +4.0；ScienceQA +3.1、VSR +7.0、V7W +7.2、What'sUp +18.0、CountBench +14.5、MMBench +2.3；A-OKVQA +2.1、size +1.3（区间含 0）；MME −0.6（区间含 0）。规则 1–3 通过。
- **规则 4 未通过**：POPE 修好（NLL −0.009，与 v0.3 持平）；但 size +0.156、MME +0.037、存在 +0.026 超门槛。原因不同：size 和存在题现在是**欠自信**（size 平均置信 0.64、准确率 0.87；存在 0.88 对 0.955）——软化后这两类输出封顶在 0.9 附近，choice:2 温度又在均衡开发集上拟合得偏高；MME 是过度自信（0.76 对 0.65）。一个题型 / 选项数一个温度，不能同时适合这些题。
- **对 laya-vision**：VQAv2 是非 +3.9 [+2.4, +5.3]、A-OKVQA +3.5 [+0.3, +6.6]、ScienceQA −0.3 [−2.2, +1.7]；POPE 三档 89.9 / 87.3 / 82.4。

## v0.4 插值步骤（v0.3 ↔ r20；`docs/experiments/2026-10-22-v04-interp-plan.md`）

α ∈ {1.0, 0.8, 0.6, 0.4}，温度都在 `data/v5/dev_mix.jsonl` 上重拟合；按事先写下的规则（存在题开发集 NLL ≤ v0.3 + 0.02 的最大 α）选中 α = 1.0（即 r20 权重 + 重拟合温度：choice 3.79、noul 4.03），全量评测一次（`runs/v04-interp/a1.0/compare/accept_v04.md`）。准确率同 r20；校准比 r20 好（MMBench、MME 恢复），规则 4 只剩 POPE（NLL 0.372 → 0.454）和 size（0.350 → 0.440）未过，两者都是过度自信。这条选择规则用的存在题开发集（COCO train 图）没有反映 POPE 的问题，规则偏松。

## r20-science（r19 + 科学题短程补训；规则 1–3 通过，校准未过）

方案 `docs/experiments/2026-10-21-v04-plan.md`，冻结规则 `docs/experiments/2026-10-21-v04-eval-freeze.md`，判定 `runs/r20-science/eval/compare/accept_v04.md`（`scripts/v04/accept.py`）。从 r19 一遍 `data/n2/science_mix.jsonl`（`scripts/data/n2/science_mix.py`，123,084 题），15,386 步；拟合温度 choice 2.91、noul 2.78。

- **对 v0.3（26 个集合，同题配对）**：21 / 23 个判定集合点估计提升，无显著退步。镜像关系 +21.1、What'sUp +19.0、关系 +16.7、CountBench +14.1、镜像位置 +10.5、V7W +6.7、GQA +6.0、VSR +5.4、位置 +5.1、COCO 计数 +4.1、VQAv2 是非 +3.8、ScienceQA **+2.8 [+1.3, +4.4]**、MMBench +2.5、CoGenT B +2.3、存在 +2.1；A-OKVQA +1.7、POPE +0.9、CV-Bench +0.5、NaturalBench +0.2（区间含 0）；size −0.7、MME −0.7（区间含 0）。dev_sqa（ScienceQA test 划分）+4.0。
- **验收**：规则 1（无显著退步）、2（≥ 15 个提升）、3（弱项 4 / 5：ScienceQA、VSR、计数、V7W；A-OKVQA 未达）通过；**规则 4 未通过**：NLL 升高 > 0.02 且下界 > 0 的有 POPE +0.23、size +0.09、MMBench +0.05、MME +0.03。
- **对 r19**：ScienceQA +4.5、POPE NLL −0.17；VSR −1.7（区间含 0）。
- **校准分析**（逐题明细离线重算）：是非题温度在各来源均衡的开发集上重拟合（2.78 → 3.75），POPE NLL 0.60 → 0.48，仍高于 v0.3 的 0.37；POPE 单独最优温度约 7。只调温度不能通过规则 4。原因与 r16 / r17 相同：大量 0/1 硬标签存在题使存在类输出过尖，一个是非题温度不能同时适合 VQAv2 类和存在类。
- **对 laya-vision**（`eval/compare/laya_vision.md`）：VQAv2 是非 +3.7 [+2.2, +5.2]、A-OKVQA +3.1 [+0.3, +6.1]、ScienceQA −0.7 [−2.7, +1.2]（v0.3 −3.5；首次区间含 0）；POPE 三档 89.3 / 86.5 / 81.2，都高于对方公开分。
- **结论**：准确率是这条线上最全面的一次提升；校准未过，按规则不发布，交负责人在「与 v0.3 权重插值」和「r21：存在题标签软化短程续训」之间决定。

## r19-n2（v0.3 + data-n2；准确率全面提升，ScienceQA 受数据 bug 拖累）

从 v0.3 一遍 `data/n2/train_mix.jsonl`（`scripts/data/n2/`，742,238 题），92,780 步，r16 的设置；拟合温度 choice 1.62、noul 1.78。

- **第一次运行止损**：混合里有 MapQA 15,000 题；第 22,000 步快照（`scripts/v04/snapshot.py`）在 dev_sqa 上十三殖民地 0.93 → 0.34、大洋洲国家 0.94 → 0.59，按方案止损，去掉 MapQA 重训（`runs/archive/r19-n2-stop-mapqa`）。
- **数据 bug**：留出规则对示意图也用 dHash，同一底图只是高亮区域不同的地图哈希相同；新加的 ScienceQA test 开发集和 MMBench 因此把 4,948 道 ScienceQA 训练题中的 2,291 道误删（州首府、殖民地、大洋大洲、磁铁）。之后改为示意图只按像素完全相同留出（`scripts/data/n2/held.py`），r20 用修正后的数据。
- **对 v0.3**：What'sUp +18.3、CountBench +15.3、镜像关系 +21.4、VSR +7.1、COCO 计数 +4.3、CoGenT B +3.5、MMBench +1.7 等，21 / 23 提升；ScienceQA −1.7 [−3.3, 0.0]；POPE NLL +0.40、存在 +0.08、ScienceQA +0.06。规则 1、2 通过，3、4 未通过。
- **对 laya-vision**：ScienceQA −5.2，其余与 r20 相近。

## r17-exist（r16 + 全尺寸存在题；POPE 准确率修好、校准只修好一半，ScienceQA 显著退步；未取代）

从 r16 接着训一遍 183,000 题（存在题 62,846 + r16 混合随机 120,000），r16 的设置。拟合温度 choice 1.81、noul 2.15。逐题配对在 `runs/r17-exist/eval/compare/`（`vs_v03.*`、`vs_r16.*`）。

- **对 v0.3：13 个集合显著提升**——relation_flip +22.4、relation +17.4、position_flip +12.0、GQA +8.2、position +6.0、VQAv2 是非 +5.9、VSR +4.8、VQAv2 选择 +4.2、V7W +4.2、lv VQAv2 +3.5、存在 +1.7、POPE +1.4 [+0.6, +2.3]、dev_mix +3.7；A-OKVQA +1.8、size +0.8（含 0）。**ScienceQA −1.8 [−3.2, −0.3]，显著退步**（r16 时 −1.0 含 0）。
- **对 r16**：POPE +2.2 [+1.4, +3.0]，其余都含 0（GQA +1.0、VSR −1.3、ScienceQA −0.8）。
- **校准**：POPE NLL 0.967 → 0.678（v0.3 0.372），答错的 1,237 题里 605 题 p > 0.99（v0.3 为 0）。离线把 POPE 的温度从 2.15 放大到约 6.5，NLL 降到 0.361（低于 v0.3），说明排序没问题、只是输出尺度过尖：0/1 硬标签的是非题（GQA、存在）把 logit 推到极端，一个题型温度对 VQAv2 类（约 2）和存在类（约 6.5）不能同时合适。size NLL 0.35 → 0.49 也是同类问题。
- **结论**：不取代。修正放进 r18：从 v0.3 一遍，one-hot gold 做 label smoothing 0.1，ScienceQA / AI2D / TQA 回放 ×2。

## r16-n1（v0.3 + data-n1；准确率全面提升，存在类校准变差；待定）

第一次在本机 CUDA 上训练（负责人 2026-10-20：不按原实验重建，自己构造新训练集，突破 v0.3）。从 v0.3（HF `lukbit/devision` af353dc…，`runs/v03/model`）出发，一遍 data-n1（`scripts/data/n1_vqav2.py`、`n1_gqa.py`、`n1_mix.py`，方案 `docs/experiments/2026-10-20-r16-cuda-plan.md`）：738,694 题，bf16（EXP77），2e-5 / 1e-5 / 5e-5，解冻 SigLIP 顶部 4 层（5e-6），92,337 步约 4.5 小时（含监测）。拟合温度 choice 1.64、noul 2.20（choice:2 3.73）。

所有比较都是本机数字：v0.3 在同一批重建的集合上评测（`runs/v03/eval/`），逐题配对（`runs/r16-n1/eval/compare/vs_v03.*.txt`，按图片重采样 95% 区间）。重建的 `test_*` 与旧机器的同名集合题目不完全相同，旧封存数字不能直接相减（laya-vision 四个集合按对方题目重建，v0.3 的数字与发布时一致）。

- **显著提升（区间整体 > 0）**：relation_flip +22.8 [+20.5, +25.1]、relation +16.9 [+15.1, +18.8]、position_flip +10.9 [+8.4, +13.8]、GQA +7.2 [+4.9, +9.6]、VSR +6.1 [+3.1, +9.2]、position +5.5 [+3.8, +7.3]、VQAv2 是非 +4.9 [+2.5, +7.4]、VQAv2 选择 +4.3 [+2.7, +5.9]、V7W +4.2 [+1.9, +6.5]、bench_lv_vqav2_yesno +4.1 [+2.9, +5.3]、存在 +1.3 [+0.1, +2.6]。这些集合的 NLL 也都下降（存在除外）。
- **区间含 0**：A-OKVQA +2.1 [−0.2, +4.3]、size +0.2、ScienceQA −1.0 [−2.4, +0.5]、POPE −0.8 [−1.7, +0.1]。
- **校准退步**：POPE NLL 0.372 → 0.967、ECE 0.039 → 0.134；存在 NLL 0.170 → 0.245；size 0.350 → 0.437。POPE 答错的题 67% 置信度 > 0.99（v0.3 答错时中位数 0.82），主要是漏报（召回 0.758）：809 道高置信漏报里 648 道 v0.3 也答错（远处的人、瓶子、勺子、遥控器等小物体）。推断原因：COCO 存在题（≥0.5% 面积）和 GQA（≥1.5%）只对足够大的物体出 yes 题，加上大量 0/1 硬标签，模型对小物体很自信地答"没有"。
- **对 laya-vision**（`runs/r16-n1/eval/compare/laya_vision.md`）：VQAv2 是非 +4.8 [+3.3, +6.2]（v0.3 +0.8）、A-OKVQA +3.5 [+0.5, +6.9]（v0.3 +1.4，区间含 0）、ScienceQA −4.5 [−6.5, −2.4]（v0.3 −3.5）；必须看图的自然科学 44.9% 对 70.0%。POPE 三档 86.7 / 84.5 / 80.9，都高于对方公开的 83.6 / 81.9 / 77.7。
- **用途标注**：`bench_lv_vqav2_yesno` 判为 monitoring：r16 的温度拟合集（n1 dev_mix 里的 data-v5 dev 部分）与它共享 38 张 val2014 图；与训练数据没有共享图片或题目（逐个核对为 0）。
- **结论**：准确率全面超过 v0.3，是这条线上最大的一次提升（关系、镜像、GQA）；但存在类问题的过度自信是发布前必须修的。下一步（r17）：加入覆盖各种大小物体的存在题（含小物体，yes = no），从 r16 短程续训，再看 POPE 校准；ScienceQA 看图题仍是长期弱项。

## v04-k（读数组合；全评文件完整，两份执行证据待补，未取得发布资格；草稿）

不是一次新训练，而是一个组合：PROTECT-K 的决策权重（从 r15 经 exp30–protect-k 链得到）+ EXP71 的读数组件（selector / 字符头 / corrector）+ 确定性 K 规则（温度题 K = 质量 × 速度²）+ 普通校准温度和两个读数域温度。冻结身份 c14863e2…；全评 2026-10-19 13:33–22:51，94 份候选输出和 v0.3 新增 6 项都已封存、check 0 问题（V04-K-FULL-ACCEPT-01）。**勘误（2026-10-20）**：check 0 只说明文件完整、哈希一致，不说明执行没有出错；test_gqa.mismatched 和 test_bind_gqa（真图）运行时日志里有 MPS GPU 错误，执行证据待补，见 `docs/experiments/2026-10-20-v04-gpu-evidence-errata.md`。统计来自 `runs/v04-k/posteval/`（只读已封存明细，V04-K-REPORT-RUN-01）。

- **主交付（exp21 transfer：ScienceQA 温度 41 + 磁力 120 道开发题，monitoring）**：准确率对 v0.3 +29.4 [+21.1, +37.7]，看图收益 +32.7 [+23.7, +41.2]；温度 41 题 +53.7（41/41），磁力 120 题 +15.0 [+5, +25]。这两组题在开发中反复使用，不是盲测；LIVE-K 筛查标签因报告里没有权重身份记为 unverifiable，原 candidate_eligible=false 保留。
- **dev16 交付要求（原规则）：未达到**。当前分支从未具备这项能力（谱系见 `docs/experiments/2026-10-19-panel-learn-plan.md` 第 1 节：EXP26 分支证明可学，当前初始化链没有继承它），不是遗忘。读数 0.3675 [0.334, 0.401]、比较 0.415 [0.385, 0.445]，都判为 fail（通过要求点估计 ≥ 0.80 且下界 ≥ 0.75）；擦除对照有效（0.303 / 0.430）。dev16 题不在读数组合覆盖范围内（routed 0 / read 0），走普通路径。
- **七个保护面板：0 项成本**（成本 = ≤ −3 点且上界 < 0）。count_narrow −2.7 [−6.0, +0.7]、flip −1.7 [−3.3, 0.0]、vsr −1.8、ext 宏平均 −1.0，其余持平。
- **29 集配对（对 v0.3）**：区间整体在 0 以下的有 test_bind_gqa −2.2（heldout；执行证据待补，暂不引用，见上方勘误）、test_position −2.0、test_size −1.6、test_vqa_choice −1.5、dev_mix −1.3；整体在 0 以上的有 What'sUp +4.9 [+2.3, +7.4]（heldout）；CountBench +3.5 [−0.2, +7.3]，区间含 0，证据不足。这些是逐项区间，没有做多重比较校正。多个配错图对照下降（如 test_count_fresh 配错图 −5.3）。
- **冻结确认项**：property 75 题，候选 85.3%，对 v0.3 +4.0 [−4, +12]，看图收益 42.7%（v0.3 45.3%）。ChartQA 80 题，候选 31.3%，对 v0.3 +6.3 [−5, +17.5]，**看图收益 −5.0 [−12.5, +2.5]**（v0.3 0），低于预写的最大值基线 49/80；倒序改变率 37.5%。没有图表 / OCR 改善的证据。
- **合成确认**（新画法温度 / 磁力 120 张图）：通过预写门槛，比较题对 v0.3 +13.3 [+4.2, +23.3]；但比较题只答对 46.7%，原始读数温度 1/240。
- **校准**：校准集 NLL 0.873 → 0.747、ECE 0.117 → 0.016（样本内）；ScienceQA NLL 0.726 → 0.487、ECE 0.110 → 0.035（monitoring）。准确率不变。
- **对 laya-vision**：VQAv2 是非 +0.3、A-OKVQA +1.7（区间都含 0），ScienceQA −3.4 [−5.4, −1.4]（v0.3 −3.5）；必须看图的自然科学 59.8% 对 70.0%，v0.3 是 49.5%（+10.3），但 ScienceQA 整体只 +0.1，其余切片抵消。POPE 三档 87.9 / 86.2 / 79.7，都高于对方公开分。
- **结论（草稿）**：目标图形的开发收益大，七个面板没有触发原代价规则，同时有几项小幅退步（逐项区间在 0 以下）；按原规则 dev16 交付要求未达到，所以没有发布资格。下一步按评审裁决定一个最小的能力修复路线，不直接重训或改门槛。

## r15-vision（训满，未通过原规则；负责人取舍采用，发布为 v0.3）

r14b 数据（r14 + v6 镜像对 + VSR）加解冻 SigLIP 最上面 4 层（5e-6），种子 1，从 exp19 第 2,000 步接着训满 10,235 步。止损判定 6 次都没有触发。详见 `docs/experiments/2026-10-14-r15-vision.md`。

- **对 v12**：
  - CountBench **+16.7 [+12.8, +21.0]**；镜像关系 **+2.9 [+1.0, +4.8]**；关系 +4.4；ScienceQA +2.3；
  - 没有概率警戒（What'sUp NLL +0.019）；
  - 未守住：VSR −1.8 [−3.9, +0.3]、A-OKVQA −1.3 [−2.9, +0.3]、VQAv2 是非 −0.6 [−2.2, +1.1]、镜像位置 −0.3 [−2.2, +1.4]，区间都含 0。
- **相对上一轮 r14**：镜像关系 +4.3 [+2.4, +6.2]、关系 +4.4、ScienceQA +1.3；CountBench +2.2，区间含 0。
- **与 laya-vision**：ScienceQA 差距 −5.8 → −3.5；必须看图的自然科学题 49.5% 对 70.0%；POPE 三档仍高于对方。
- **代价**（评审 R15-R2，事后诊断，不新增门槛）：
  - VSR 正类召回 72.6% → 67.3%，−5.3 [−8.0, −2.6]；
  - A-OKVQA 净少答对 15 题，NLL / ECE 变差；
  - CountBench 选项倒序后答案改变的比例 16.9% → 39.5%，但倒序时准确率仍 +17.9；
  - What'sUp NLL +0.019，未到 0.02 的阈值；
  - 开发面板存在性召回 −2.2。
- **限制**：不能把这些变化单独归因于解冻（没有训满、视觉冻结的对照）；测试集是适应性使用。决定表述见实验文档的"交负责人的决定表述"。

## r14b-mirror（第 2,000 步止损，未训完）

负责人看完 r14 的结果后说"先试试吧"。r14 的回放里几乎没有左右类题：VSR 只有 53 题，镜像对一题都没有。所以本轮把扩展回放减掉 3,001 题，换成 v6 左右镜像对 2,000 题和 VSR 1,000 题，其余块与 r14 逐题相同（`scripts/r14b/build.py`）。

- **结果**：第 2,000 步首次监测，原格式计数面板比 v12 低 −3.3 [−6.3, −0.7]，按冻结的规则止损。
- **同期对比**：同一步 r14 是 −0.7 [−4.7, +3.7]。新增的左右镜像对、VSR 面板是 −0.3 / −1.4，区间都含 0，没有回升。
- **直接比较**：r14b 对 r14 同一步、同 300 题为 −2.67 [−6.67, +1.00]，两种配方在这一步的差别既未证明存在，也未证明等效。
- **解读**：数据替换（多镜像、多 VSR、少扩展）和它带来的批次顺序变化混在一起，又没有换种子的重复，分不清原因。评审建议当前选 D（保持 v12），R14-R1 第 1 轮。

止损规则的统计功效和下一步方向交给负责人和评审，见 `docs/experiments/2026-10-13-r14b-mirror.md`。

**后续 exp16**（同种子配对短程，`docs/experiments/2026-10-13-exp16-seedpair.md`）：
- 两个配方用种子 1 各训到第 2,000 步，原格式计数 r14b − r14 两种子合并为 −1.5 [−3.5, +0.7]，按训练前写的读法是未检出，不能把止损归因于数据替换。
- 同一配方换成种子 1 不止损（+0.0），说明第 2,000 步的单次止损对训练轨迹敏感。
- 镜像对和 VSR 未检出早期改善。

## r14-count（验收规则在训练前写下；按规则未取代 v12）

**背景**：v12 重训（r13b）两组都因原格式计数退步止损；之后用 exp14 / exp15 的 2,000 步快速检查逐块验证（`docs/experiments/2026-10-12-exp15-steps.md`）：配平 0–10 计数（完整候选 +22.7）和 PixMo 存在性 / 窄候选计数通过，绑定（测不出效果）和 CoSyn（科学 −1.2）不通过。本轮把通过的块合并放大，负责人"可以直接训了"。

**改动**：从 v12 接着训一遍，`data/r14/train.jsonl` 81,876 题、10,235 步，学习率 2e-5 / 1e-5 / 5e-5、warmup 300，结构不变；回放 71%（存在性每类是否各半、原格式计数、科学全量、通用、扩展每源 2,400）。每 2,000 步按冻结规则止损判定，没有触发。

**验收**（`docs/experiments/2026-10-12-r14-count.md`）：主要条件满足——**CountBench 0.189 → 0.334（+14.5 [+10.0, +18.9]）**；旧能力保护未满足——五个集合下界 ≤ −2：`test_relation_flip` −1.4 [−2.8, −0.1]（上界也 < 0），`test_vsr` / `test_position_flip` / `test_bind_gqa` / `bench_lv_aokvqa`（−0.7 至 −1.0，区间含 0）；概率警戒 1 项（What'sUp NLL +0.051）。其余持平或提高：V7W +1.5、GQA +1.1、ScienceQA +1.0（NLL −0.022）、POPE 持平。

**与 laya-vision**：VQAv2 是非 +0.3、A-OKVQA +1.9；ScienceQA −4.8（v12 −5.8）；POPE 三档 88.4 / 86.4 / 80.4，都高于对方。

**结论**：计数目标达成，按规则默认保留 v12；没达标的集中在左右 / 关系类，下降 0.7–1.4 个点，候选原因是镜像对没有单独保底回放。是否采用由负责人决定。

## r13-binding（验收规则在训练前写下；按规则未取代 v12）

**背景**：demo 里"三个红圆 + 两个绿方块，问方块颜色"v12 答红色 0.95。诊断发现模型答的是最显眼物体的属性（`docs/experiments/2026-10-08-r13-binding-plan.md`，评审五轮）。筛选 S0/S1/S2 都没过事先的门槛；负责人批准的 exp13-long 也没过合成门槛（0.643 < 0.8），**负责人 2026-10-10 决定作为取舍采用该配方，进入本轮**。

**改动**：从 v12 接着训一遍，`data/r13/round/train.jsonl` 207,109 道、25,889 步；学习率同 v12（5e-5 / 5e-5 / 1e-4，新模块 1e-4），warmup 500，按反事实单元打包批次；新加 S2 问题条件视觉 token（模型结构变化：`ModelConfig.question_conditioning`，零初始化，CPU 单题延迟基本不变、同图多题每题约 +6 ms，`runs/r13/cpu/`）。训练 2026-10-10 18:51 → 10-11 05:33，全程无 GPU 故障。拟合温度 choice 2.37、**noul 5.0（到了上限）**；分桶 choice:2 1.63、choice:3-5 2.71。

**验收**（与 v12 逐题配对；合成按场景族、其余按图片重采样 2,000 次；`runs/r13-binding/eval/compare/`）：

| 规则 | 结果 |
|---|---|
| **主要：`test_bind_gqa`（封存，第一次打开）提高且下界 > 0** | **通过**：0.679 → 0.755，+7.6 [+5.2, +10.2]；左右上下 +13.3 [+10.4, +16.4]、属性是非 +4.6 [−0.3, +9.7]、属性二选一 0.0、同属性比较 −1.5 [−8.3, +5.0] |
| **主要：`test_bind_cf` 同图两类都对提高且下界 > 0（三层分别报告）** | **通过**：0.138 → 0.633，+49.5 [+47.9, +51.1]；三层 0.817 / 0.782 / 0.756（均 ≥ 0.7）；颜色 0.880、计数 0.691、陷阱题 0.929 |
| 旧能力保护：所有 `test_*` 与四个 `bench_lv_*` 下界 > −2 | **已证明退化**：`bench_pope` −6.5 [−7.6, −5.3]、`bench_lv_pope` −6.7 [−7.7, −5.6]。**未证明守住**（区间跨 −2，上界 ≥ 0）：VSR +0.3 [−2.8, +3.7]、V7W −0.7 [−2.8, +1.3]、VQA 是非 −0.9 [−3.1, +1.3]、尺寸 −1.5 [−3.2, +0.1]、A-OKVQA −1.2 [−3.2, +0.8]、ScienceQA −0.9 [−2.2, +0.6]、新图计数 −2.8 [−5.5, 0.0]。显著提高：关系 +16.9、关系镜像 +19.8、位置镜像 +7.7、位置 +4.7、GQA +3.4。`test_exist`、VQA 多选、`dev_mix` 持平 |
| 概率质量警戒（校准后 NLL 升 > 0.02 且下界 > 0） | **触发两项**：新图计数 0.576 → 0.640（+0.063 [+0.040, +0.088]）、ScienceQA 0.506 → 0.540（+0.035 [+0.010, +0.063]）；VQA 多选 +0.015 [−0.003, +0.036] 未触发 |
| 迁移（报告） | What'sUp 0.284 → 0.589（+30.5 [+25.6, +35.1]）、CoGenT B +5.9 [+3.5, +8.0]、SugarCrepe 属性 −0.8、关系 −1.4（区间含 0）；CountBench（公共候选 1–10，只看选项的最好基线 0.122）0.189 → 0.122，−6.7 [−11.4, −1.8]，两者都接近随机 |
| demo 回归 | **通过**：三红圆两绿方块，"方块什么颜色"green 1.00（v12 red 0.95）；"有没有绿色的圆"0.02（v12 0.48）；"圆比方块多吗"0.63 |

**退化的原因**（2026-10-11 按评审 R13-POST-01–03 修订；复算见报告第五节和 `critic/critic.md`；都是候选解释，没有单变量对照）：
- POPE：模型偏向答"否"。`bench_pope` adversarial 档答"是"比例 0.54 → 0.35、召回 0.83 → 0.62、精确率 0.77 → 0.88。已确认的数据偏斜集中在**真实照片存在性题**：COCO-exist + Objects365 的 13,179 道中"是"只占 15.0%（v12 50.2%），来自回放"不存在"保底只收否题；全部是非题"是"占 41.7%，合成题占 33.7%。温度不改变 argmax，不是原因也不是修法；noul 温度到上限 5.0 不能算作同一偏移的症状。AUROC 也略降（0.960 → 0.951），不只是阈值问题。
- 计数：PixMo 两类计数和 TallyQA complex 的候选项围绕正确数 ±3、只有 2–5 个。"选候选里的下中位数"这条不看图的规则就能答对 51.9% / 44.0%（随机 25.9% / 30.8%）。CountBench 上 r13 有 416/491 次答"5"，配错图也有 397 次；v12 原本就有 384 次答"6"。这一项与是非比例无关。
- ScienceQA：更正"本轮不含示意图"。回放里有 ScienceQA 3,158、AI2D 121、TQA 103 道；必须看图的同文字多答案题从 1,161 道（93 组）减到 747 道（67 组）。该 393 题切片 0.532 → 0.476，−5.6 [−10.4, −0.5]，属探索性分析。

**与 laya-vision**（`eval/compare/laya_vision.md`，括号内为 v12）：VQAv2 是非 +0.1（+0.8）；A-OKVQA +1.5（+2.7）；ScienceQA −6.6 [−8.7, −4.5]（−5.8）；POPE 三档 80.0 / 78.6 / 76.5，**三档都低于对方公开分** 83.6 / 81.9 / 77.7（v12 89.1 / 86.8 / 79.1）。

**结论**：
- **本轮要解决的绑定问题解决得很明显**：封存的真实照片绑定测试 +7.6，左右关系在镜像集上大幅提高（相对位置 +19.8），What'sUp +30.5，demo 那张图答对了。
- **但按事先写的采用规则，默认保留 v12**：POPE 两个集合已证明退化 6.5 个点，概率质量警戒触发两项。
- 退化至少有三处可见的数据问题：存在性题偏否、计数候选捷径、回放未按切片保护。它们是候选解释，没有对照。exp13-long 的混合"是"占 43.5%，但没评 POPE，无法对照。2026-10-11 曾建议从 R13 做恢复；**负责人随后明确撤回该路线，下一轮从 v12 开始**。当前建议见 `docs/experiments/2026-10-11-v12-retrain-plan.md`，旧能力保护以 v12 为基线；未启动训练。
- 负责人 2026-10-11 决定：不采用 r13，也不把它作为训练起点（"13 的破坏性太大"）；下一轮从 v12 开始。

## v12-ext（当前最佳，发布为 v0.2；验收规则在训练前写下）

**背景**：负责人要求架构不变，从网上找多样、量大的图片决策数据，做一个扩展数据包从 v11 接着训，不走评审。来源与许可证见 `docs/research/data-pack-sources-2026-10-06.md`，方案与验收规则见 `docs/experiments/2026-10-06-v12-ext-plan.md`，转换脚本在 `scripts/data/ext/`（`build_ext.sh`）。

**改动**：从 v11 接着训一遍，`data/ext/train_mix.jsonl` 353,830 道、44,229 步（micro batch 8），学习率同 v11（5e-5 / 5e-5 / 1e-4），warmup 500，FP32，seed 0。
- 新来源 315,342 道：Objects365 80,000、TallyQA 40,000、CLEVR 35,000、FigureQA 30,000、SNLI-VE 30,000、IconQA 27,260、Vision-Flan 24,152、VisOnlyQA 11,486、CLEVR-Math 10,000、Super-CLEVR 10,000、MapQA 7,500、SpatialSense 7,059、PixMo-Count 2,885。
- 回放 38,488（含 v9b 的左右原题 2,488）。
- 训练 2026-10-06 22:54 → 10-07 14:36（约 15.7 小时，含每 5,000 步一次的周期评测）；全量评测 14:47 → 16:31。
- 拟合温度 choice 1.96 / noul 1.64（v11 1.99 / 1.82）；分桶 choice:2 1.88、choice:3-5 1.98。

**新技能有没有学到**（`dev_ext`，3,408 道，每个来源按图片哈希留出，训练中监测过，只报告）：总体 是非 59.7 → 65.2、选择 48.0 → 71.3，NLL 0.896 → 0.573，ECE 0.108 → 0.025。
- 学会的：CLEVR-Math 33.3 → 77.0、IconQA 41.3 → 71.7、Vision-Flan 62.7 → 92.3、Objects365 73.0 → 91.0、TallyQA 72.0 → 83.3、SpatialSense 60.0 → 71.3、SNLI-VE 79.3 → 87.7、CLEVR 46.3 → 59.0。
- **没学会的：图表 / 地图 / 几何上"找两处比较"**：FigureQA 50.0 → 51.0（是非题，即随机）、MapQA 49.7 → 53.3、VisOnlyQA 29.6 → 25.5、Super-CLEVR 41.0 → 48.7。和结构筛选 D 的结论一致：模型找不到文字点名的两个区域再做比较。

**验收**（差值 = v12 − v11，逐题配对、按图片重采样 2,000 次，seed 0；`runs/v12-ext/eval/compare/v11_vs_v12.*.json`）：

| 规则 | 结果 |
|---|---|
| **主要：下列至少一项提高且下界 > 0** | **未满足**，四项都只是点估计提高：新图计数 test 0.723 → 0.740，+1.7 [−2.0, +5.2]；必须看图的自然科学 323 题 0.467 → 0.455，−1.2 [−7.1, +4.3]（`sqa_slices.v11_v12.json`）；`bench_pope` +0.7 [−0.2, +1.6]；`test_vqa_choice` +0.7 [−0.6, +2.1] |
| 旧能力保护：所有 `test_*` 与四个 `bench_lv_*` 下界 > −2 | **一处已证明退化**：`test_exist` 0.955 → 0.938，−1.7 [−2.9, −0.5]。其余 16 个集合下界都 > −2，没有其他显著下降（`test_vqa_yesno` −0.6 [−2.8, +1.5]、`bench_lv_vqav2_yesno` −0.5 [−1.5, +0.6]） |
| 左右：原图与镜像同时答对，下界 > −5 | **通过，且大幅提高**（`pairs.v12.*.json`）：位置 0.614 → 0.717，+10.3 [+6.7, +13.9]；相对位置选择 0.390 → 0.653，+26.3 [+22.5, +30.2]；相对位置是非 0.102 → 0.185，+8.3 [+5.3, +11.4]。看图收益（真图 − 配错图）位置 +7.8、相对位置 +11.3，区间都 > 0 |
| 概率质量警戒（校准后 NLL 升 > 0.02 且下界 > 0） | **未触发**，三个集合 NLL 都显著下降：新图计数 0.640 → 0.576（−0.064 [−0.106, −0.022]）、VQA 多选 0.307 → 0.271、ScienceQA 0.541 → 0.506。ECE：新图计数 0.043 → 0.065 升，VQA 多选 0.038 → 0.026、ScienceQA 0.030 → 0.022 降 |

其他集合（显著的加粗）：

| 集合 | v11 | v12 | 差值 [95%] |
|---|---|---|---|
| `test_relation_flip` | 0.575 | 0.669 | **+9.4 [+6.5, +12.4]** |
| `test_relation` | 0.651 | 0.711 | **+5.9 [+3.4, +8.5]** |
| `test_position_flip` | 0.798 | 0.850 | **+5.2 [+2.5, +7.8]** |
| `test_gqa` | 0.745 | 0.784 | **+3.9 [+1.7, +6.3]** |
| `test_vsr` | 0.653 | 0.679 | **+2.7 [+0.0, +5.5]** |
| `bench_lv_aokvqa` | 0.603 | 0.626 | **+2.3 [+0.1, +4.5]** |
| `test_position` | 0.850 | 0.872 | **+2.2 [+0.5, +4.0]** |
| `bench_lv_scienceqa` | 0.747 | 0.766 | **+1.9 [+0.3, +3.4]** |
| `test_count_fresh` | 0.723 | 0.740 | +1.7 [−2.0, +5.2] |
| `test_v7w` | 0.712 | 0.721 | +0.9 [−1.2, +3.0] |
| `bench_lv_pope` | 0.842 | 0.850 | +0.8 [+0.0, +1.7] |
| `bench_pope` | 0.844 | 0.851 | +0.7 [−0.2, +1.6] |
| `test_vqa_choice` | 0.891 | 0.898 | +0.7 [−0.6, +2.1] |
| `test_size` | 0.857 | 0.860 | +0.3 [−1.4, +1.9] |
| `bench_lv_vqav2_yesno` | 0.729 | 0.725 | −0.5 [−1.5, +0.6] |
| `test_vqa_yesno` | 0.716 | 0.710 | −0.6 [−2.8, +1.5] |
| `test_exist` | 0.955 | 0.938 | **−1.7 [−2.9, −0.5]** |

**`test_exist` 为什么掉**：1,120 题中，标注"不存在"的题 v12 新答错 30 道、新答对 4 道；标注"存在"的题新答错 5 道、新答对 12 道。也就是 v12 更愿意说"有"。看了 8 张以 > 85% 说"有"的图：一部分像 COCO 漏标（摆好餐具的餐桌问叉子、桥上路灯问红绿灯、看台座椅问长椅），也有真错（车站图里没有汽车、鱼三明治认成热狗）。所以这处下降部分来自标注噪声，但不能全算噪声。Objects365 标注比 COCO 更全，可能是原因之一，未验证。

**与 laya-vision**（`eval/compare/laya_vision.md`，括号内为 v11）：
- VQAv2 是非 +0.8 [−0.9, +2.3]（+1.2）；A-OKVQA +2.7 [−0.6, +6.0]（+0.4），区间都含 0。
- ScienceQA −5.8 [−7.9, −3.6]（−7.7），差距缩小 1.9 个点，是目前最小的一次。
- 必须看图的自然科学题 0.455（v11 0.467，对方 0.700），仍差 24.5 个点；缩小的部分来自不需要看图的题。
- POPE 三档 89.1 / 86.8 / 79.1，高于对方公开分 83.6 / 81.9 / 77.7（v11 是 86.3 / 84.9 / 81.3：random、popular 升，adversarial 降 2.2）。
- 校准：VQAv2、A-OKVQA 的 NLL 和 ECE 都低于对方；ScienceQA ECE 0.022（对方 0.028），NLL 0.506（对方 0.456）。

**结论**：
- **按事先写的采用规则，默认保留 v11**：主要条件没满足，`test_exist` 已证明退化。
- **但这是迄今范围最广的一次提高**：8 个留出集显著提高、只有 1 个显著下降；左右与相对位置在镜像集上大幅提高，且看图收益同时变大，说明进步来自看图而不是题面；三个警戒集合的 NLL 都显著下降。
- 主要条件没满足，是因为挑的四个集合本身只涨了不到 2 个点（计数和 VQA 多选已经较高，示意图题没动），不是 v12 在这些集合上退步。
- **负责人决定（2026-10-07）：作为取舍采用 v12，取代 v11，并作为对外的 v0.2 发布。** 通过条件不改写。
  - 接受的代价：`test_exist` −1.7（0.955 → 0.938，更愿意说"有"）；POPE adversarial 档降 2.2；新图计数 ECE 0.043 → 0.065；主要条件未满足。
  - 换来的：关系 / 位置 / GQA / VSR / A-OKVQA / ScienceQA 显著提高；相对位置是非题第一次明显动了；ScienceQA 与 laya-vision 的差距从 −7.7 缩到 −5.8。
- **下一步线索**：扩展数据里"在图表 / 地图 / 示意图上找两处再比较"的三个来源都学不会，与 ScienceQA 必须看图题、结构筛选 D 指向同一个短板。继续加同类数据不太可能解决，需要在方法上想办法。

## v11-v9b-recovery（被 v12 取代；验收规则在训练前写下）

**背景**：方案 `critic/plan-round11.md`（修订 1，评审通过）。
- 阶段 0：新建了一套没用过的图上的计数集（`scripts/data/v11_count_eval.py`；dev 300、封存的 test 600）。新图 dev 上 v9b 比 v8 低 4.0 [−7.7, −0.3]，已有 dev 上只低 0.7，没有达到事先写好的启动阈值（实验记录"新图上的计数"一节）。
- **负责人决定作为取舍照样训练**，筛选和验收规则不变。

**改动**：从 v9b 接着训 1 遍，`data/v11/recovery_mix/train.jsonl` 9,988 道不同的题、1,249 步、warmup 125，学习率与 v10 相同（5e-5 / 5e-5 / 1e-4），FP32。
- 计数 3,000（v10 是 1,244 道训 2 遍）；
- 其他回放 4,500，含 ScienceQA 665、VSR 761、Visual7W 475（v10 只有 184 / 210 / 132 道）；
- 左右原题 2,488，与 v9b 相同。
- 训练 09:33 → 约 10:25；拟合温度 choice 1.99、noul 1.82（v9b 2.40 / 2.26，v10 3.51 / 3.01）。

**筛选**（与 v9b 比，`runs/v11-v9b-recovery/eval/gates.json`）：六道全部通过。
- 新图计数 dev +2.3 [−0.3, +5.3]（净多对 7 题），看图收益变化 +4.0 [+0.7, +7.7]；
- 已有计数 dev +1.7；
- 示意图 dev 0.0；
- 旧能力切片 +0.04；
- 位置同时对 +6.4；
- 无来源警戒。

**验收**（差值 = v11 − 对照，按图配对重采样；`runs/v11-v9b-recovery/eval/compare/`）：

| 规则 | 结果 |
|---|---|
| **主要：新图计数 test（600 题，第一次开封）与 v9b 比，至少高 2 个点且下界 > 0** | **未满足**：0.727 → 0.723，−0.3 [−2.7, +1.8] |
| 新图计数 test 与 v8 比，下界 > −2 | **未证明满足**：0.732 → 0.723，−0.8 [−3.2, +1.5] |
| 新图计数 test 的看图收益 | v8 / v9b / v11：+41.2 / +39.7 / +40.3（配错图都在 0.32–0.33），没有变化 |
| 旧计数 136 题（只作参考） | v8 0.809、v9b 0.757、v11 0.772。对 v9b +1.5 [−4.4, +7.4]，对 v8 −3.7 [−8.8, +1.5] |
| 左右：原图与镜像同时答对，与 v9b 比下界 > −5 | **通过，且提高**：位置 0.534 → 0.614，+8.0 [+5.5, +10.7]；相对位置选择 0.368 → 0.390，+2.2 [+0.1, +4.5]；是非 0.129 → 0.102，−2.6 [−5.3, 0.0]（只报告） |
| 所有旧 test 和四个 bench_lv，与 v9b 比下界 > −2 | **两处未证明满足，没有显著下降**：VSR −0.7 [−2.3, +1.0]，A-OKVQA −1.1 [−2.3, 0.0]。其余 14 个集合下界都 > −2；位置 +2.6、位置镜像 +2.8 显著提高 |
| 必须看图的自然科学 323 题 | 无警戒：0.474 → 0.467，−0.6 [−4.0, +2.8]；看图收益 +5.9 → +4.6，变化 −1.2 [−5.6, +2.8] |
| 概率质量警戒（校准后 NLL 升 > 0.02 且下界 > 0） | **未触发**，三个集合的 NLL 都下降：新图计数 0.646 → 0.640，VQA 多选 0.310 → 0.307，ScienceQA 0.550 → 0.541（−0.008 [−0.013, −0.003]）。ECE：新图计数 0.030 → 0.043，ScienceQA 0.019 → 0.030，VQA 多选持平 0.038 |

**与 laya-vision**（`eval/compare/laya_vision.md`，括号内为 v9b）：
- VQAv2 是非 +1.2（+0.9）、A-OKVQA +0.4（+1.5），区间都含 0。
- ScienceQA −7.7 [−9.8, −5.6]（−8.2），差距缩小 0.5 个点，是目前最小的一次，但仍很大。
- 必须看图的自然科学题 0.467（v9b 0.474，对方 0.700），没有改善。
- POPE 三档 86.3 / 84.9 / 81.3，高于对方公开分 83.6 / 81.9 / 77.7。
- 校准：VQAv2 和 A-OKVQA 的 ECE 低于对方；ScienceQA 0.030，与对方 0.028 相当。

**结论**：
- **主要目标没达到。** 在 600 道新图 test 上，v11 与 v9b 几乎相同（−0.3）。
- **"v9b 计数退步"在大样本上也没有得到证实。** 同一 600 题上 v9b 只比 v8 低 0.5 [−3.0, +1.8]。阶段 0 在 300 道新图 dev 上看到的 −4.0 没有在 test 上重现。结合 136 题的宽区间，v8 → v9b 的计数差距很可能比以前估计的小。
- **代价和收益**：
  - 没有证明退化的集合，VSR 和 A-OKVQA 只是下界略低于 −2；
  - 左右题继续提高（位置同时对 +8.0）；
  - ScienceQA 和 laya-vision 的差距缩小到 −7.7；
  - 拟合温度从 2.40 / 2.26 降到 1.99 / 1.82，NLL 在三个集合上都下降。说明一遍训更多不同的题，比 v10 那样重复训，更不容易过度自信。
- **按事先写的采用规则**，主要条件没满足，默认不取代 v9b。
- **负责人决定（2026-10-06）：作为取舍采用 v11，取代 v9b。** 通过条件不改写。
  - 接受的代价：主要目标（新图计数提高）没有达到；VSR（−0.7 [−2.3, +1.0]）和 A-OKVQA（−1.1 [−2.3, 0.0]）没有证明满足 −2 保护线；新图计数和 ScienceQA 的 ECE 略升（0.030 → 0.043、0.019 → 0.030）。
  - 换来的：左右题提高（位置同时对 +8.0）；拟合温度下降、三个警戒集合的 NLL 都下降；ScienceQA 与 laya-vision 的差距从 −8.2 缩小到 −7.7；没有证实的退步。

## v10-count（验收规则在训练前写下；未取代 v9b）

**背景**：第 10 轮原计划补 ScienceQA 示意图题，阶段 A（磁铁吸引 / 排斥）的真实与合成两次集中训练都没有达到"有效"，按方案停止（实验记录"示意图读数没学会"一节）。本轮改走备选：补回 v9b 掉的计数。方案 `critic/plan-round10-count.md`（评审通过）。

**改动**：从 v9b 接着训 2 遍，`data/v10/count_mix/train.jsonl` 4,976 题、1,244 步、warmup 124。
- 左右原题 2,488 道，与 v9b 相同。
- 回放 2,488 道，其中计数 1,244 道（`v9_mix.py --count-share 0.5`）。其他来源因此减少：VSR / ScienceQA / Visual7W 为 210 / 184 / 132 道，v9b 是 375 / 328 / 234 道。
- 训练 22:57 → 约 23:55。拟合温度 choice 3.51、noul 3.01，v9b 为 2.40 / 2.26。

**筛选门槛**（`dev_mix` / `dev_lr_pairs`，`runs/v10-count/eval/count_gates.json`）：四道全部通过。
- 计数 288 题：0.6875 → 0.7257，净多对 11 题，+3.8 [+0.7, +7.2]；
- 位置同时对：+11.9；
- 4,490 题的旧能力切片：+0.02；
- 无来源警戒。

**验收**（按训练前写下的规则；差值 = v10-count − 对照，按图配对重采样；`runs/v10-count/eval/compare/`）：

| 规则 | 结果 |
|---|---|
| 计数补回：`test_vqa_choice` 计数 136 题与 v8 比，点估计 ≥ −2 | **未满足**。与 v9b 完全持平：0.757 → 0.757，净 0 题，+0.0 [−5.2, +5.2]。与 v8 比 −5.1 [−11.0, +0.7]，与 v3 比 −1.5 [−9.6, +5.9]。dev 上净多 11 题的回升没有出现在测试集上 |
| VQA 多选与 v8 比下界 > −2 | **未满足**：0.901 → 0.887，−1.4 [−2.5, −0.4]。与 v9b 比 −0.35 [−1.1, +0.4] |
| 左右收益保住：位置同时对与 v9b 比下界 > −5 | **通过，且继续提高**：位置 0.534 → 0.675，+14.1 [+10.7, +17.3]，看图收益变化 +8.9 [+5.2, +12.5]。相对位置选择 0.368 → 0.484，+11.5；是非 0.129 → 0.112，−1.7 [−4.4, +0.9] |
| 旧能力与 v9b 比下界 > −2 | **一处越线**：laya-vision ScienceQA −1.1 [−2.1, −0.3]，显著下降。其余 15 个集合下界都 > −2；位置 +4.2、位置镜像 +5.5、相对位置镜像 +3.1 显著提高 |
| 校准 | 拟合温度继续升高（3.51 / 3.01）。ECE：dev_mix 0.028 → 0.041，VQA 多选 0.038 → 0.049，位置 0.053 → 0.023，POPE 持平 |

**与 laya-vision**（`eval/compare/laya_vision.md`，括号内为 v9b）：
- VQAv2 是非 +0.8（+0.9）、A-OKVQA +2.0（+1.5），区间都含 0。
- ScienceQA −9.3 [−11.5, −7.2]（−8.2），差距扩大 1.1 个点。
- 必须看图的自然科学题从 0.474 降到 0.427（−4.6 [−8.7, −0.3]），看图收益从 +5.9 降到 −1.2：这部分题比 v9b 更不看图。回放里 ScienceQA 从 328 道减到 184 道，是最可能的原因，未单独验证。
- POPE 三档 86.3 / 84.4 / 81.0，仍高于对方公开分，比 v9b 低约 0.3 个点。
- 校准：VQAv2、A-OKVQA 的 ECE 仍低于对方（0.033 / 0.055，对方 0.062 / 0.092）。ScienceQA 的 ECE 0.039 第一次高于对方的 0.028。

**结论**：
- 本轮的主要目标没有达到：计数在测试集上与 v9b 完全相同，dev 的回升没有迁移过去。
- 回头看，v8 → v9b 的计数 −5.1 [−11.0, 0.0] 本身区间就很宽，136 题只差 7 题，可能有一部分是题量小带来的波动。
- 副作用：
  - 左右继续变好，因为左右原题又训了两遍；
  - ScienceQA 显著变差；
  - 拟合温度持续升高。同一批 2,488 道左右题已累计训了 4 遍，过度自信在加重。
- 建议不取代 v9b；是否取代由项目负责人决定。

## v9b-lr-replay（被 v11 取代；验收规则在训练前写下）

**改动**：从 v8 接着训 2 遍，`data/v9/x1_r2500/train.jsonl` 4,988 题、1,248 步。
- 左右题 2,488 道：R 的原题，即 `data/v9/r_pairs.jsonl` 去掉 12 道近重复题；位置 2,430、相对位置选择 24、是非 34。
- 回放 2,500 道：按 `scripts/data/v9_mix.py` 的配额表缩放，含 VSR / ScienceQA / Visual7W。
- 学习率 5e-5 / 5e-5 / 1e-4，warmup 50。
- 为什么是这个配方：第 9 轮诊断里，集中训左右题学得会（R），混进固定预算就保不住（S-mix），R 的原题加回放保住了（筛选 X1）。经过见 `experiments/2026-09-30-rlcd-plateau.md` 第 9 轮各节，方案见 `experiments/2026-10-05-v9b-lr-replay-plan.md`。
- 训练 17:37 → 约 18:35，评测到 20:20，都在 MPS 上。
- 拟合温度 choice 2.40、noul 2.26（v8 为 1.54 / 1.54）：套温度前更过度自信。

**按训练前写下的规则逐条判断**（差值 = v9b − v8，按图配对重采样；`runs/v9b-lr-replay/eval/compare/`）：

| 规则 | 结果 |
|---|---|
| 左右真的学会：位置题原图与镜像同时对，判为"有效"（下界 > 0 且 ≥ +10） | **通过**：位置 0.233 → 0.534，+30.2 [+25.7, +34.4]。相对位置选择 0.178 → 0.368，+19.0 [+15.3, +22.7]；是非 0.063 → 0.129，+6.6 [+4.0, +9.1]（只报告） |
| 是看图得来的：看图收益（真图 − 配错图）的变化下界 > 0 | **通过**：位置 +10.9 [+5.5, +16.8]，相对位置 +4.6 [+1.5, +7.5] |
| 旧能力保留：所有 test 与 laya-vision 集合下界 > −2 | **未全部满足**：4 个集合下界越线，见下表；其中 VQA 多选显著下降 1.1 个点，另 3 个区间含 0（未证明满足保护线，但也没有证明退化） |
| 计数：与 v3 比 | v3 → v9b −1.5 [−8.8, +5.9]（136 题，区间过宽，未证明满足保护线）；与 v8 比 −5.1 [−11.0, 0.0] |
| 与 laya-vision 比较 | 已生成 `eval/compare/laya_vision.md`，见下 |

| 集合 | v8 | v9b | 差值 [95% 区间] |
|---|---|---|---|
| 位置镜像 `test_position_flip` | 0.598 | **0.770** | **+17.2 [+13.0, +21.2]** |
| 相对位置镜像 `test_relation_flip` | 0.505 | **0.576** | **+7.2 [+4.9, +9.6]** |
| 位置 `test_position` | 0.767 | **0.824** | **+5.7 [+3.8, +7.8]** |
| 相对位置 `test_relation` | 0.621 | **0.650** | **+2.9 [+1.0, +4.8]** |
| POPE `bench_pope` | 0.835 | **0.844** | **+0.9 [+0.2, +1.7]** |
| laya-vision POPE | 0.832 | **0.842** | **+0.9 [+0.3, +1.6]** |
| 尺寸 / 存在 / GQA | 0.853 / 0.953 / 0.751 | 0.862 / 0.952 / 0.750 | +0.9 / −0.1 / −0.1（区间含 0，下界 > −2） |
| laya-vision VQAv2 是非 / A-OKVQA / ScienceQA | 0.733 / 0.613 / 0.740 | 0.726 / 0.613 / 0.742 | −0.6 [−1.3, +0.0] / 0.0 [−1.3, +1.5] / +0.2 [−1.1, +1.4] |
| **VQA 多选** `test_vqa_choice` | 0.901 | 0.890 | **−1.1 [−2.1, −0.1]**（显著下降，下界越线；计数 −5.1） |
| VQA 是非 `test_vqa_yesno` | 0.725 | 0.714 | −1.1 [−2.6, +0.5]（下界越线） |
| VSR `test_vsr` | 0.664 | 0.659 | −0.4 [−2.5, +1.6]（下界越线） |
| Visual7W `test_v7w` | 0.715 | 0.708 | −0.7 [−2.1, +0.6]（下界越线） |

**与 laya-vision**（`runs/v9b-lr-replay/eval/compare/laya_vision.md`，括号内为 v8）：
- VQAv2 是非 +0.9 [−0.6, +2.4]（+1.5）；A-OKVQA +1.5（+1.5）；ScienceQA −8.2 [−10.3, −6.2]（−8.4）。
- 必须看图的 ScienceQA 题仍差 20.1 个点。
- POPE 三档 86.6 / 84.8 / 81.1，对方公开 83.6 / 81.9 / 77.7。
- 校准：同一批题上 ECE 0.026 / 0.052 / 0.019，对方 0.062 / 0.092 / 0.028。

**与 laya-vision 的差距怎么看**（逐题配对；ScienceQA 切片 `eval/compare/sqa_slices.v8_v9b.json`）：
- **相对位置没变**：三个能逐题配对的集合，与对方的差值都在 v8 的 0.6 个点以内，区间重叠。左右题的收益没有体现在 laya-vision 的集合上，这些集合本来就很少考左右。
- **ScienceQA 仍是唯一的大差距（−8.2）**，集中在必须看图的题：自然科学 323 题我们 47.4%、对方 70.0%。这些题上我们的看图收益只有 +5.9（真图 47.4% 对配错图 41.5%），基本没在读示意图。
  - v9b 在这个切片上比 v8 +2.5 [−2.8, +7.4]，看图收益的变化 −0.6 [−5.9, +4.3]。
  - 照片上学会的"找两处、比左右"没有迁移到示意图里的两处比较（磁铁、溶液）。实验记录 10-04 曾推测两者卡在同一处，这次的结果不支持"补左右就能顺带补示意图"。
  - 社会科学（−7.7）和三选项题（−11.4）的差距多数不需要看图，属于知识或文字理解。
- **VQAv2 是非的领先从 +1.5 缩到 +0.9**，来自我们自己 −0.6 的下降，区间含 0。两边各有约 700 多道题只有一方答对，错误相当互补。
- **A-OKVQA 领先 +1.5，未变，不显著。**
- **校准仍明显好于对方**：三个集合的 ECE 都更低（0.026 / 0.052 / 0.019，对方 0.062 / 0.092 / 0.028），但都比 v8 略高（v8 为 0.023 / 0.049 / 0.017），与拟合温度升高一致。NLL 在 VQAv2、A-OKVQA 上更好，ScienceQA 上更差（准确率低）。
- **POPE 三档都高于对方公开分**（+3.0 / +2.9 / +3.4），比 v8 再高约 1 个点。我们答"是"的比例 0.41–0.46、recall 0.771，偏保守，优势来自 precision。对方只公开了每档准确率，不能逐题配对，这一项不是同口径的配对比较。
- **口径**："未见过"按 image id 剔除；按像素约有 1–3% 的题与训练照片相同（实验记录 10-05），去掉后结论不变。laya-vision 201M 有多模态预训练起点，我们的起点是文本 Laya 加新投影层，两者不是同起点的比较。
- **对下一步的含义**：要缩小 ScienceQA 的差距，需要针对示意图的监督，例如磁极 N/S、两处比较的示意图题。继续加照片上的空间数据，预计帮不上。

**校准**：POPE ECE 0.101 → 0.071，相对位置 0.047 → 0.042，位置 0.041 → 0.053，VQA 多选 0.020 → 0.038。

**结论**：
- 第 9 轮的目标达成了：左右关系第一次在留出的镜像集上明显学会，而且是看图得来的。位置题同时对从 23% 到 53%，相对位置选择题从 18% 到 37%。相对位置是非题仍只有 13%。
- POPE 两个集合都显著提高 0.9 个点。
- 代价：VQA 多选显著下降 1.1 个点，主要在计数；另有 VQA 是非、VSR、Visual7W 三个集合点估计下降 0.4–1.1 个点，区间下界越过 −2。按规则这一条未全部满足。
- 拟合温度明显升高，过度自信加重。
- **决定（2026-10-05，项目负责人）**：v9b 取代 v8 成为当前最佳。理由：左右关系第一次在留出镜像集上学会、POPE 提高；代价（VQA 多选 −1.1、计数 −5.1，三个集合未证明满足保护线、温度升高）如上，下一轮补计数。

## v8-vsr-v7w（被 v9b 取代；验收规则在训练前写下）

**改动**：从 v7 接着训一遍，`data/v5/train_mix.jsonl` 43,380 题、34,077 张图、5,423 步（比 v7 多约 55%）= VSR 6,411 + Visual7W telling 7,500 + ScienceQA 4,948 + 新增 TQA 21 + 回放 18,500（v7 的保底 17,000，计数 3,000、GQA 1,500）+ 左右镜像 3,000 对；dev 为 `data/v5/dev_mix.jsonl`（5,178）。数据与剔除见 `research/data-quality.md`（data-v5）。方案与评审在 `critic/`（DV5-01 至 06）。

**结论范围**：检验"新增真实照片问答 + 加大计数 / GQA 回放"这一整套配方相对 v7 的效果，不把变化归因到单一成分；自然科学题不加权，只监测。

**验收规则**（2026-10-04 训练前写下，看到结果后不改）：

| 要回答的问题 | 判断方法 | 通过条件 |
|---|---|---|
| 新能力 | `test_vsr`（904）、`test_v7w`（1,000）与 v7 逐题配对（v7 的结果由 `configs/v7-sqa-v5sets.yaml` 先评好） | 两者区间下界 > 0；真图明显优于配错图 |
| VSR 的构成 | 报告完整集合和去掉左右 4 种关系的集合（左右 92 题），按 VSR 的 7 个大类拆分 | 以完整集合判断；大类和单个关系只作诊断；`test_vsr` 是按图自建的划分，不与官方分布的分数直接比较，训练含 1,307 道官方 VSR test 题，官方 test 不再是独立测试 |
| Visual7W 的构成 | 按 what / where / who / how / why / when 拆分 | 只作诊断；when / why 的文字先验最强（dev 上 0.63 / 0.47），单独说明 |
| 旧能力保留 | 所有 test 集合与 laya-vision 集合与 v7 逐题配对 | 区间下界 > −2，不事后放宽；计数（对 v7 和 v3，136 题）区间过宽时写"未证明满足保护线" |
| ScienceQA 自然科学（监测） | `scripts/sqa_slices.py`：学科 × 必须看图的交叉切片，真图−配错图的差中差 | 不作通过条件，如实报告 |
| 左右 | `scripts/pair_stats.py` | 不作通过条件，如实报告 |

**结果**（训练 5,423 步，2026-10-04 22:43 → 2026-10-05 01:15，约 2 小时 32 分，内存平稳；评测首次在 MPS 上跑，01:15 → 02:59 共 1 小时 44 分，以前 CPU 约 3 小时。拟合温度 choice 1.54、noul 1.54，v7 为 1.72 / 1.61）。v7 的结果是在 CPU 上评的，v8 在 MPS 上：同一模型两种设备在 500 题上预测完全一致、概率最大差 4e-4（`runs/analysis/mps_check/`），不影响配对结论。

与 v7 在同一批题上逐题配对（差值 = v8 − v7，配对题数都等于集合题数；`runs/v8-vsr-v7w/eval/compare/`）：

| 集合 | v7 | v8 | 差值 [95% 区间] | 配错图 v7 → v8 |
|---|---|---|---|---|
| **test_vsr**（904） | 0.618 | **0.664** | **+4.5 [+1.5, +7.7]** | 0.500 → 0.490 |
| **test_v7w**（1,000） | 0.663 | **0.715** | **+5.2 [+3.3, +7.1]** | 0.396 → 0.437 |
| ScienceQA（laya-vision 集合） | 0.709 | **0.740** | **+3.1 [+2.0, +4.5]** | 0.642 → 0.662 |
| 相对位置 | 0.581 | **0.621** | **+4.0 [+1.7, +6.4]**（上下 +5.0 [+2.3, +7.9]，左右 +3.7 [+0.7, +6.5]） | |
| 位置 | 0.752 | 0.767 | +1.5 [−0.9, +3.9] | |
| VQA 多选 | 0.899 | 0.901 | +0.2 [−0.6, +1.1]（计数 **+5.1 [+0.7, +10.3]**，颜色 +1.3） | |
| GQA / 存在 / 尺寸 / VQAv2 是非（test） | 0.750 / 0.955 / 0.854 / 0.726 | 0.751 / 0.953 / 0.853 / 0.725 | 都在 ±0.3 内，下界 ≥ −1.8 | |
| VQAv2 是非（laya-vision 集合） | 0.733 | 0.733 | 0.0 [−0.7, +0.7] | |
| A-OKVQA（laya-vision 集合） | 0.625 | 0.613 | −1.1 [−2.6, +0.4] | |
| POPE（我们的 8,676 / laya-vision 9,000） | 0.842 / 0.839 | 0.835 / 0.832 | **−0.7 [−1.3, −0.2]** / **−0.7 [−1.2, −0.2]** | |
| 位置镜像 / 相对位置镜像 | 0.589 / 0.515 | 0.598 / 0.505 | +0.9 [−3.5, +5.6] / −1.0 [−3.7, +1.7] | |

**新能力拆开看**（`compare/new_sets_breakdown.json`、`compare/new_sets_picture_gain.json`）：
- VSR 的提高来自看图：真图 +4.5，配错图 −1.0，真图−配错图的差中差 +5.5 [+2.3, +9.0]。去掉左右 4 种关系 +4.7 [+1.4, +7.9]；左右 92 题 +3.3 [−5.6, +12.1]（仍接近随机，0.543）。按大类：拓扑 +6.3 [+1.5, +11.1]，投影 +3.8 [−1.2, +9.1]，相邻 +5.4、邻近 +9.3（区间含 0），其余格子太小。
- Visual7W 的提高**大部分是文字先验**：真图 +5.2，但配错图也 +4.1 [+2.0, +6.3]，差中差只有 +1.1 [−1.5, +3.6]。按类型：who +15.3 [+10.0, +21.3]（配错图也从 0.380 升到 0.520），what +5.3 [+2.3, +8.3]，where +1.5，how 0.0，when / why +5.0（区间含 0）。模型学到了 Visual7W 答案的常见写法，看图能力的增长在这个集合上没有被证明。

**按训练前写下的规则逐条判断**：

| 规则 | 结果 |
|---|---|
| 新能力：test_vsr、test_v7w 对 v7 下界 > 0，真图明显优于配错图 | **通过**：+4.5 [+1.5, +7.7]、+5.2 [+3.3, +7.1]；真图 0.664 / 0.715 对配错图 0.490 / 0.437。但 Visual7W 的增量主要在配错图上也有（见上），不能写成看图能力提高 |
| VSR 构成（诊断） | 完整集合与去左右口径都显著提高；左右类仍在随机附近 |
| Visual7W 构成（诊断） | who / what 提高；when / why 的文字先验最强，增量与配错图同步 |
| 旧能力保留：所有 test 与 laya-vision 集合下界 > −2 | **大部分通过，三处越线**：A-OKVQA −1.1 [−2.6, +0.4]、位置镜像 +0.9 [−3.5, +5.6]、相对位置镜像 −1.0 [−3.7, +1.7]，点估计都在 ±1.1 内、区间含 0，属于证据不足而非显著退步；POPE 两个集合**显著下降 0.7 个点**（下界 −1.3 / −1.2，未越 −2 的线），recall 0.76 → 0.73，答"是"更少 |
| 计数（对 v7 和 v3）、GQA | 计数对 v7 **+5.1 [+0.7, +10.3]**；对 v3 0.772 → 0.809，+3.7 [−2.2, +9.6]，下界越线 0.2，**未证明满足保护线**（点估计已超过 v3）；颜色对 v3 +5.3 [0.0, +10.7]。GQA +0.1 [−1.8, +2.1] |
| ScienceQA 自然科学（监测） | 必须看图的自然科学 323 题 0.390 → 0.449，+5.9 [+0.6, +11.1]；差中差 +3.1 [−1.9, +7.7]，看图的贡献未被证明。ScienceQA 总体 +3.1 [+2.0, +4.5]（`compare/sqa_slices.json`） |
| 左右成对（监测） | 原图与镜像同时答对：位置 18.9% → 23.3% [20.0, 26.5]，相对位置 10.2% → 11.9% [10.3, 13.8]（选择题 17.8%）；仍不高于随机的 25% |

**与 laya-vision 201M 逐题配对**（v8）：VQAv2 是非 0.733 对 0.717，+1.5 [−0.1, +3.0]；A-OKVQA 0.613 对 0.598，+1.5 [−1.8, +4.8]；**ScienceQA 0.740 对 0.824，−8.4 [−10.5, −6.2]**（v7 −11.5）；POPE random / popular / adversarial 0.852 / 0.836 / 0.808，对方公开 0.836 / 0.819 / 0.777。详细比较见 `runs/v8-vsr-v7w/eval/compare/laya_vision.md`（`scripts/lv_report.py`）：ScienceQA 的差距集中在必须看图的题——这 393 题对方 72.5%、我们 50.4%（−22.1），其中自然科学 323 题 70.0% 对 44.9%（−25.1）；三选一题 −12.4。逐题对照只有对方对 346 题、只有我们对 170 题。校准上我们三个集合的 ECE 都更低（VQAv2 是非 0.023 对 0.062，A-OKVQA 0.049 对 0.092，ScienceQA 0.017 对 0.028），ScienceQA 的 NLL 对方更低（0.456 对 0.530）。

**建议**：取代 v7。新增的两项能力按规则通过，其中 VSR 的提高确实来自看图；同时 ScienceQA +3.1、相对位置 +4.0、计数 +5.1 都显著。校准因集合而异：ScienceQA、VSR、Visual7W 的 ECE 和 NLL 下降，POPE（ECE 0.091 → 0.103）和相对位置（0.017 → 0.046）变差。dev 在第 2000 步后基本不再提高，不宜原样再训一遍。代价：POPE 显著下降 0.7 个点（答"有"更保守）；A-OKVQA −1.1 和两个镜像集合的区间下界越过 −2（点估计接近 0）；Visual7W 的增量大部分是文字先验。

**决定（2026-10-05，项目负责人）**：v8 取代 v7 成为当前最佳。

**产物**：checkpoint `runs/v8-vsr-v7w/stage2`；评测 `runs/v8-vsr-v7w/eval/`（含配错图 / 倒序对照）、对比 `runs/v8-vsr-v7w/eval/compare/`；v7 在新集合上的对照 `runs/v7-sqa/eval/test_{vsr,v7w}.*`（`configs/v7-sqa-v5sets.yaml`）；SwanLab `v8-vsr-v7w/stage2`、`v8-vsr-v7w/eval`。


## v7-sqa（被 v8 取代）

**改动**：从 v6 接着训一遍（3,497 步，15:54–17:37，内存平稳），`data/v4/train_sqa_mix.jsonl` 27,969 道 = data-v4 全部 ScienceQA 训练题 4,948（示意图按像素完全相同才剔除，比 data-v3 多 2,095 道：州首府、大洋洲地理、经济原理、磁铁……）+ 新增 TQA 21 + 按能力保底的回放 17,000 + v6 的左右镜像 3,000 对。dev 为 `data/v4/dev_mix.jsonl`。拟合温度 choice 1.72、noul 1.61（v6 1.77 / 1.51）。数据见 `research/data-quality.md`。

本轮训练前**没有**单独写验收规则；下面沿用 v6 的口径（其他能力与上一轮配对、区间下界 >−2；计数与 v3 比；左右看原图与镜像同时答对）。

**与 v6 在同一批题上逐题配对**（差值 = v7 − v6，配对题数都等于集合题数；`runs/v7-sqa/eval/compare/v6_vs_v7.*`）：

| 集合 | v6 | v7 | 差值 [95% 区间] |
|---|---|---|---|
| **ScienceQA（laya-vision 集合，2,097）** | 0.584 | **0.709** | **+12.5 [+10.6, +14.4]** |
| A-OKVQA / VQAv2 是非（laya-vision 集合） | 0.622 / 0.736 | 0.625 / 0.733 | +0.3 [−1.1, +1.5] / −0.3 [−1.0, +0.3] |
| POPE（laya-vision 9,000 / 我们的 8,676） | 0.834 / 0.836 | 0.839 / 0.842 | +0.5 [+0.0, +1.1] / +0.6 [+0.1, +1.1] |
| VQAv2 是非（test） | 0.714 | 0.726 | +1.2 [+0.0, +2.6] |
| 位置 | 0.735 | 0.752 | +1.6 [−0.7, +3.9]（上下 +1.7 [+0.8, +2.9]，左右 +1.6） |
| 位置（镜像） | 0.541 | 0.589 | +4.8 [+0.8, +9.2] |
| 相对位置 / 镜像 | 0.581 / 0.505 | 0.581 / 0.515 | +0.1 [−2.4, +2.4] / +1.0 [−1.8, +4.0] |
| 存在 / 尺寸 / VQA 多选 | 0.955 / 0.853 / 0.899 | 0.955 / 0.854 / 0.899 | 都是 0.0–0.1，区间在 ±1.4 内 |
| GQA | 0.760 | 0.750 | −1.0 [−3.0, +1.0]（空间类 200 题 −3.0 [−11.2, +5.0]） |
| 　VQA 多选里的计数 / 颜色 | 0.794 / 0.700 | 0.757 / 0.733 | −3.7 [−9.6, +1.5] / +3.3 [−0.7, +8.0] |

**ScienceQA 的涨幅从哪来**（`scripts/sqa_slices.py`，`compare/sqa_slices.json`；"配错图"= 同一题换一张别的图）：

| 切片 | 题数 | v6 真图 / 配错图 | v7 真图 / 配错图 | v7 − v6 |
|---|---|---|---|---|
| 全部 | 2,097 | 0.584 / 0.569 | 0.709 / 0.642 | +12.5 [+10.7, +14.4] |
| 社会科学 | 762 | 0.490 / 0.480 | **0.810** / 0.711 | **+32.0 [+28.1, +35.8]** |
| 自然科学 | 1,287 | 0.640 / 0.622 | 0.653 / 0.606 | +1.3 [−0.3, +3.0] |
| 语言 | 48 | 0.562 / 0.562 | 0.583 / 0.500 | +2.1（题太少） |
| 必须看图（文字、选项、提示都相同而答案不同，393） | 393 | 0.377 / 0.364 | 0.450 / 0.323 | +7.4 [+2.5, +12.7] |
| 训练里出现过同样题面（436） | 436 | 0.427 / 0.378 | 0.576 / 0.369 | +14.9 [+10.1, +19.7] |
| 训练里没出现过题面（1,661） | 1,661 | 0.625 / 0.619 | 0.744 / 0.713 | +11.9 [+9.9, +13.8] |

- **涨幅基本来自社会科学的模板题**：按主题，地理 594 题 0.49 → 0.83（贡献全部 262 道增量里的 201 道）、经济 64 题 0.59 → 1.00、美国历史 89 题 0.44 → 0.61；自然科学几乎不动（物理 0.55 → 0.55，生物 0.71 → 0.72）。
- **知识题和看图题分开看**（2026-10-04 数据验收修订解释，指标不变）：配错图后 v7 仍有 0.642（v6 0.569），说明模板和常识等非图像线索仍很强（经济题配错图也是 1.00），但不能据此把总涨幅精确拆成知识与视觉的贡献。真图相对错配图的差值从 1.5 升到 6.7 个点；同文字多答案的 393 题从 0.377 升到 0.450，进一步分层后，社会科学 52 题为 0.212 → 0.827，自然科学 323 题为 0.409 → 0.390。读图收益主要由地理题支持，尚不能推广为自然科学示意图能力提升；自然科学子集差值的 95% 区间为 [−6.2, +2.5] 个点，也不足以断言显著退化。
- 题面没在训练里出现过的 1,661 题也涨了 11.9 个点，不只是记住了原题。

**左右**（`scripts/pair_stats.py`，`compare/pairs.v7-sqa.*.json`）：原图与镜像同时答对，位置 14.2% → 18.9% [15.8, 22.0]，相对位置 9.7% → 10.2%，仍低于随机的 25%，左右还是没学会；3,000 对镜像回放保住并略微改善了 v6 的结果。

**计数与 v3 比**（`compare/v3_vs_v7.test_vqa_choice.kind.json`）：0.772 → 0.757，−1.5 [−8.1, +5.1]，点估计接近 v3，但区间宽、下界越过 −2；颜色 0.693 → 0.733，+4.0 [−1.3, +9.3]。

**与 laya-vision 201M 逐题配对**：VQAv2 是非 +1.5 [−0.1, +3.0]，A-OKVQA +2.6 [−0.5, +5.9]（我们没训练过的图片上 +3.3 [+0.0, +6.5]），**ScienceQA −11.5 [−13.7, −9.3]**（v6 是 −24.0）；POPE random / popular / adversarial 0.865 / 0.842 / 0.810，对方公开 0.836 / 0.819 / 0.777。

**按 v6 的口径**：其他能力保留——test 和 laya-vision 集合里只有 GQA 下界 −3.0（点估计 −1.0）和计数（对 v6 −3.7 [−9.6, +1.5]，对 v3 下界 −8.1）越过 −2 的线，两者的区间都含 0；其余集合区间下界都 > −1.1，POPE、VQAv2 是非（test）、位置上下、位置镜像显著上升。

**建议**：取代 v6。ScienceQA +12.5 是这一轮的目标，而且是本项目在 ScienceQA 上最大的一次提高；其余集合没有显著退步。要记下的代价：涨幅大部分来自地理、经济这类模板题，自然科学没进步；计数点估计又掉了 3.7（区间含 0），下一轮回放仍要给计数保底。

**决定（2026-10-04，项目负责人）**：v7 取代 v6 成为当前最佳。

**产物**：checkpoint `runs/v7-sqa/stage2`；评测 `runs/v7-sqa/eval/`（含配错图 / 倒序对照）、对比 `runs/v7-sqa/eval/compare/`；SwanLab `v7-sqa/stage2`、`v7-sqa/test`。


## v6-flip-lr（被 v7 取代；验收规则在训练前写下）

**改动**：从 v5 接着训一遍，`data/v6/train_flip_mix.jsonl` 48,000 道 = 12,000 对左右题（原图 + 左右镜像、答案对调；相对位置 7,065 对、位置 4,935 对，同一原图最多 3 道）+ 24,000 道回放（先保底：计数 3,205、颜色 3,199、上下相对位置 3,010、A-OKVQA 2,530、GQA 1,724、尺寸 1,634、位置上下 1,132、ScienceQA / AI2D / TQA 各约 1,100，其余随机）；长提示截到整题约 150 token。训练中监测 `data/v6/dev_lr_pairs.jsonl`（dev 的 772 道左右题 + 镜像）。数据见 `research/data-quality.md`。

**验收规则**（2026-10-04 训练开始前写下，看到结果后不改）：

| 要回答的问题 | 判断方法 | 通过条件 |
|---|---|---|
| 左右是否真的学会 | test_position / test_relation 的左右题及其镜像版（`test_*_flip`），按原题 id 配对：原图与镜像**同时**答对的比例；另看配错图和选项倒序的翻转率 | 同时答对率明显高于随机（两面都对的随机水平约 25%），且左右选择题倒序后答案翻转率从 v5 的 78% 明显下降；只看两边各自的平均准确率不算 |
| 计数是否补回 | test_vqa_choice 的计数题，**与 v3 比**（不只和 v5 比） | 与 v3 的差值区间下界 >−2 个点；颜色同样报告 |
| 上下题 | test_relation 上下题**按选择 / 是非分开**，是非题再按答案 true / false 分开 | 是非题不再明显偏向答"否"（true 的召回不低于 false 的 10 个点以上）；总体上下题不低于 v5 |
| 其他能力保留 | 其余 test 集合和 laya-vision 集合与 v5 逐题配对 | 区间下界 >−2 个点 |

成对统计需要一个读取明细、按原题 id 配对原图与镜像的小脚本（评审 OPT-T03），在 v6 评测完成前写好（`scripts/pair_stats.py`）。

**结果**（训练 6,000 步约 2 小时 37 分，内存平稳；拟合温度 choice 1.77、noul 1.51，比 v5 的 1.21 / 1.19 更过度自信）。

与 v5 在同一批 test 题上逐题配对（差值 = v6 − v5；`runs/v6-flip-lr/eval/compare/`）：

| 集合 | v5 | v6 | 差值 [95% 区间] |
|---|---|---|---|
| 相对位置 | 0.535 | **0.581** | **+4.5 [+1.9, +7.3]**（左右 +4.8 [+1.5, +8.1]，上下 +3.5 [+0.2, +6.8]） |
| 位置 | 0.716 | 0.735 | +2.0 [−0.3, +4.2]（左右 **+5.3 [+0.9, +10.0]**，上下 −1.4 [−2.7, −0.2]） |
| 尺寸 | 0.827 | **0.853** | **+2.5 [+1.0, +4.0]** |
| VQA 多选 | 0.883 | **0.899** | **+1.5 [+0.5, +2.6]**（计数 +9.6 [+4.4, +15.4]，颜色 +5.3） |
| GQA / VQAv2 是非 / 存在 / POPE | 0.761 / 0.719 / 0.959 / 0.840 | 0.760 / 0.714 / 0.955 / 0.836 | 都在 ±0.5 个点内，区间含 0 |

**按训练前写下的规则逐条判断**：

| 规则 | 结果 |
|---|---|
| 左右真的学会：原图与镜像**同时**答对明显高于随机（约 25%），且倒序翻转率明显下降 | **未通过**。同时答对：位置 5.0% → 14.2%，相对位置 2.5% → 9.7%（选择 11.9%、是非 7.5%），都**低于**随机的 25%——模型对同一问题多半给同一个答案，镜像后不跟着变。翻转率这一半通过：左右选择题倒序后答案改变 78.2% → 35.2%。 |
| 计数补回：与 **v3** 比下界 >−2 | **差一点**：0.772 → 0.794，+2.2 [−2.2, +7.4]，下界越线 0.2；比 v5 +9.6 显著补回。颜色与 v3 持平（+0.7）。 |
| 上下题：是非题不再明显偏"否"，总体不低于 v5 | **通过**：是非题答案为"是"的正确率 0.451 → 0.619，与"否"（0.689）差 7 个点；上下总体 0.721 → 0.755。 |
| 其他能力保留（与 v5 比下界 >−2） | test 集合：GQA 下界 −2.2 刚越线（点估计 −0.1），其余通过。laya-vision 集合（10-04 补评）：VQAv2 是非 0.0 [−0.6, +0.6]、A-OKVQA 0.0 [−1.6, +1.4]、POPE −0.3 [−0.9, +0.2] 通过；**ScienceQA 0.598 → 0.584，−1.5 [−3.0, −0.1]，下界越线**。 |

镜像测试集（`runs/*/eval/test_*_flip.*`，v5 的结果是 2026-10-04 训练后补评）：v5 原图 / 镜像 0.514 / 0.503（位置）、0.479 / 0.522（相对位置）；v6 0.567 / 0.541、0.527 / 0.505。成对明细 `runs/v6-flip-lr/eval/compare/pairs.*.json`。

与 laya-vision 201M 逐题配对（v6）：VQAv2 是非 0.736 对 0.717，+1.9 [+0.2, +3.4]；A-OKVQA 0.622 对 0.598，+2.4 [−0.8, +5.7]；ScienceQA 0.584 对 0.824，−24.0；POPE random / popular / adversarial 0.856 / 0.838 / 0.808（对方公开 0.836 / 0.819 / 0.777）。配错图后：A-OKVQA 0.382、ScienceQA 0.569（v6 在 ScienceQA 上仍主要靠文字）。

**决定（2026-10-04，项目负责人）**：v6 取代 v5 成为当前最佳（未通过的规则：左右成对、计数对 v3 下界 −2.2、ScienceQA 对 v5 下界 −3.0，均记录在上表）。理由：我们自己的 test 集上计数 +9.6、尺寸 +2.5、相对位置 +4.5（上下 +3.5）、位置左右 +5.3、多选 +1.5，没有显著退步；laya-vision 集合三项持平。代价：ScienceQA −1.5（下界 −3.0，越过 −2 的线）、拟合温度上升（过度自信加重，套温度后各集合 ECE 仍在 0.002–0.10）、左右关系仍未学会。

**结论**：镜像成对训练让模型对选项顺序没那么敏感、左右单题准确率小幅提高，但没有学会"按文字找到物体再判断左右"——同时答对率仍低于随机，这是之前所有尝试（实验记录）之后又一个负结果。另一方面回放保底很有效：计数、尺寸、多选和上下题都比 v5 好。


## v5-data3（被 v6 取代）

**改动**：从 v3 接着训一遍（7,771 步），数据 `data/v3/train_cont.jsonl` 62,164 道 = data-v3 新增的 32,164 道（A-OKVQA 14,918、TQA 5,941、AI2D 5,443、ScienceQA 2,848、上下相对位置 3,014）+ 从 data-v2 混合随机回放 30,000 道；不再把 data-v2 混合整遍重训。ScienceQA 的提示文字训练时也送入模型。dev 是 `data/v3/dev_mix.jsonl`（4,531），温度按题型 + 选项数分桶拟合：choice 1.21、noul 1.19，桶 choice:2 1.31、choice:3-5 1.19（v3 是 1.58 / 1.29，过度自信进一步减轻）。约 3 小时 30 分（中途内存紧张、换页拖慢约半小时，关掉 OrbStack 后恢复）。

**验收规则**（训练进行中、看到最终结果前由评审写下；原方案文档在评审的临时 worktree 里，未提交、已不存在，规则以此处记载为准）：A-OKVQA、ScienceQA 相对 v3 的 95% 区间下界 >0；GQA 与 VQA 多选的损失容忍 2 个点（区间下界 >−2）；相对位置上下题 ≥80%；新任务真图明显优于错配图。

**test 集与 v3 逐题配对**（差值 = v5 − v3；配对题数都等于集合题数；`runs/v5-data3/eval/compare/`）：

| 集合 | v3 | v5 | 差值 [95% 区间] | ECE v3 → v5 |
|---|---|---|---|---|
| POPE（3 档） | 0.832 | 0.840 | +0.7 [+0.3, +1.3] | 0.102 → 0.099 |
| GQA | 0.755 | 0.761 | +0.6 [−1.1, +2.4] | 0.011 → 0.023 |
| VQAv2 是非 | 0.714 | 0.719 | +0.5 [−1.3, +2.3] | 0.031 → 0.029 |
| 存在 | 0.958 | 0.959 | +0.1 [−0.4, +0.7] | 0.013 → 0.011 |
| VQA 多选 | 0.892 | 0.883 | −0.8 [−2.1, +0.3] | 0.026 → 0.017 |
| 　其中计数 | 0.772 | 0.699 | **−7.4 [−14.0, −1.5]** | |
| 　其中颜色 | 0.693 | 0.647 | −4.7 [−10.0, +0.7] | |
| 尺寸 | 0.841 | 0.827 | −1.4 [−3.0, +0.2] | 0.027 → 0.043 |
| 位置 | 0.730 | 0.716 | −1.4 [−3.5, +0.9]（上下 0.920 不变，左右 −2.8） | 0.029 → 0.012 |
| 相对位置 | 0.519 | 0.535 | +1.6 [−0.2, +3.4] | 0.019 → 0.031 |
| 　其中上下 | 0.572 | **0.721** | **+14.8 [+10.6, +19.2]** | |
| 　其中左右 | 0.503 | 0.479 | −2.5 [−4.3, −0.6] | |

配错图后各集合回到随机或只看题目的水平（存在 0.491、多选 0.376、GQA 0.533），分数仍来自看图。

**按验收规则（test 部分）**：
- GQA 保留：通过（下界 −1.1 > −2）。
- VQA 多选保留：**证据不足**——点估计只掉 0.8 个点，但区间下界 −2.1 刚过 −2 的容忍线；其中计数显著退步 7.4 个点、颜色 −4.7（这两类是 v3 进步最大的地方）。
- 相对位置上下题：**未达标**——从 0.572 升到 0.721（+14.8，显著），但低于 80% 的门槛（v4 专项训练是 0.828）。
- 左右仍在随机水平。

**laya-vision 评测集与 v3 逐题配对**（配对题数都等于集合题数）：

| 集合 | 题数 | v3 | v5 | 差值 [95% 区间] | v5 配错图 |
|---|---|---|---|---|---|
| A-OKVQA | 1,138 | 0.561 | **0.622** | **+6.2 [+3.9, +8.5]** | 0.369 |
| ScienceQA | 2,097 | 0.485 | **0.598** | **+11.4 [+9.2, +13.7]** | 0.576 |
| VQAv2 是非 | 4,887 | 0.728 | 0.736 | +0.8 [−0.1, +1.6] | 0.519 |
| POPE（9,000） | 9,000 | 0.830 | 0.837 | +0.7 [+0.2, +1.2] | 0.522 |

- v5 在 A-OKVQA 上从正确图片得到的收益很大：真图 0.622、配错图 0.369（四选一随机 0.25）。
- **v5 在 ScienceQA 上从正确图片得到的收益很小**：真图 0.598、配错图 0.576，只差 2.3 个点（逐题：175 题真图才对，127 题配错图反而对）——题目和提示文字已经给出了大部分答案。按选项数：二选一 0.732、三选一 0.590、四选一 0.501、五选一 0.227。v3 没有 ScienceQA 的配错图结果，所以 v3 → v5 的增益里视觉占多少（ΔG）没有测。

**按事先写下的验收规则逐条判断**：

| 规则 | 结果 |
|---|---|
| A-OKVQA 相对 v3 区间下界 >0 | **通过**：+6.2 [+3.9, +8.5] |
| ScienceQA 相对 v3 区间下界 >0 | **通过**：+11.4 [+9.2, +13.7] |
| GQA 损失 ≤2 个点（下界 >−2） | **通过**：+0.6 [−1.1, +2.4] |
| VQA 多选损失 ≤2 个点（下界 >−2） | **证据不足**：−0.8 [−2.1, +0.3]，下界刚越线；计数 −7.4 显著退步 |
| 相对位置上下题 ≥80% | **未通过**：0.721（比 v3 +14.8） |
| 新任务真图优于错配图 | A-OKVQA **通过**（+25.3）；ScienceQA **很弱**（+2.2） |

**按事先规则的判断**：v5 **不满足正式取代 v3 的条件**——关键旧能力（VQA 多选，尤其计数）的保留证据不足，上下题未达标。但 v5 在常识推理（A-OKVQA）上有明确的视觉增益，POPE、VQAv2 是非也小幅提高，是下一轮更合适的起点：第 6 轮建议从 v5 起，在回放里给计数、颜色设最低配额（评审 OPT-T02），并补上下相对位置题，再按同一套规则验收。ScienceQA 的分数主要来自文字，不能写成"看懂了教材插图"。

**最终决定（2026-10-04，项目负责人）**：v5 取代 v3 成为当前最佳。理由：A-OKVQA +6.2、ScienceQA +11.4、上下相对位置 +14.8、POPE / VQAv2 是非小幅提高，其余 test 集合在 ±1.5 个点内；与 laya-vision 相比除 ScienceQA 外都已追平或领先。这一决定没有按事先写下的验收规则（VQA 多选保留证据不足、上下题未达 80%），记录在此；计数题 −7.4 [−14.0, −1.5] 是已知退步，第 6 轮回放给计数、颜色设最低配额来补。

**与 v4 在同一批 test 题上配对**（差值 = v5 − v4；`runs/v5-data3/eval/compare/v4_vs_v5.*`）：GQA +3.3 [+1.1, +5.7]（v4 的 GQA 退步在 v5 没有出现）；相对位置上下题 −10.7 [−14.8, −6.7]（0.828 → 0.721，v4 是上下题专项训了两遍）；存在、位置、尺寸、VQA 多选、VQAv2 是非、POPE 都在 ±2 个点内、区间含 0。v5 的拟合温度（1.21 / 1.19）远低于 v4（2.67 / 2.00），过度自信轻得多。v4 没有跑 laya-vision 评测集，新题型无法与它比。

**产物**：checkpoint `runs/v5-data3/stage2`；评测 `runs/v5-data3/eval/`（含 laya-vision 集合的配错图 / 倒序对照）；SwanLab 见 `runs/v5-data3/stage2/swanlab_run.json`。

## v4-relation-tb（未取代 v3）

**改动**：从 v3 接着训，数据是上下相对位置题 3,186 道（COCO 1,402 + GQA 场景图 1,784，上下各半）加从 v3 训练混合随机回放的 9,558 道，2 个 epoch 3,186 步，学习率同前、预热 200（`docs/research/data-quality.md`）。约 1 小时 20 分。拟合温度 choice 2.67 / noul 2.00（v3 是 1.58 / 1.29）：同一批数据训两遍后明显更过度自信。

**与 v3 在同一批测试题上逐题配对**（差值 = v4 − v3，95% 区间按图片重采样；`runs/v4-relation-tb/eval/compare/`）：

| 集合 | v3 | v4 | 差值 [95% 区间] |
|---|---|---|---|
| 相对位置 | 0.519 | **0.561** | **+4.2 [+1.5, +6.8]** |
| 　其中上下 | 0.572 | **0.828** | **+25.5 [+20.4, +30.6]** |
| 　其中左右 | 0.503 | 0.479 | −2.4 [−5.3, +0.7] |
| POPE | 0.832 | 0.838 | +0.6 [+0.0, +1.3] |
| VQAv2 是非 | 0.714 | 0.722 | +0.8 [−1.0, +2.7] |
| VQAv2 多选 | 0.892 | 0.889 | −0.3 [−1.3, +0.7] |
| 存在 | 0.958 | 0.959 | +0.1 [−0.5, +0.7] |
| 尺寸 | 0.841 | 0.843 | +0.2 [−1.3, +1.7] |
| 位置 | 0.730 | 0.714 | −1.6 [−3.4, +0.3] |
| GQA | 0.755 | **0.728** | **−2.7 [−4.8, −0.7]**（空间题 −7.0 [−14.0, 0.0]，其余 −1.6） |

8 个集合的平均准确率 0.780 → 0.782。校准变差：相对位置 ECE 0.019 → 0.070，GQA 0.011 → 0.035。

**结论**：补上下相对位置题有效，上下从 0.572 回到 0.828（v2 是 0.860）；左右仍在随机水平。但 GQA 显著退步、校准变差，按"上下恢复且其余无显著退步"的标准不取代 v3。第 5 轮从 v3 起步，用已含这些上下题（3,014 道）的 data-v3 混合一遍训练，避免同一批数据重复多遍。

**产物**：checkpoint `runs/v4-relation-tb/stage2`；评测 `runs/v4-relation-tb/eval/`；SwanLab `irzm68kz`。

## v3-data2（被 v5 取代）

**改动**：对齐结果沿用 v2（`runs/v2-align-lora/align`），阶段 2 不再是 2a + 2b 两段，而是一段在 data-v2 训练混合上训练：134,556 题（左右类 4.07%，每图最多 6 题），1 个 epoch 16,820 步，学习率、预热与 v2 相同。dev 用 data-v2 的 dev_mix（第 4000 步起换成按图片减半的 dev_mix_half，每 2000 步评一次；见 `runs/v3-data2/run.json`）。约 6 小时，无塌缩，中途停过一次并从第 4000 步的续训状态接上。拟合温度 choice 1.58 / noul 1.29（v2 是 2.71 / 1.57，过度自信明显减轻）。

**与 v2 在同一批测试题上逐题配对**（差值 = v3 − v2，95% 区间按图片重采样；`runs/v3-data2/eval/compare/`）：

| 集合 | 题数 | v2 | v3 | 差值 [95% 区间] | ECE v2 → v3 |
|---|---|---|---|---|---|
| VQAv2 多选（13 类） | 1,420 | 0.826 | **0.892** | **+6.5 [+4.8, +8.4]** | 0.142 → **0.026** |
| COCO 存在 | 1,120 | 0.931 | **0.958** | **+2.7 [+1.5, +3.9]** | 0.023 → 0.013 |
| VQAv2 是非 | 1,000 | 0.691 | 0.714 | +2.3 [+0.0, +4.7] | 0.030 → 0.031 |
| COCO 位置 | 1,274 | 0.708 | 0.730 | +2.2 [−0.5, +4.8] | 0.035 → 0.029 |
| GQA val | 992 | 0.750 | 0.755 | +0.5 [−2.4, +3.6] | 0.051 → 0.011 |
| POPE（全部 3 档） | 8,676 | 0.828 | 0.832 | +0.4 [−0.5, +1.3] | 0.084 → 0.102 |
| COCO 尺寸 | 1,100 | 0.837 | 0.841 | +0.4 [−1.9, +2.7] | 0.029 → 0.027 |
| COCO 相对位置 | 1,950 | 0.598 | **0.519** | **−7.9 [−11.5, −4.3]** | 0.017 → 0.019 |

8 个集合的平均准确率 0.771 → 0.780。POPE 对 v3 是严格未见集，对 v2 是训练中监测过的集合。

- **多选题大幅进步来自颜色和计数**：颜色 0.493 → 0.693（+20.0 [+9.3, +30.0]），计数 0.493 → 0.772（+27.9 [+17.6, +37.5]），材质 +7.9 [+0.7, +15.1]；其余类别本来就在 0.9 以上。data-v2 修好了这两类题的数据（见 `research/data-quality.md`）。
- **校准整体更好**：多选 ECE 0.142 → 0.026，GQA 0.051 → 0.011；POPE 略差（0.084 → 0.102）。
- **来自看图**：配错图后各集合回到随机或只看题目的水平（存在 0.494、多选 0.361、位置 0.500、尺寸 0.503）。
- **左右仍没学会**：位置左右 0.516 → 0.542（+2.7 [−1.7, +7.5]，不显著），相对位置左右 0.518 → 0.503；GQA 空间题 0.53。相对位置选择题倒序后 76% 的答案跟着翻转，说明还是按选项位置猜。
- **相对位置的上下题退步**：0.860 → 0.572（−28.8 [−34.2, −23.5]）。v3 答 "below" 和 "false" 的偏向很强（选择题 above 0.42 / below 0.79，是非题 true 0.32 / false 0.77）。原因：data-v2 训练混合里上下类相对位置题只有 172 道（左右 1,402 道）。`v2_mix.py` 给相对位置整类只配 2,500 道，从上下 1,402 / 左右 7,414 的池里随机抽，再经左右占比限额和重新配平，上下只剩 172 道，而 v2 的上下能力来自 v1 的 COCO 出题数据（2b 阶段）；v3 从 v2 的对齐结果起步，没有经过那一段。不是遗忘，是没学到。位置类（单物体在图中上下）上下题有 4,552 道，所以位置题上下没有退步（0.902 → 0.920）。

**laya-vision 评测集**（`configs/v3-data2-lvbench.yaml`，与 laya-vision 201M 逐题配对，差值 = v3 − 对方）：VQAv2 是非 0.728 对 0.717，+1.1 [−0.4, +2.6]（打平略高）；A-OKVQA 0.561 对 0.598，−3.8 [−7.2, −0.4]（比 v2 的 −7.1 缩小一半）；ScienceQA 0.485 对 0.824，−34.0；POPE random / popular / adversarial 0.850 / 0.836 / 0.805，对方公开分 0.836 / 0.819 / 0.777。相对 v2：A-OKVQA +3.3 [+0.9, +5.9]，其余持平。详见 `research/laya-vision-gap.md`。

**还没做**：Hugging Face 目录（`../devision-hf`）目前放的仍是 v2。

**产物**：checkpoint `runs/v3-data2/stage2`；评测 `runs/v3-data2/eval/`；SwanLab `uz6smi3f`（评测汇总在 `test/`、`ref/`，用的是改名前的集合名（eval_ 和 lv 开头）；结果文件已改名为 `test_*` / `bench_*`）。

## v2-align-lora

**改动**：只有一处——对齐阶段让 ModernBERT 也通过 LoRA 学（v1 只训投影层）。数据（data-v1）、步数、阶段 2 的配方与 v1 完全相同。

**与 v1 相比的增长**（同一批题；数字已套用拟合温度；v1 当时没有存逐题明细，只能比点估计，没有置信区间）：

| 集合 | 题数 | v1 | v2 | 增长 |
|---|---|---|---|---|
| VQAv2 是非（val） | 1,000 | 0.648 | **0.724** | **+7.6 个点** |
| COCO 实例框出题（val_cocoqa） | 1,000 | 0.761 | **0.812** | +5.1 |
| POPE adversarial（旧 300 题） | 300 | 0.773 | 0.787 | +1.4（300 题，标准误约 2.4 个点，在噪声内） |
| GQA testdev（是非 + 二选一） | 1,000 | 0.635 | 0.642 | +0.7（在噪声内） |

分阶段看，增长主要在对齐和阶段 2a：

| 阶段 | 指标 | v1 | v2 |
|---|---|---|---|
| 对齐 | val 补词准确率（真实图） | 0.501 | 0.611 |
| 对齐 | 真实图 − 错配图的差（图片带来的部分） | 0.140 | 0.207 |
| 2a 终点 | VQAv2 / GQA / POPE | 0.634 / 0.607 / 0.713 | 0.723 / 0.635 / 0.767 |
| 2b 终点 | VQAv2 / GQA / POPE | 0.648 / 0.635 / 0.773 | 0.724 / 0.642 / 0.787 |

2b（COCO 题 + 回放）对 VQAv2 基本没有再加分（0.723 → 0.724），只提高了同源的 COCO 出题集。

**v2 在 data-v2 测试集上**（训练没见过的图片，带对照；v1 没有在这批集合上评测过）：存在 0.931、尺寸 0.837、VQA 选择 0.826、GQA 0.750、VQAv2 是非 0.691、相对位置 0.708（上下 0.90，左右 0.52）、关系 0.598（上下 0.86，左右 0.52）。配错图后全部回到随机或只看题目的水平，说明分数来自看图。

**与 laya-vision 201M 在同一批题上逐题配对**：VQAv2 是非 +0.4 个点 [−1.3, +2.0]（打平）；A-OKVQA −7.1 [−10.5, −3.5]、ScienceQA −35.3（这两类题我们没训练过）；POPE 三档 0.842 / 0.835 / 0.805，对方公开分 0.836 / 0.819 / 0.777。详见 `research/laya-vision-gap.md`。

**还弱的地方**：左右关系在随机水平；选择题里颜色、计数约 0.49；POPE 偏向答"没有"（recall 0.71）；VQA 选择题 ECE 0.142。

**限制**：

- 与 v1 的比较没有置信区间；VQAv2 / GQA 是温度拟合集，POPE / COCO 是训练中监测过的集合（准确率不受温度影响，但不是严格的未见集）。
- 两轮的初始投影层不同（当时 CLI 在播种前建模型，评审 MC1，已修），差距大，初始化差异大概率解释不了，但没有量化。
- 要给 v1 → v2 的增长加区间：用 `configs/v2-align-lora-testsets.yaml` 的同一批集合补评 v1（只评测），再 `devision-compare`。

**产物**：checkpoint `runs/v2-align-lora/stage2b`；评测 `runs/v2-align-lora/eval/`；SwanLab 对齐 `uxknjl4o`、2a `nlkszqdl`、2b `lei97w39`（评测汇总在 2b 的 `test/`、`ref/`，用的是改名前的集合名）。

## v1-release-0.1

**改动**：首个完整流程。对齐（COCO caption 8.3 万张图，只训投影层）→ 2a（GQA 6 万 + VQAv2 是非 4 万，低学习率 + 预热，避免长训练塌缩）→ 2b（COCO 实例框题 5.7 万 + 回放 2.3 万）。Mac 上约 16 小时。

**结果**：VQAv2 是非 0.648、GQA 0.635、COCO 出题 0.761、POPE adversarial 0.773（300 题）；ECE 拟合集 0.025–0.042，POPE 0.098。左右关系在随机水平。

**产物**：checkpoint `runs/v1-release-0.1/stage2b`（发布的 0.1）；评测 `runs/v1-release-0.1/eval/`（只有汇总 JSON，没有逐题明细）；SwanLab 对齐 `or3c40kc`、2a `dftqxbzw`、2b `njwfkisb`。

## 2026-10-25 18:24 +08:00：负责人纠偏，恢复训练优先

最近数轮偏重诊断/接入，真实训练仅exp94读出100步、exp96末层q/v低秩64步，尚未形成性能提升。撤回自行设定的18:00冻结；下一阶段优先D3真实科学图示监督适配与旧能力回放，先核对隔离，再扩大可训层的独立对照。已实际盘点data/v4/scienceqa.jsonl为4948题4443图，science_mix为123084题，盘点不代表已通过全量隔离。证据runs/v05-prep/science_training_inventory_20261025.json。已将纠偏写入原CC-Connect任务064e9c68，无新增任务。本次尚未启动新训练，不虚报。

## 2026-10-25 19:10 +08:00：exp110真实科学适配开始参数更新

**当前阶段/实际状态**：按负责人18:24指令撤销旧冻结、恢复训练优先；仍采用v0.4，v0.5未发布。19:03实查RTX5070空闲无训练评测，完整核对最新实验/日志后预写[exp110](experiments/2026-10-25-exp110-d3-science-adaptation.md)，19:08启动。19:10:44实查Python435289实际更新53/256步，CE .366、梯度范数1.280，峰值显存2.92GB；训练前640条已完成，尚无after结果，不能声称能力改善。约1秒/步，预计本轮训练与复验再需5–8分钟。

- **已执行，直接观察**：科学池ScienceQA4948、AI2D5855、TQA5540题；96份输入/评测文件，核对98869个路径、保留52418个评测引用，按身份排除478、额外像素排除4。1783个历史无来源图片引用不可访问，unknown来源隔离；未验证这些缺失图的像素，也未证明近重复与外部预训练曝光隔离。证据data/v05/exp110-retry2/MANIFEST.json、pixel_hashes.json；两次失败准备的脚本及日志全保留。
- **实际训练**：固定2048题，三科学源各512，UI171/图表171/旧任务170回放；初始化D3固定03560c…，末四层MLP及全注意力q/k/v/o共16模块、秩8、532480参数，CE监督、lr1e-4/warmup16、有效batch8、save64。与失败exp96数据/可训路径均不同，不能单独归因。配置configs/exp110-d3-science.yaml；实际身份、逐步loss/梯度和checkpoint在runs/exp110-d3-science/train，外层training.log。骨干BF16，需最终部署FP32复验，不能继承旧校准。
- **基线直接观察**：科学256题72.2656%/NLL .610308；图表128题86.7188%；UI128题100%；大小64题70.3125%、位置64题89.0625%。固定面板与原生输出首题核对通过；均为monitoring，非正式五用途。
- **运行限制**：nohup supervisor提前退出，独立session训练子进程健康；内部2400秒预算仍逐题执行，但原外层保护已失效，不能虚报EXIT_CODE=0。保持主动核对，不为采集码重启健康任务。exp81实查2073754784 bytes、所有唯一权重/失败证据保留；原CC-Connect064e9c68唯一每小时启用、last_error空，未新建任务。
- **发布阻断**：科学退化待after验证；正式ImageJevBench v0.3.0接入/public300与score语义尚缺（exp109证据是旧v0.1 profile228/456，不能代替目标版本），五用途、独立校准、全回归与部署验收仍未完成。发布授权有效，当前证据不支持达标发布。

**下一步**：本轮完成256步及固定before/after、科学无图/倒序对照，独立按图bootstrap，科学至少+3pp且下界>0、NLL不恶化、UI/图表各退化≤2pp才晋级后续。19–21点按结果决定有效续训，21–23点回归与可行校准，23–07点正式验收/材料，07–09点发布缓冲；不能解除正式证据缺口时明确未达标，不降低标准。实际已执行py_compile及git diff --check，未重装环境、改许可或外发发布。

2026-10-25 19:20:28 +08:00补充实查：exp111已新增116/1024步，CE .508239、梯度 .288819，64步patch/optimizer已保存；GPU45%/3464MiB，Python435954与supervisor435950正常，暂无after结论。日志runs/exp111-d3-science-coverage/training.log，未启动并行GPU任务；原小时Cron继续保留唯一任务。

## 2026-10-25 20:07:18 +08:00：exp111未通过，exp112已实际更新参数

**阶段/结果**：exp111新增1024步与640题after完整结束，GPU实查空闲后才启动exp112，未重复抢卡。exp111科学184/256，相对原D3185/256为−0.3906pp、图片配对95%[−3.125,+2.344]pp，未通过预写+3pp门槛；科学NLL .610308→.568093不能替代准确率门槛。图表111→116/128（+3.9063pp），UI128/128保持，大小45→47、位置57→56/64。停止末四层路径简单加步数，不晋级。独立验证1792条预测概率/标签/NLL、1024步与17组patch/optimizer，见runs/exp111-d3-science-coverage/train/{original_baseline_verification,integrity_verification}.json；原supervisor退出码缺失，不虚报exit0。

**本轮实际训练**：预写[exp112](experiments/2026-10-25-exp112-d3-science-depth.md)，从固定原始D3初始化，唯一配方变量末四层→末十二层（三次全注意力+十二层MLP），48模块、1597440参数。2048条输入与exp110逐字节一致，100份池/评测文件SHA256未变，沿用已核对的身份/可访问RGB隔离。640题before与exp110概率最大差0.0。当前实际31/256步，CE 0.800925、梯度1.155042、峰值显存5.40GiB；预计约20:14前完成训练与固定after/control，尚无新能力改善结论。训练PID438209、supervisor438205，工具会话63344；日志runs/exp112-d3-science-depth/training.log，checkpoint与完整身份在train/，save64、内部2400秒/外部2700秒，异常即停。

**重要限制/阻断**：1783张历史保留图片不可访问，不能证明其像素及近重复隔离；正式ImageJevBench v0.3.0/public300与admission、五用途、最终FP32路径校准/性能验收均未解除，当前证据不支持达标发布。exp86 CUDA问题已在历史原始复现中定位processor image_seq_len169与模型64不一致，非已确认词表越界，seq64修复及失败目录均保留，不重复运行。exp81权重实查2073754784 bytes；所有唯一权重保留。

**下一步**：按exp112原科学+3pp/配对下界>0/NLL不恶化/UI图表≤2pp下降判据决定是否扩大覆盖；失败则不简单续步。20–22点有界训练、22–01点回归/必要独立校准、01–07点正式验收与材料、07–09点发布缓冲；不恢复旧冻结，不降低发布质量。原CC-Connect064e9c68每小时启用，未新建或递归调度。更新exp111/112实验记录与发布就绪文档撤销旧冻结；实际py_compile及git diff --check通过，未重装环境、改许可或发布。

## 2026-10-25：负责人停止全部工作

负责人明确要求全部停止、不再训练，并停止定时任务。已停用CC-Connect任务064e9c68（enabled=false），移除本项目系统小时监控/执行cron，保留PAUSE标记。实查无本项目训练、评测或独立定时执行进程；权重、数据与历史实验全部保留。v0.5目标暂停，未经负责人明确恢复不启动训练、实验或发布。

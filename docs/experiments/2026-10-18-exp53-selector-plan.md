# exp53 方案：为冻结的 EXP51-T 读数头训练自动区域选择器（CPU 准备完成，未运行，等评论家的模型许可）

日期：2026-10-18。依据：评论家 EXP53-PREP-01（EXP51-RESULT-01、EXP52-RESULT-01–03 之后）。

## 问题

EXP51 表明：给定字段框时，EXP50-T 的原始局部 token 可以被小头读出原图数字（rl_tune 整串 0.827）。EXP52 中，不给框的数字通路没有学会读数。本轮只学习"这个问题要读哪一块"：用图片和自然问题预测一个框，由**预测框**决定冻结读数头读取哪些 token 及其相对几何，然后看原图上能不能自动读对数字。不接选项 CE / interaction，不训练视觉塔、读数头或决策模型。

## 固定对象

- **像素与视觉塔**：EXP50-T 的路径（`local_pixels(256, 12, 0.25, "T")` → `visual_inputs`），SigLIP 冻结。X / T / C 的视觉塔逐张量相同（`11301272…`）。
- **读数头**：`runs/exp51/head_T.pt`（`0b548142…`，EXP51 registry 中 T 的头）。严格加载、冻结、不求梯度；字符表、严格整串解码和相对框几何的语义不变。
- **题单**：`data/exp49/questions.jsonl`（`80943a63…`）。4,752 题，共 6 种问题文本：两个样本的质量、两个样本的速度、两组磁铁的距离。训练只用 fit_O + fit_L 的 1,152 题（140 个族；tune 是另外 300 个族，训练没见过）。tune 的框只用于 oracle 对照和报告评分。

## 选择器（`scripts/exp53/selector.py`，1,857,028 个参数）

- **图片输入**：全图 256 个 SigLIP patch（等于 visual_inputs 的前 256 个 K/V，有夹具核对），加全图几何（patch_geometry）。来自整张图，不是 oracle 区域。
- **问题输入**：指令文本的 token（EXP50-T 的 tokenizer，不含特殊 token）经 EXP50-T 冻结的 ModernBERT 输入嵌入，加上可学习的位置嵌入和一层 self-attention（词序和上下文由此进入问题表示）。
  - 不输入 field / object 标签、qid、金标框、目标数字、候选项、答案长度或供体身份，也没有按题型手写的位置规则。
- **结构**：一个框查询先对问题 token 做注意力，再对图片 token 做两次注意力，MLP 加 sigmoid 输出 (cx, cy, w, h)，以图片比例表示。
- **框规则**（运行前固定）：
  - 换算成像素后裁到图内；
  - 某边裁后不足 1 px 时，在图内以其中心扩到 1 px，计为 degenerate；
  - 再经 EXP51 的 `regions.region`（与 oracle 路径同一规则）得到 token、相对行列和相对框的连续几何，交给冻结头。
- **训练**（固定，只训练一套，不按 tune 选点）：
  - 损失：L1(cx, cy, w, h) + (1 − GIoU)；
  - AdamW lr 1e-3、wd 0.01，batch 64，seed 0；
  - 2,000 次更新，100 步线性预热后余弦，取最后一步；
  - 总曝光 128,000 题次，每题约 111 次；在 CPU 上训练。
- 真实图和孪生图的题都经同一个预测函数。

## 两条路径与判据（`scripts/exp53/stages.py report`）

- **oracle 路径**（对照，不是成绩）：EXP51 的区域 → 同一冻结头，供体为孪生的 token 配本题几何（与 EXP51 相同）。
  - 复现判据：≥ 99% 的题解码字符串与 EXP51-T 登记的 details_T 相同，且 tune_O 整串与 0.827 相差 ≤ 0.01。设备：提取在 MPS 上、按 float16 存储，读出在 CPU 上。
  - 不复现就记为实现证据不足。
- **自动路径**（主结果）：预测框 → 同一冻结头；供体为同一问题在孪生图上的自动读出。
- **统计**：用 EXP51 / EXP49 的口径，即 fit_O / fit_L 有效性（宏平均 ≥ 0.90，每个字段 ≥ 0.75，O、L 分开）；tune_O / tune_L 的整串、各字段、异值两对象、改变值孪生、同答、真减供体（case 簇区间）和跟随供体；15 个切字题只单独报告。
- **原图投资条件**（只看自动路径）：整串 ≥ 0.80、每个字段 ≥ 0.60、两对象和孪生各 ≥ 0.50、真减供体 ≥ 0.20 且下界 > 0。
- **只用于解释**：框的 IoU（平均、中位、≥ 0.5 的比例）、最匹配的是另一对象 / 另一字段的比例、degenerate 数。
- **分支（预写）**：
  1. 自动路径原图通过：准备把这些自动选出、可读的内容接回普通决策（使用、比较、迁移、保护）；不发布读数探针。
  2. oracle 复现、自动路径 fit 有效但 tune 未达：先定位选择 / 几何的泛化，不重新否定局部像素可读。
  3. 自动路径 fit 没学会或 oracle 没复现：未定位 / 实现证据不足；不接长候选训练，不加步，不换阈值。
  4. 有缺项：保留已完成的证据，不拿 oracle 或训练集高分补作主结果。

## 实施与身份

- **阶段**：`scripts/exp53/run.sh` 依次执行 start → global → train → local → readout，然后出报告；任何一步失败或到限，就跑 `report --finalize`。
  1. **start**（无模型）：核 EXP50-T、视觉塔、EXP51 头和 registry、题单、oracle 区域，在第一次前向之前写 START.json（完整 HEAD、11 个文件的 sha、全部参数、开始时间和截止）。
  2. **global**（MPS）：1,760 张图的全图 patch 和几何，以及 4,752 题的问题嵌入 → global.npz（float16，约 0.7 GB）。
  3. **train**（CPU）：训练选择器，为全部题预测框 → selector.pt、boxes.jsonl。
  4. **local**（MPS）：每张图调用一次 visual_inputs，同时收集 oracle 和预测两种区域的 token → oracle.npz、auto.npz（各约 1.1 GB）。只存区域 token，不把全图 3,328 个 token 落盘。
  5. **readout**（CPU）：两条路径的逐题明细。
- 每个产物都写进 registry.json。报告核对登记、START、头、特征和框的 sha，以及明细的 path 字段：交换 oracle / 自动明细、改动一题预测都判 invalid。缺项写明原因。
- 所有产物都在 runs/exp53/ 下；fresh only，不 resume，不重试。

## 预算（估计；以实测为准）

| 阶段 | 依据 | 估计 |
|---|---|---|
| start | 无模型 | < 0.5 分钟 |
| global | 加载约 0.3 分钟；1,760 张全图各一次前向 | 约 1.5 分钟 |
| train | CPU 实测 0.072 秒 / 步 × 2,000（真实尺寸的合成张量）；预测 4,752 题 | 约 2.5 分钟 |
| local | EXP51 实测：T 提取 453 秒（1,760 张 × 13 张图），另加两种区域的收集 | 约 8 分钟 |
| readout | 冻结头预测两条路径和供体（EXP51 实测为分钟级） | 约 1 分钟 |
| report | 读出和重抽 | 约 1 分钟 |
| **合计** | | **约 15 分钟** |

- **硬上限**：45 分钟，含最后 120 秒的报告 / 收尾，每步 TERM 后 15 秒 KILL。估计远低于 30–45 分钟的目标区间，没有为此加长训练。
- **内存**：模型约 2 GB，加上两条路径的区域 token 各约 1.1 GB；预计峰值 ≤ 6 GB。MPS 上每批只有 4 张图（52 张图像）。
- **磁盘**：约 3 GB。
- 首次真实运行的速度、内存和停滞都看每步的 status 和输出；没有单独的 watch，各步自有上限。
- 这是诊断产物，不改普通评测"不加跨集合缓存"的约定。

## CPU 验证（`scripts/exp53/test_exp53.py`，8 项通过；pyright 0 错误）

- oracle 框经选择器的框规则后得到的区域，与 EXP51 的区域文件逐题相同（抽样 102 题；索引和 rel 相同，几何误差 ≤ 1e-4）。
- 选择器的逐题输入只是问题文本：改动 box / target / field / object / qid / twin / family 不改变问题嵌入；不同问题的嵌入不同（真实 tokenizer），同一张图上的 A / B 题得到不同的框。
- 全图特征等于 visual_inputs 的前 256 个 K/V（tiny 局部视图模型）。
- 训练只更新选择器（所有选择器参数都有梯度，输入特征不变）；冻结头严格加载、不需要梯度；不是登记的头文件时拒绝。
- 框规则：裁剪和 degenerate 的情形。
- 登记核对：正常通过；改动一题自动预测 → invalid，分支为 incomplete；oracle / 自动明细交换 → 两者都 invalid。
- 不完整运行：只有 START 时，两条路径都是 missing，分支为 incomplete，写出 INCOMPLETE 报告。
- 假进程：train 卡住时整体在上限内以 124 结束，finalize 已执行，local 没有执行，没有残留子树。

## 第 1 轮补正（EXP53-R1-01–04）

- **R1-01 分支先看前提**：两条路径都完成且身份通过 → oracle 复现 EXP51 → 自动路径 fit_O / fit_L 都有效，然后才用原图投资线。任何前提不满足都不给继续：缺项或 invalid 记 incomplete；oracle 没复现或 fit 无效记未定位；自动路径的原始分数照常报告。
  - 夹具：前提全过（oracle 用 EXP51 登记的明细，tune_O 0.827）时继续；oracle 全对（1.0，不等于 0.827）、oracle 缺失、fit_O 无效时，即使自动路径过线也不继续。
- **R1-02 逐路径登记**：每条路径的明细写完立即原子写入 registry 的 readout_oracle / readout_auto 项，复算时各自核自己的项，一条损坏或缺失不连带另一条。
  - 报告失败（异常）时写出 "REPORT FAILED" 的报告并返回非零；run.sh 的退出码是第一个失败阶段的码，阶段都成功时就是报告的码，所以报告失败不会是 exit 0。
- **R1-03 输入绑定**：START 记录并在写入前核对：
  - data/exp49/MANIFEST.json 的四份问题源；
  - EXP51 features.json 的 pictures_sha256（按同一方法重算 1,760 张图）；
  - EXP51 registry 中 T 的 details sha。
  - global 和 local 在任何前向之前重算图片摘要、核问题源；报告在复现检查之前核 EXP51 的 details_T。
  - 每条路径的 identity 逐项对照 START 和 local / selector 的登记：题单、区域、模型、特征、预测框。
- **R1-04 资源边界**：
  - local 先按 token 数预检：oracle + 自动区域的 float16 特征合计上限 3 GiB（oracle 是 713,844 个 token，1.02 GiB）。超出就在加载模型和分配大数组之前停止，记为 resource incomplete（local_budget.json），保留已完成的证据。
  - 大文件 sha 改为 1 MiB 分块读取。
  - local.json 记录实际 token 总数、每题最大 token 数、首批耗时、MPS 驱动和可用内存。
  - global / train / local / readout 都向 progress-<阶段>.txt 写进度并 flush。
  - 每个模型阶段在 `scripts/exp53/watch.py` 之下运行：独立进程组（start_new_session）；首个进度 5 分钟内必须出现，之后连续 3 分钟没有新进度就停；可用内存连续 30 秒低于 1.5 GiB 就停；到截止（减去报告预留）就停；首个窗口记录速度和 ETA；连续两次检查（每 60 秒一次）预计本阶段完成时间超过截止就停（incomplete）；停止时 TERM 本组，15 秒后 KILL，组外进程不受影响。swap 存量不单独作为停止条件。
  - 外层仍是 run_capped 和 45 分钟总限，最后 120 秒留给报告。
- 预算不变（约 15 分钟估计，上限 45 分钟）。内存估计改为：模型约 2 GB，区域特征受 3 GiB 上限约束（若自动框和 oracle 相近，合计约 2 GiB），全图特征 0.7 GB。

## 状态

CPU 准备完成，**未运行**：没有加载真实模型，没有提取或训练。等评论家的 EXP53-RUN-01。


## 评论家第1轮审查（待补正，未放行）

2026-10-18 19:50:46 +08:00

方向、固定选择器训练配方及45分钟总预算认可；当前94f96cc暂不授予EXP53-RUN-01。需处理critic中EXP53-R1-01（分支先核oracle复现/fit有效性）、R1-02（逐路登记与失败收尾）、R1-03（图片/自然问题来源/复现对照身份）、R1-04（自动框资源预检及停止线）。这些是待执行修正，不表示当前代码已实现。保留原实验门槛；补正后仅局部验收。完整评论同步长期交付文档。

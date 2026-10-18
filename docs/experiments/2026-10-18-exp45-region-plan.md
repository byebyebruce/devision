# exp45 方案：监督实际参与决策的视觉区域选择（S / G 两臂；CPU 准备完成，未运行，等评论家的模型许可）

日期：2026-10-18。依据：评论家 EXP45-PREP-01 / 02（EXP44 终点裁决之后）。

## 问题与假设

- EXP43：从文字转录按对象取值可用（tune 两对象 0.943）。
- 原图、左右问法、EXP44 的按问题补数都没有建立绑定。
- 待验证的假设（不是已证实的根因）：如果直接监督"按问题选择目标字段区域"，这个信号能不能传到正常的选项决策？
- 可读性、视觉信息量、优化不足和补数格式问题都还没有完全分开，本实验不宣称一次排除所有解释。

## 与 r13 的区别

r13 试过两种做法，都没有证明模块单独有收益（见 2026-09-30-rlcd-plateau.md 中 r13 的修订结论）：

- S1：另建一个训练专用的定位头，保存前删掉；
- S2：QuestionConditioning，但没有定位监督。

本实验不是原样重跑：

- 监督直接加在 `src/devision/model.py:QuestionConditioning` 的注意力上。这正是读 SigLIP patches 并回加到视觉 token 的那条真实路径，推理时也走它。
- 目标是当前真实版式中被问对象的被问数字字段。
- 对照是同结构、只差这项监督的 S 臂。

## 两臂（唯一的臂间变量是区域损失权重）

| | S（对照） | G（区域监督） |
|---|---|---|
| 配置 | `configs/exp45-s.yaml` | `configs/exp45-g.yaml` |
| region_weight | 0.0 | 0.5 |

- 两臂都从 EXP40-M 冻结终点（`3fb4232e…` / `74e32e7c…`）独立起跑，都启用零输出初始化的 QuestionConditioning（`question_conditioning: true`，lr_cond 1e-4）。
- 其余与 EXP42-C 相同：CE，aux_weight 0；学习率、seed 1、micro / eval 8、视觉塔顶 4 层、val_sub、监测和保存设置。
- 数据：EXP42 的 144 个单元，每臂 576 次更新。训练 JSONL 原样不变（`27e57372…`），区域标签放在独立 sidecar 中。
- **注意力路径两臂相同**：
  - 只要给了 region_file，训练时就让模块保留注意力（`keep_weights`，即 `need_weights=True, average_attn_weights=True`）。S 的权重为 0 也走同一条路。
  - 两臂都不做额外前向，分辨率相同。
  - 训练结束、合并 LoRA 之前关闭 keep_weights，所以合并后的评测和保存的模型都走 decide 的原路径。
- **区域损失**（`src/devision/train/rlcd.py`）：
  - 注意力：在头和 64 个视觉 token query 上取均值，得到 256 个 SigLIP patch 上的分布。
  - 目标：框与每个 patch 的相交面积占比（`box_grid`，按实际 letterbox 变换），以 0.9 / 0.1 与均匀分布混合（`REGION_SMOOTH = 0.1`）。不用"patch 中心落在框内"的规则，所以小框不会丢。
  - 损失：KL(目标 ‖ 注意力)，逐题取均值；loss = CE + region_weight × KL，一次 backward、一次 clip / step。KL 非有限时在更新前报错。
  - 权重和平滑都在开跑前锁定，不看结果改值。
- **初始化与批计划**：模块在设种子之后加入（r13 原有的代码位置），两臂相同，所以两臂的初始化、LoRA 和批次顺序都相同。
  - 与 EXP42-C 相比，因为多了模块的随机初始化，LoRA 初始化不同；所以 M、C 只作参考，S 才是匹配对照。
  - PLAN.json 不变（576 个满 8 的批，每批一个单元；批计划只用 python random，模块初始化不消耗它）。
- **身份**：region_file、region_weight 写进 TrainConfig 和 resume 键；train_report 写入权重、平滑和 sidecar 的 sha256；chain 另写进 START.json。

## 区域标签（`scripts/exp45/build.py` → `data/exp45/`）

- **原图**：按生成器的版式公式取框（只读重放 `real_layout.families()`），不按答案串长度裁框。
  - 温度：被问对象瓶下该字段的那一行——行中心按 temp_render（行序由 speed_first 决定），高一行，宽为该对象所在半幅、左右各缩 2 px。
  - 磁力：该对的距离标注带——宽为整个 Pair 框，高为标注行 ± 字号。
- **L 图**：exp37 共享版式中该对象该字段的固定数字框（大小只由版式决定）。
- **核对**：每个框都用实际画出的内容核过——被问数值的文字（温度是整行，磁力是距离标注）在框内，另一对象同字段的文字不与框重叠；L 图核框内数字等于 gold。
  - 训练 4,608 题、观察用的 4,752 题（fit_O / fit_L / rl_tune / rl_tune_L）全部通过。
- 数值、候选、gold、转录或框都不进入模型输入。框只在训练中作为监督目标；tune 的框只用于观察，不参与 loss、拟温或选 checkpoint。
- **分布**（训练）：原图和 L 各 2,304 题。A / Pair 1 在左 296 / 256、在右 88 / 128（B / Pair 2 相反）；speed_first 为 True 1,152、False 1,920（温度题）。
- **sha256**：regions.jsonl `a8dc5142…`；regions_readout.jsonl `e83ae1f5…`。
- **预览**：`runs/exp45-preview/sheet.png`，12 组。规则：字段 × 被问对象 × 左右，每个字段的原图取 A 在左 / B 在右、L 图取 A 在右 / B 在左，温度画在右侧的取速度行在上。
  - 每组画出框（红色）和 16 × 16 目标分布。我看过，框都落在被问对象的被问字段上。

## 读出（`scripts/exp45/report.py`；运行前写定）

- **主读出**：decide、原问题和完整原图、温度 1；fit_O / fit_L / rl_tune / rl_tune_L。
  - 统计沿用 EXP42 的四项和 case 簇；差值 S − M、G − M、G − S（另报 S − C、G − C 作参考）。
  - M、C 复用 EXP42 的冻结记录；exist / science 只报告风险；报告不混入拟温概率。
- **身份**：
  - M / C 是冻结身份；
  - 每臂 START / run.json / train_report 的 objective、question_conditioning、region_weight、sidecar sha、起点和数据 sha 都与本臂一致；
  - 每个文件都记录在本臂冻结身份下；摘要和逐题核对同前。
  - 缺项记为 incomplete。
- **条件**（预写；投入条件，不是发布条件）：两臂的四个主集合都读到才判定，否则全部 incomplete。
  - progress_S / progress_G：原图 rl_tune 异值宏平均 ≥ 0.60；两对象 ≥ 0.20 且（该臂 − M）下界 > 0；孪生 ≥ 0.20 且（该臂 − M）下界 > 0。
  - region_attribution：原图 rl_tune 两对象 G − S ≥ 0.10 且下界 > 0。如果 S 也过线而 G 没有可靠的额外收益，不归功于区域损失。
  - 只有 fit 或 L 改善、两臂都没达到原图条件：停止这条路线，不降阈值、不加种子；结论只限于这个预算和设置。
- **注意力观察**（只作机制观察，不作门槛；`scripts/exp45/attention.py`，S、G 各在 rl_tune 和 fit_O 上）：
  - 注意力在目标框上的质量 vs 另一对象同字段框上的质量 vs 框的均匀面积占比；
  - 注意力最高的 patch 是否落在框内；
  - 同一张图的 A / B（Pair 1 / 2）两题，注意力最高的 patch 是否不同。
  - 按字段、版式和左右分开。定位学会了但决策没好，不授权长训。

## 运行与预算（`scripts/exp45/chain.sh`）

- **启动前检查**：没有其他模型进程；M、C 冻结；M 的 sha；exp42 数据和 exp45 两个 sidecar 的 sha；运行目录原本不存在。
- **截止**：启动后 90 分钟，最后 120 秒留给报告。各步有界（TERM，15 秒后 KILL）；不重试，不接下一个候选。
- **顺序**：S（训练 → 冻结 → 4 个主集 → 注意力 2 集 → exist / science）→ G（同上）→ 报告。
- **watch**（`scripts/exp45/watch.py`）：
  - 首窗或相邻两个窗口 > 4 秒 / 更新 → 停止；
  - MPS > 16 GB 且可用内存 < 2 GB → 停止；
  - 停滞上限：首窗前 20 分钟、第 500 窗口之后 20 分钟、其余 10 分钟。
- **预算估计**：参考 EXP42-C 实测 1.39–1.41 秒 / 更新、EXP42 / 44 的评测吞吐。

| 部分（每臂） | 估计 |
|---|---|
| 加载 1 + 3 次 val_sub 约 5 + 576 × 约 1.5 秒（多了模块和 need_weights 路径）约 14.5 + 保存 1 | 约 21.5 分钟 |
| 4 个主集约 6 + 注意力 2 集（2,376 题，只前向）约 2.5 + exist / science 约 1.5 | 约 10 分钟 |
| 每臂合计 | 约 31.5 分钟 |
| 两臂 + 报告 | 约 64 分钟（上限 90，余量约 26 分钟） |

- 首窗出来后按实测重算。如果装不下，先砍注意力观察和风险集，不砍主读出。

## CPU 验证

- `tests/train/test_region.py`，6 项：
  1. 零初始化模块加入后，decide 的答案不变（1e-6 以内），注意力分布和为 1；
  2. S / G 的批次顺序和第一步损失相同，之后 G 不同；report 记录权重和平滑；保存加载后不再保留注意力，decide 不需要框；
  3. 默认关闭；给了 region_file 但没开 question_conditioning 时拒绝；
  4. 缺框、框出界都拒绝；目标是 256 维、和为 1、最小值为 0.1 / 256；
  5. KL 非有限时在更新前失败，不留 model.safetensors；
  6. KL 的梯度到达 question（query）、patches（key）和注意力投影，并且换一个框梯度就不同。
- `scripts/exp45/test_exp45.py`，4 项：
  1. sidecar 覆盖每道训练题，框在图内，对象、字段、图片和版式对应；
  2. 注意力观察在 tiny checkpoint 上端到端可运行；
  3. 报告拒绝错误的权重或 sidecar；任何一臂缺主集时条件全部 incomplete；
  4. 缺臂时为 incomplete，C 的读数复现 EXP42。
- 已有测试照常通过：tests/model、test_pipeline、test_resume、test_distill 共 52 项；test_binding_screens（r13 的 S2）、test_number_aux、test_objective。
- chain 演练（假 uv）：启动检查通过；S、G 各 9 项按序 completed、冻结记录 8 个集合；报告在假摘要处拒绝（预期）。
- pyright 0。没有运行真模型。

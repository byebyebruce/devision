# exp48 方案：直接监督图像与对象的数值交叉差分（C / X 两臂；CPU 准备完成，未运行，等评论家的模型许可）

日期：2026-10-18。依据：评论家 EXP48-PREP-01..04（EXP47 终点之后）。

## 假设

当前逐题目标可能让模型长期停在"图像偏置 + 问题偏置"的加性解附近。直接监督"同图换对象、同问换数值图"的交叉差分，能不能在普通推理路径里学出对象与数值的对应？

- 这是待检验的假设，不是已查明的梯度根因。
- EXP47 的跨空间 RMS 比值只说明放大后的差分能传到候选 logits，不能写成"排除了所有融合或读出问题"。
- 不加产品 oracle 框、目标数字、左右坐标或特制推理分支；不扫倍率，不加 region_weight。

## 两臂（只差 interaction_weight）

| | C | X |
|---|---|---|
| 配置 | `configs/exp48-c.yaml` | `configs/exp48-x.yaml` |
| interaction_weight | 0 | 1.0 |

- 两臂都从 EXP45-G 冻结终点（`c34b43dd…` / `2a5dfa64…`）独立起跑，保留 question_conditioning 的结构和权重。
- 两臂都关闭 region / number / grounding / distill（没有 region_file）。
- 两臂都读取同一份侧表，并计算交叉差分项用于监测；C 的权重为 0，不反传；随机数消耗和批次相同。
- 数据和设置：exp42 的 4,608 行、576 次更新（PLAN 为每批一个 8 题单元，其中原图和 L 各一个四题组），CE，学习率 new 2e-5、head 1e-5、LoRA 5e-5、cond 1e-4、视觉塔顶 4 层 5e-6，micro / eval 8，warmup 100，seed 1，amp off，固定终点。
- val_sub 只用于监测和拟温；主读出全部用温度 1。

## 损失（`src/devision/train/rlcd.py`）

- 对每个四题组：a = 视图 0 中第一个对象（A / Pair 1）的 gold，b = 另一个对象的 gold；视图 1 中两者交换。
- 定义：
  - m[v,o] = z[v,o,a] − z[v,o,b]，z 为普通决策的原始候选 logits；
  - J = (m[0,A] − m[0,B] − m[1,A] + m[1,B]) / 4；
  - interaction_loss = 批内各四题组 softplus(1 − J) 的均值。
  - 在 PLAN 下，每批正好是原图和 L 各一个四题组，所以这就是"两种画法的均值"。
- margin 1、权重 1 固定，不调参。a / b 来自金标映射（侧表），不看数值大小、候选位置或当前预测。不 detach 任何一个角。
- 只用本次 8 题的前向，与 CE 相加后一次 backward / clip / step；没有额外前向。
- 可分加性的 m = u[v] + w[o] + k 会使 J ≡ 0，损失恒为 softplus(1)，所以不能只靠问题偏置或不区分对象的图像分数来降低它。
- 高 J 也可能来自某一个角被过度放大，不等于四题全对，所以保留 CE，并以逐题离散读出验收。
- 这些孪生图还带有随数值变化的粒子大小和磁铁几何，成功只支持当前场景里的对象数值决策，不能写成通用 OCR。
- **身份**：TrainConfig.interaction_file / interaction_weight（写进 resume 键）；train_report 记权重、公式版本 `J4/softplus(1-J)/v1` 和侧表 sha；chain 另写进 START.json。默认关闭时不读侧表，也没有额外计算。

## 侧表与 PLAN（`scripts/exp48/build.py`、`plan_check.py`）

- `data/exp48/quads.jsonl`（`5fecf2c7…`）：从 data/exp42/train.jsonl 构造，原数据不改。
  - 共 1,152 个四题组 = 144 单元 × 4 次重复 × 原图 / L；每个画法和字段各 192 组；140 个族。
  - 每行记录：四题组、视图、对象、a / b 的候选索引。
  - 对象从题面文字识别，不按 qid 硬编码。
- **全量核对**（不满足即拒绝）：
  - 四个角唯一，金标严格交换；
  - 四题候选字符串和顺序相同，同对象跨视图的问题文本相同，state_text 相同；
  - 图片属于规定的视图和画法；
  - 每行恰好属于一个四题组。
- **真实 PLAN**（`data/exp48/PLAN.json`；read_decision_jsonl → _items（G 的 tokenizer）→ random.seed(1) → plan_batches(8)）：576 次更新；每批正好两个完整四题组（一个原图、一个 L）；没有四题组被拆开。

## 读出与条件（`scripts/exp48/report.py`；运行前写定）

- G 的四个主集和 exist / science 复用 EXP45-G 的冻结温度 1 记录：先做 frozen check，再走逐题身份和全集核对。冒烟复现了 EXP45 的读数（原图 tune 两对象 0.0035）。
- C、X 各自在固定终点评 fit_O / fit_L / rl_tune / rl_tune_L，再评 exist / science；温度 1、普通 decide。
- **指标**：
  - 沿用 EXP42 的异值单题、两对象同时对、值改变孪生两视图同时对、同答，字段等权，case 簇重抽（2,000 次，seed 0）；
  - 新增**严格交换四题组四题全对**：按 quads_of 从读出集构造，按字段等权；不合格的组列出排除数，不混入。
    - 原图 tune：176 组合格，274 组因"没有交换"被排除（两对象同值或孪生不是互换）。
    - fit_O / fit_L 各 144 组。
- **差值**：X − C、X − G、C − G（含四题全对的差值区间）。
- **条件**（预写；只用于讨论是否扩大候选训练，不是发布条件；两臂四个集合都读到才判定，否则全部 incomplete）：
  - progress_X（原图 rl_tune）：X 异值宏平均 ≥ 0.60；两对象 ≥ 0.20 且 X − G 下界 > 0；孪生 ≥ 0.20 且 X − G 下界 > 0；四题全对 ≥ 0.10 且 X − C 下界 > 0。
  - attribution：X − C 两对象 ≥ 0.10 且下界 > 0。两臂同时进步只说明从 G 续训的配方有效。
  - stop_flag：两臂 fit_L 的异值读数都 < 0.60 时，不再用同样设置加步或加权。
  - 只有 L / fit 改善，或只有 J / 训练 loss 改善，都不接原图长训。

## 运行与预算（`scripts/exp48/chain.sh`）

- **启动前检查**：没有其他模型进程；G 冻结检查和 sha；exp42 数据的 sha；侧表 sha；PLAN 显示 576 次更新且每批两个完整四题组；运行目录原本不存在。
- **截止**：启动后 90 分钟，最后 120 秒留给报告；C → X 顺序执行；不重试，不接任何别的运行。
- **watch**（`scripts/exp48/watch.py`）：首窗或相邻两个窗口 > 3 秒 / 更新时停止；内存规则同前；停滞上限 20 / 20 / 10 分钟。
- **预算**：每臂训练约 21 分钟（576 × 约 1.43 秒 + 3 次 val_sub + 加载和保存），评测约 7.5 分钟，合计约 28.5 分钟；两臂加报告约 58 分钟，上限 90。首窗出来后按实测重算。

## CPU 验证

- `tests/train/test_interaction.py`，8 项：
  1. J 与手算一致，四个角的梯度符号正确；
  2. 可分加性偏置相互抵消（J = 0，损失 = softplus(1)），正确方向的交互使损失下降、反向的使损失上升；
  3. 重排候选并重映射 a / b 后，损失不变；padding 候选和非 a / b 候选的梯度为 0；交换视图后语义不变；
  4. 四题组被拆进不同批时拒绝；
  5. 侧表缺角、行重复、同一角出现两次、a = b、索引越界都拒绝；
  6. 默认关闭时与原 CE 运行相同（权重 0 时的损失序列与不加侧表完全一致）；C / X 的批序和第一步损失相同、之后不同；report 记录了公式版本；
  7. 损失非有限时在更新前失败，不留 model.safetensors；
  8. 梯度到达共享的 projector 和决策模型，不是一个游离的辅助头。
- `scripts/exp48/test_exp48.py`，3 项：侧表覆盖每行且只覆盖一次、1,152 个四题组都有完整四角；quads_of 拒绝缺角、不交换、候选顺序不同（非严格模式下计入排除）；报告的身份拒绝、条件 incomplete、G 读数复现、原图 tune 四题组分母（176 / 排除 274）。
- 回归：test_region、test_number_aux、test_objective、test_resume、test_pipeline 共 24 passed。
- chain 演练（假 uv）：启动检查通过，两臂各 7 项按序 completed，冻结记录 6 个集合；报告在假摘要处拒绝（预期）。
- pyright 0。没有运行真模型。

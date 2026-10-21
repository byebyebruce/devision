# 决策侧比较监督方案（DECISION-CMP-PREP-01 / -R1-01 / -R1-02；EXP57-PREP-01–03 的 CPU 实现已完成；未训练）

日期：2026-10-18。本文件不放行任何训练或真实前向。

## 问题定位（措辞按 R1-01 收窄）

EXP55 中，给足正确数字（金标数值表 G）后，读数题达 0.992，但温度比较只有 117 / 300；P / G 约 95% 选 "sample A"。只能说：**在这个接口下，给足正确数字后比较仍未建立**。没有排除模板、规则 / 算术或监督覆盖方面的原因，所以最小的可学性检验放在决策侧。

## 数据（`scripts/cmp/build.py` → `data/cmp/`；只用 real_layout questions_train）

- **全量 train 复算**：
  - 温度比较金标 A / B / neither = 2,244 / 3,000 / 756，选项位置 1,947 / 2,032 / 2,021；
  - train / tune / confirm 的族和 case 交集都是 0；3,000 个温度族的两个视图金标数值都改变；
  - same mass 轴 A / B 为 763 / 1,519，**各轴的标签并不完全平衡**；
  - "A 只占 37%"只能排除"总选 A"在这批原始数据上能得高分，**不能确定已训练模型偏置的来源**（这张表也不是模型实际的历史训练曝光）。
- **固定的一份小数据**（稳定族哈希 `sha256("cmp-v1:" + family)`，按轴比例分层）：
  - 600 个温度族 × 2 视图 × 5 题 = 6,000 题（4 读数 + 比较）；
  - 500 个磁铁族 × 2 视图 × 2 道读数题 = 2,000 题（不训练磁力比较，因为表里没有大小 / 极性）；
  - 合计 **8,000 题、2,200 行**（`train.jsonl` `b0f389da…`）。
  - 每行的 state 是原 state_text 加上它自己读数题金标数值的表（exp55 接口：`devision.reading.serialise / fmt`）。
  - 选项按 `random.Random(sha256(行 id + qid))` 固定换序，换序后比较题金标位置为 406 / 382 / 412。
  - 1,200 道比较题的金标都与该行金标读数推出的 m v² 比较一致（只在数据准备阶段核对，推理时不用公式）。
- **抽样后的比较分布**：
  - A / B / neither = 442 / 600 / 158；
  - 按行的轴：same mass A / B 146 / 304，same speed 152 / 152，both differ 144 / 144，equal 全部 neither 158。
  - 分层用的是族的一个视图的轴；轴是按行记录的，实际分布见上，各轴都有覆盖。
- **训练内 fit 切片**：从这 600 族中再按哈希分层固定 100 个完整族（`fit_slice.json`）：比较题 200 道，A / B / neither = 67 / 100 / 33；轴 same mass 69 / same speed 50 / both differ 48 / equal 33。运行前已固定，不按结果挑选。
- **真实批计划**（`PLAN.json`；rlcd 的代码路径：read_decision_jsonl → rlcd._items（EXP50-T 的 tokenizer / config）→ random.seed(1) → plan_batches(8)）：
  - 8,000 个 item，**1,000 次更新**，每批都是 8；
  - 4,000 个 (family, qid) 单元，每个单元正好 2 题（两个视图），没有被拆开的单元；
  - 每题曝光恰好 1 次，比较题 1,200 次，尾批 8。

## 训练方案（固定一套）

- **起点和冻结**：
  - 从 EXP50-T 的决策权重开始；
  - 冻结视觉塔、projector / 条件模块、选择器、字符头，以及选择器所用的 ModernBERT 词嵌入；
  - 只训练决策 LoRA（r 16）和 Laya 选项评分头；只用金标 CE，不用 RLCD、interaction 或 region。
- **参数**：LoRA lr 5e-5，头 lr 1e-5，预热 100，seed 1，1,000 次更新，固定末步，不按 tune 选择或延长（评论家已接受，不扫描）。
- **不加带图回放**：这是一次短的文字比较可学性检验，不是带保护证据的发布候选，也不称为"单变量证明"。
- **文字训练通路**：已实现为 `TrainConfig.text_only`，见下面的「EXP57 实现」一节（训练前另行评审）。

## 预写判据与分支（R1-02 已裁定）

- **fit（训练内学会）**：G 文字比较在 fit 切片上总体 ≥ 0.95，A / B / neither 的召回各 ≥ 0.90（raw 温度 1；训练末仍拟合 fit 温度，但不使用，见 EXP57-NOTE-01）。
- **G-tune（比较泛化）**：
  - 总体 ≥ 0.80，三类召回各 ≥ 0.60；
  - 相对同题冻结起点的 case 簇区间下界 > 0；
  - 分轴和两视图同时正确都报告（equal 轴全是 neither，不作为独立的额外证据）。
- **P-tune（自动事实表，取 EXP56 核实过的实时表）**：温度比较总体 ≥ 0.65，相对起点的配对收益区间下界 > 0。G 过线不能代替 P 过线。
- **读数保留**：G all ≥ 0.98，P all ≥ 0.866，原 EXP55 的 P_use 门槛照用；两对象 / 孪生另报。
- **短筛保护**：exist / science 相对 EXP50-T 各不低于 −0.02（点估计，区间另报，不宣称等效）；七保护、与 v0.3 的迁移和完整发布检查仍另有要求。
- **分支**：
  - fit 没学会：只约束这次的监督 / 优化预算；
  - fit 过而 G-tune 不过：泛化 / 规则应用没有建立；
  - G 过而 P 不过：检查读取错误与比较的耦合；
  - G / P 都过但保护失败：只考虑保护修复，不能发布。
  - 不在运行中加步、改门槛或换权重。

## 预算（单位已更正）

| 项目 | 依据 | 估计 |
|---|---|---|
| 训练 | 1,000 次更新（文字前向和反传，估计 0.3–0.5 秒 / 步） | 5–8 分钟 |
| fit 切片（G） | 200 行文字请求 | < 1 分钟 |
| tune G 和 P | 各 600 次请求、2,400 题，合计 1,200 次请求、4,800 题（EXP55 实测 6.1 个请求 / 秒） | 约 3.5 分钟 |
| 冻结起点同题基线 | 已有 EXP55（G）和 EXP56（P）的冻结记录，可复用则不重跑；不能复用时另加 1,200 次请求 | 0–3.5 分钟 |
| exist / science 保护（带图） | EXP50-T 实测 3:16 + 1:51 | 约 5 分钟 |
| 加载、保存、报告 | | 约 3 分钟 |
| **合计** | | **约 18–24 分钟** |

- 硬上限 45 分钟，不以缩减必要判据来兑现预算。

## EXP57 实现（EXP57-PREP-01–03；CPU 准备，未加载权重、未前向、未训练）

### 文字训练通路（`src/devision/train/rlcd.py`，`text_only`，默认关闭）

- 打开后每题都按 image = None 计算：
  - `_forward(pixels=None)` 调用的是 `model(input_ids, attention_mask, marker_pos, marker_mask, qtype)`，与 `Decider.decide(image=None)` 是同一个调用；序列仍由 `question_item` / `collate_items` 构造；
  - 不打开图片，不调用视觉塔 / `visual_inputs`，也不插入空白或零视觉 token。JSONL 里的 image 字段只作来源标识。
- 验证集评测（`_val_logits`）和训练末的温度拟合也走同一条通路。
- **拟温（EXP57-NOTE-01，评审已接受的偏离）**：rlcd 训练末仍在 fit 行上拟合温度，并存进 checkpoint 的 config。本轮全部判据都用 raw 温度 1：decide 在内存里把温度设为 1，exist / science 用 `--no-temperature`。这份温度不用于择优、改权重或任何门槛，不是独立的校准证据，以后部署也不能拿它代替正式校准。训练主干不为此改动，耗时计入原 45 分钟。
- **冻结**：projector、question conditioning（含局部视图几何）和 reader 都 `requires_grad(False)`；projector 参数组为空，conditioning 不进优化器；视觉塔照旧冻结，词嵌入在 LoRA 下冻结。**只训练**决策 LoRA 和决策头。
- **拒绝组合**：与 local_views、number_reader、region、grounding、aux、distill、vision / encoder 解冻、interaction、新的 question_conditioning 同用时直接报错。
- `text_only` 写进续跑键。默认的带图通路不变，起点模型的 local views / conditioning 配置原样保存。

### CPU tiny 测试（`tests/train/test_text_only.py`，13 项，全过）

模型形状与 EXP50-T 相同（question conditioning + local views，conditioning 随机非零）。

- **训练前向等于普通文字评分**：与 `decide(image=None)` 的概率一致到 1e-6，`_val_logits` 一致到 1e-5；`visual_inputs` / 视觉塔调用 0 次，图片路径都不存在。
- **一次更新只改允许的部分**：在优化器第一步之后逐张量比较。
  - 优化器里只有 lora_ 和决策头参数，变化的也只有它们；
  - 导出（LoRA 合并）后，vision、projector、condition、`decision.encoder.embeddings` 与起点逐位相同，encoder 层因合并而改变。
- **金标跟随选项**：换序后，金标随所在选项移动。
- **带图通路不变**：默认值为 False；同一份样本在默认设置下仍会去打开图片（FileNotFoundError）。
- **拒绝**：8 种冲突组合。
- **批次顺序**：训练实际拿到的批计划，等于 random.seed(seed) → plan_batches(_items) 预先算出的计划。
- **真实计划复核**：`_items` 未改。重跑 `scripts/cmp/build.py` 的 plan（只用 tokenizer / config），`PLAN.json` 与原文件逐字节相同（`c2288175…`）：8,000 题、1,000 次更新、4,000 个两视图单元、拆开 0、每题曝光 1。

### 实验编排

- **配置**（`configs/exp57.yaml`）：
  - init runs/exp50-t/stage2，data data/cmp/train.jsonl，val data/cmp/fit.jsonl；
  - text_only，ce，micro 8，LoRA r16，lr_lora 5e-5，lr_head 1e-5，warmup 100，seed 1，1 个 epoch，amp off；
  - eval_every 0，save_every 0，不设 evaluate。
- **fit 行**（`data/cmp/fit.jsonl` `e2d1d880…`）：fit 切片 100 族的 200 行，按 train.jsonl 中的顺序；`exp57.py fit-rows` 生成，start 时与重新计算的结果核对。
- **`scripts/exp57/exp57.py`**：
  - `start`（不加载模型）：绑定以下身份并写 START.json：
    - EXP50-T `1a8e5439` / `99c6246e`，及其已记录的 exist / science；
    - data/cmp 的 train `b0f389da`、fit_slice `7d348626`、PLAN `c2288175`（再核对 1,000 / 4,000 / 0 / 曝光 1）、MANIFEST `d4ab423d`、fit.jsonl；
    - EXP55：START `bf059567`、registry `b185564f`、inputs_G `a9c453c2`、details_G `ad792b3e`；
    - EXP56：START `ccdd23f2`、registry `84c24f2e`、report `70d235e1`、inputs_P `5792961f`、details_P `13b98a86`；
    - 两个 registry 里登记的每个文件；questions / rl_tune；YAML 的固定参数；代码哈希；各阶段请求数和预算。
  - `train`（MPS，watch 下）：`devision-pipeline configs/exp57.yaml`；从 `logs/stage2.log` 的 `step N` 行生成进度。
  - `freeze`（只读张量）：
    - 先核对训练正好 1,000 次更新；
    - 再逐张量与 EXP50-T 比较：vision / projector / condition / 词嵌入都相同，变化只在 encoder 层（LoRA 合并）和决策头；
    - 结果写 `export_check.json`，再写 FROZEN.json。
  - `decide`（MPS，watch 下）：
    - 先核对权重再加载，温度保持 raw，image = None；
    - 新模型依次跑 fit（200）、G（EXP55 inputs_G，600）、P（EXP56 inputs_P，600），再用 EXP50-T 跑 fit（200）；
    - 每个集合完成后立即登记，明细的身份里绑定输入 state 的哈希。
  - exist / science：devision-eval，`--no-temperature --max-questions 8`，与 EXP50-T 的设置相同；结束后 freeze.py record。
  - **保护成绩的身份（EXP57-R1-01）**：report 对新旧两份成绩分别核对以下各项，任何一项不符就拒绝，该保护记为缺项（incomplete），不算通过：
    - summary 的 checkpoint 是各自的 stage2（runs/exp57/stage2 和 runs/exp50-t/stage2）；
    - raw 温度、无对照、数据 sha；
    - 明细用 exp41 的 check_details 对源题单逐题核对：唯一、完整，gold / 图片、概率、correct 都一致；
    - 从核过的明细重算的 n / 准确率等于 summary 的 n / accuracy_all。
    点差和按图片的区间都从核过的明细算。EXP50-T 复算为 exist 697 / 736 = 0.9470108696、science 354 / 577 = 0.6135181976。
  - `report`（不加载模型）：
    - 温度比较：总体、A / B / neither 召回、分轴、两视图同时正确、预测分布；磁铁比较另报；
    - G / P 相对起点（EXP55 / EXP56 的冻结记录，经 read_condition 核对）、fit 相对 EXP50-T，都按 case 簇自助法给 95% 区间；
    - 读数保留：exp54 统计、read_right、P_use；
    - exist / science 相对 EXP50-T 已记录的结果，按图片配对给区间；
    - 门槛和分支见上；缺项记为 incomplete，不当作低分。
- **`run.sh`**：
  - 顺序：start → train → freeze → decide → exist → science → report；
  - 硬上限 45 分钟，最后 120 秒留给报告；train / decide 在 watch 下运行（首次无进度 5 分钟、之后 3 分钟、内存 1.5 GiB / 30 秒、ETA，deadline 比外层早 35 秒，TERM 后 10 秒 KILL）；
  - 只跑新目录，不续跑、不重试，后面不接任何步骤；任何一步失败都转到 `report --finalize`。
- **CPU 测试**（`scripts/exp57/test_exp57.py`，19 项，全过）：
  - start 遇到数据改动即拒绝，且不留 START；PLAN / YAML 检查；fit 行的取法；日志转进度；export 检查（任一冻结部分变化、无变化或张量名不同都拒绝）；
  - 比较统计和 case 簇配对；9 个门槛 / 分支用例；缺项为 incomplete，但 fit 不过的结论不受影响；
  - 用 stub 跑 run.sh：在 watch deadline 停止并 finalize，子进程组被杀；freeze 失败后不再启动 decide / eval。
- **真实文件上的不加载模型演练**（START 写到临时目录）：
  - 身份全部通过；请求数 fit 200 / G 600 / P 600 / base_fit 200，exist 736 题、science 577 题；
  - 报告在缺项时给出 incomplete；
  - 起点基线复现已验收的数字：G 比较 0.390（117 / 300），P 0.383（115 / 300），读数 G all 0.992、P all 0.886，A / B / neither 召回 0.965 / 0.047 / 0.027。

### 预算（45 分钟硬上限）

| 阶段 | 依据 | 估计 |
|---|---|---|
| 加载、step 0 验证（fit 1,000 题） | 文字前向 | 约 1 分钟 |
| 训练 1,000 次更新 | 文字前向和反传，0.3–0.5 秒 / 步（未实测，watch 按 ETA 管控） | 5–8 分钟 |
| 末尾验证、拟温（不使用）、合并、保存 | | 约 1–2 分钟 |
| freeze（逐张量比较） | 两个约 1.6 GB 的文件 | < 1 分钟 |
| decide：1,600 次文字请求 | EXP55 实测约 6 个请求 / 秒 | 约 5 分钟 |
| exist / science | EXP50-T 实测 3:16 + 1:51 | 约 5 分钟 |
| 报告 | | < 1 分钟 |
| **合计** | | **约 19–23 分钟** |

- 新 checkpoint 以后要和读取组件（选择器 / 字符头）重新绑定（bundle 绑定的是决策 checkpoint 的哈希）。本轮 P 用的是 EXP56 固定的实时表，不重新读取。


## EXP57 第2轮验收与运行许可

2026-10-18 22:09:30 +08:00，评论家：02c308b 局部验收通过，EXP57-R1-01及NOTE-01收口。放行EXP57-RUN-01唯一一次原配置文字CE训练与既定终点评测：1,000更新，启动+45分钟上限，最后120秒报告，全部判据raw=1。许可不代表已经启动；完整记录归档在[长期交付记录](2026-10-17-v04-delivery-plan.md)。本轮未授权发布或自动接续训练。

## EXP57 结果（22:09:55–22:41:47，完整运行；fit 未过，按预写分支判定）

- **运行**：
  - HEAD 02c308b，START 4e9fe57c，report e5a8565d；
  - 全部阶段退出码 0，用时 31:52（上限 45 分钟），watch 没有触发停止；
  - 训练每步约 1.0 秒，比估计慢 2–3 倍；decide 4.7 个请求 / 秒。
- **端点**：
  - 1,000 次更新；FROZEN 962503fb / 09a5b447；
  - export_check：432 个张量中，228 个冻结张量全部未变，变化只在 encoder 层（LoRA 合并）、决策头、scorer、type_emb。
- **温度比较**（raw 1；括号内为 A / B / neither 召回）：

| 集合 | EXP57 | 起点 EXP50-T | 配对差（case 簇 95% 区间） | 两视图同时正确 |
|---|---|---|---|---|
| fit（训练内，200 题） | 0.875（0.776 / 0.910 / 0.970） | 0.345 | +0.530 [+0.445, +0.610] | 0.80 |
| G-tune（300 题） | 0.817（0.699 / 0.860 / 1.000） | 0.390 | +0.427 [+0.356, +0.494] | 0.68 |
| P-tune（300 题，EXP56 实时表） | 0.757（0.637 / 0.840 / 0.784） | 0.383 | +0.373 [+0.302, +0.443] | 0.587 |

- **分轴**：fit 上 both differ 0.688、same mass 0.928、same speed 0.920、equal 0.970；G 上 both differ 0.742。「总选 A」消失：G 上的预测为 A 100 / B 163 / neither 37，起点是 A 284。
- **读数保留**：G all 1.000，P all 0.876（门槛 0.866；起点 0.886），P_use 通过。
- **保护**：exist −0.001 [−0.004, +0.000]，science −0.014 [−0.036, +0.007]，都在 −0.02 以内。
- **门槛**：fit False（0.875 < 0.95，A 召回 0.776 < 0.90）；G_tune、P_tune、reading、screens 都是 True。
- **预写分支**（fit 优先）：fit 没有学会，只约束这次的监督 / 优化预算，不下路线层面的结论。EXP58 的启动条件（EXP57 全部通过）不成立。
- **待评审裁定的事实**：预写分支没有单列「fit 未过而 G / P 过线」这种组合；缺口集中在 A 类和 both differ 轴（要按 m·v² 比较）。本轮没有按 tune 选择，也没有追加步数。

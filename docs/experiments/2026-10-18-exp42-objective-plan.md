# exp42 方案：同起点、同数据、同预算的 RLCD+CE / 纯 CE 对照（CPU 准备完成，未运行，等评论家的模型许可）

日期：2026-10-18。依据：评论家 EXP42-PREP-01（EXP41 终点复核之后）。

## 问题

在当前起点、短预算下，RLCD 的随机扰动策略项会不会妨碍金标监督学会按对象读数？

- 这不是"还没有用强化学习"：`rlcd._rlcd_loss` 一直是随机扰动 logits 的 RLCD 项加 gold CE。
- 本实验不宣称 RLCD 普遍无效，也不是在定位根因。
- 它是集中拟合诊断：不混入新的回放、新字号、新损失或新数据源，不用 tune / confirm 训练，不碰新冻结确认集。

## 两臂（唯一的臂间变量是 objective）

| | R | C |
|---|---|---|
| objective | `rlcd`（现有 RLCD 项 + gold CE） | `ce`（只用 gold CE） |
| 起点 | exp40-M 冻结终点 `runs/exp40-m/stage2`（`3fb4232e…` / `74e32e7c…`），两臂各自独立初始化 | 同左 |
| 配置 | `configs/exp42-r.yaml` | `configs/exp42-c.yaml`（与 R 只差 name 和 objective） |

- **objective 开关**（`src/devision/train/rlcd.py`，最小改动）：
  - `TrainConfig.objective`，默认 `"rlcd"`；不传这个参数时行为与以前完全相同。
  - `"ce"` 时只返回 gold CE（对真实选项做 log_softmax，padding 排除），与 RLCD 分支里的 CE 是同一个式子。不用 sigma = 0 来模拟（原公式要除以 sigma²）。
  - CE 分支照样抽取同样的 RLCD 噪声，所以两臂的随机数消耗、批次顺序和第一步完全相同。
  - objective 写进 TrainConfig（SwanLab 配置、resume 键）和 train_report.json；chain 另写进每臂的 START.json。
  - 其他值拒绝。
- **相同的设置**：结构、图像入口、可训参数、学习率、种子、数据顺序、有效 batch、更新数、温度口径。
  - 学习率：projector 2e-5、head 1e-5、LoRA 5e-5、视觉塔顶 4 层 5e-6；micro / eval batch 8，amp off，seed 1，warmup 100，epochs 1，固定 576 次更新。
  - 固定终点，不挑最好的一步，不看趋势自动加时。
- **批计划**：在第一次更新之前、由 `random.seed(1)` 决定，与 objective 无关，所以两臂相同。

## 数据（`scripts/exp42/build.py` → `data/exp42/`；MANIFEST.json）

- **来源**：exp40-M 自己的真实版式训练行（`data/exp40/train_M.jsonl`，`1afe9d2d…`）。
- **单元**：读数单元 (family, field)，要求两个对象在 v0 视图上的 gold 不同。
  - 可选单元数：质量 160、速度 153、距离 113，与评论家核的一致。
  - 每个字段按 sha256("family:field") 取前 48 个，共 144 个单元。
  - 每个单元完整保留 8 题：A/B × v0/v1 × 原图/L。题目、选项、gold 和图片都与 exp40 相同，L 图的数字在 exp40 中已全量核过。
  - 共 1,152 题；不按任何模型输出选样。
- **重复**：同一 epoch 内重复 4 次，group 写成 "r<k>:<family>"、id 加 ":r<k>"，所以每个 (group, qid) 单元仍恰好 8 题。共 4,608 次呈现、576 次 8 题的更新。
- **批计划核对**（`scripts/exp42/plan_check.py` → PLAN.json）：走真实的 read / _items / plan_batches 路径。
  - 4,608 个 item，576 个满 8 的批，576 个单元，最大 8，拆开的为 0。
  - 每个批正好是一个单元（同字段的 A/B × 两视图 × 原图和 L）。
- **文件**：

  | 文件 | 内容 | sha256 |
  |---|---|---|
  | train | 训练 | `27e57372…` |
  | fit_O | 576 道原图题，训练拟合读出 | `73af6056…` |
  | fit_L | 576 道 L 图题，训练拟合读出 | `1160408a…` |
  | val_sub | r13b val 每 4 行取 1 行，973 题 | `1903baea…` |

- **val_sub 是预算导致的监测缩减**：训练中只在第 0 步、末点和合并后各评一次 val_sub，用来监测和拟合温度。主读出全部用温度 1，不受拟合温度影响。用完整的 3,892 题 val，每次约 7 分钟（按 exp40 实测推算），总时长会超过 90 分钟（见下面的预算）。

## 读出（`scripts/exp42/report.py`；运行前写定）

- **口径**：全部用 devision-eval `--no-temperature`（温度 1），走 decide 的真实路径。
- **评测对象**：
  - 起点：exp40-M 冻结终点，只评一次，两臂共用；
  - R、C 的终点。
- **集合**：
  - fit_O / fit_L：训练过的题，只代表训练拟合，不代表泛化；
  - rl_tune（原图）/ rl_tune_L（L 图）：600 张 tune 图，没训练过；
  - exist / science：只报告风险，与起点比较。
- **逐字段统计**（质量 / 速度 / 距离）：
  - different：异值题的单题准确率；
  - both：异值时两对象同时对；
  - twin：该对象的 gold 在两视图之间改变时，两视图都对；
  - same_answer：异值时 A / B 答案相同的比例。
  - 字段等权宏平均。
- **差值**：C − R、R − 起点、C − 起点；按 case 连接簇重抽（exp40 的算法，seed 0，重抽 2,000 次）。
- **身份**（不满足就拒绝）：
  - exp40-M 仍是冻结身份；
  - 每臂 START.json、run.json、train_report.json 的 objective 与本臂一致，起点和数据 sha 正确；
  - 每个评测文件都记录在本臂冻结的权重 / config 下；
  - 摘要的 checkpoint、temperature `none (raw)`、control none、data_sha256 都正确；
  - 逐题无漏无重，gold / image_id 一致，概率有效，prediction = argmax，correct 一致（复用 exp41 的 check_details）。
  - 缺 run 或缺集合时记为 incomplete，不给通过。
- **预写条件**（这是下一阶段的投入门槛，不替代发布门槛）：
  - **进步**（每臂）：原图 rl_tune 上
    - 异值读数宏平均 ≥ 0.60；
    - 异值两对象同时对 ≥ 0.20，且相对起点的差值下界 > 0；
    - 孪生两视图同时对 ≥ 0.20，且相对起点的差值下界 > 0。
    - 每个字段和分母同时报告；不能靠同值题过线。
  - **归因于 CE**：原图 rl_tune 的两对象同时对 C − R ≥ 0.10，且下界 > 0。如果两臂都进步，首先解释为集中监督这个配方本身有效。
  - **停止**：两臂在 fit_L 上的异值读数都 < 0.60，只说明这个起点、这组参数、576 步没有学会，于是停掉这条 objective 对照路线。这不证明特征缺失，也不证明需要更大的模型。
  - **只有 fit 或只有 L 有收益**：不接同样配方的原图候选长训。拟合成功但不泛化的情况单独列出；接近阈值的记为不确定，不自动重试。

## 运行、守卫与预算（`scripts/exp42/chain.sh`）

- **顺序**（按优先级）：
  1. 起点评 fit_O / fit_L / rl_tune / rl_tune_L；
  2. R 训练（受 watch 监控）→ 冻结 → 同样 4 集 + exist / science，每个文件都记录在冻结身份下；
  3. C 同上；
  4. 起点的 exist / science；
  5. 报告。
  - 一臂失败不影响另一臂。不重试，跑完不接任何别的运行。
- **启动前检查**：没有其他模型进程；exp40-M 冻结检查通过，sha 正确；4 个数据文件的 sha 与 MANIFEST 一致；PLAN.json 存在；运行目录原本不存在。
- **截止**：启动后 90 分钟，最后 120 秒留给报告。各步用 run_capped，到点 TERM，15 秒后 KILL。
- **watch**（`scripts/exp42/watch.py`）：复用 B 的 watch，只换阈值。`scripts/v04b/watch.py` 的 main 加了 val_windows 参数，默认值不变。
  - 首窗或相邻两个窗口 > 3 秒 / 更新 → 停止（exp40 实测 1.45–1.53）；
  - MPS > 16 GB 且可用内存 < 2 GB → 停止；
  - 停滞上限：首窗前 20 分钟，第 500 窗口之后 20 分钟，其余 10 分钟。
- **预算**：按 exp40 实测推算，首窗出来后再按实测重算。exp40 每轮验证约 11.5 分钟，其中 r13b val 约 6.8 分钟；exp32 的 rl_tune 评测约 2.3 分钟。

| 部分 | 估计 |
|---|---|
| 起点 4 集（rl_tune 2.3 + rl_tune_L 2.3 + fit 2 × 0.8） | 约 6 分钟 |
| 每臂：加载 1 + val_sub 3 × 1.7 + 576 × 1.5 秒（约 14.4）+ 保存 1 | 约 21.5 分钟 |
| 每臂评测：4 集约 6 + exist / science 约 1.5 | 约 7.5 分钟 |
| 起点 exist / science + 报告 | 约 2.5 分钟 |
| 合计 | 约 67 分钟（上限 90，余量约 23 分钟） |

- 如果沿用完整 r13b val，每臂约多 15 分钟，合计约 97 分钟，超过上限。所以只缩减监测（val_sub），两臂更新数和主读出都不变。
- 超时的话，起点的 exist / science 在最后、优先级最低。

## CPU 验证

- `tests/train/test_objective.py`，5 项：
  1. 默认就是 rlcd，与显式写 rlcd 逐位相同，且含策略项；
  2. ce 的数值和梯度等于标准 cross_entropy（逐题只取真实选项，含 padding），padding 选项上的梯度为 0；
  3. 两种 objective 调用后 torch 的随机数状态相同；
  4. 未知的 objective 被拒绝；
  5. tiny 训练：默认与显式 rlcd 的每步损失和批次顺序相同；ce 的批次顺序和第一步损失与 rlcd 相同、之后不同；report 记录了 objective。
- 已有的 test_distill / test_resume 照常通过。
- `scripts/exp42/test_exp42.py`，3 项：
  1. 数据就是 exp40 的行（只多了 id / group / repeat），144 个单元按固定哈希序取前 48，单元恰为 8 题、原图和 L 各 4 题，v0 的两对象 gold 不同，重复的 group 命名正确；
  2. 用假的温度 1 记录跑报告：读出、分母（rl_tune 异值质量 276 题、速度两对象 191 对、距离 85 对）、簇差值和预写条件；
  3. 错臂（R 的身份写着 ce）被拒绝；缺臂或缺集合记为 incomplete。
- chain 演练：临时目录 + 假 uv，不跑模型。启动检查通过；起点 6 项、两臂各 7 项按序 completed；两臂冻结记录齐全；报告通过臂的身份检查后，因为假摘要没有 data sha 而拒绝（预期）。
- `scripts/v04b`、`scripts/exp40` 的测试照常通过（watch 加参数后行为不变）。pyright 0。

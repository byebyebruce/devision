# 面板数字绑定的直接诊断（PANEL-BIND-PREP-01；CPU 准备完成，未运行模型）

日期：2026-10-20。评审 PANEL-D1-RESULT-01 / PANEL-BIND-PREP-01 选择评审意见中的方案 (c)。不训练，不算一轮。worktree `devision-panel-bind`，分支 `panel-bind-prep`（从 db03f03 拉出）。

## 问题

EXP74 D1 的混合补词差值从 0.029 升到 0.250（门槛 0.30，未过）。这个差值混合了数字和其余文字，不能说明 A、B 两端是否各自读出了完整数值。这里直接拆开：D1 的提升里，异值的两端完整数值占多少？与历史 EXP26 阶段 A 相比差在哪里？据此决定下一笔训练投入。

## 设计（`scripts/panel_bind/probe.py`）

- **端点**（串行，同题同设置）：
  - 冻结起点 `runs/v04-k/cand`（model 185b9496…）；
  - D1 `runs/exp74-panel-d1/d1`（f2efe49d…）；
  - 历史 `runs/exp26-align/stage1`（822eade1…，阶段 A，不是阶段 B 的 5,000 步）。
  - 历史端点只作参考，不是同起点的因果对照，成绩也不借给当前候选。
- **题单**：`data/exp26/align_val.jsonl` 全部 1,000 张（6a36f5bc…）：1,000 个族，A ≠ B 837 张，A = B 163 张。不加题、不挑题、不删题。
- **路径**：align 自己的 caption 路径。align._mlm_logits 只用整图，ModernBERT MLM 头与训练时相同（缓存的 answerdotai/ModernBERT-large），batch 16，caption ≤ 40 token，不经过 conditioning，不引入 oracle 区域或候选答案。
- **模式一：原随机遮罩复现（random）**：逐步照抄 align._evaluate（每种模式 Random(1)、rng.choice 取 caption、_masked_batch 遮罩 0.5、错图取同一批 16 张里的下一行）。每个目标 token 标为"数字"（落在 A / B 数字跨度内，含分隔符）或"其余"；数字和其余的分母、命中数相加必须还原总数（夹具已核）。三个复现参照：
  - 起点对 D1 step 0（0.4414 / 0.4128）；
  - D1 对 D1 step 1,250（0.9320 / 0.6821；保存时合并了 LoRA）；
  - EXP26 对 exp26-align step 2,500（0.9755 / 0.6282）。
  - 容差：真图、换图两个准确率都要在 ±0.003 以内（覆盖 LoRA 合并带来的数值差和 MPS 的浮点差）。超出就把该端点判为 invalid，不宣称完成定位。
- **模式二：数字全遮（numbers）**：A、B 两个数字的全部 token（含分隔符）同时遮掉，其余文字保留。真图和同一张固定错图（同上：同批的下一行）用完全相同的输入和遮罩。
  - 解析式只适用于 "A: <量> <数> <单位>. B: <量> <数> <单位>."（不套用 real_layout 的 regex）。
  - 无法解析、token 与数字字符不能精确对应、超过 40 token 的 caption，在任何前向之前就拒绝（夹具核过真实 tokenizer 下 1,000 条 caption 全部精确对应）。
  - 全遮仍会泄露数字的 token 长度：这只是受约束的 MLM 诊断，不是 decide 或产品读数的成绩。
- **报告**（`scripts/panel_bind/report.py`，不加载模型）。每个端点、每种图片模式：
  - 数字 token 准确率；A 完整、B 完整；
  - A ≠ B 时：两端同时完整正确、两端预测相同、A / B 对调、错误的一端等于另一端的 gold；
  - A = B 单列。
  - 按 family 重采样 2,000 次（seed 0）给配对区间：真图减错图（A ≠ B 两端都对）、D1 减起点、D1 减 EXP26（同题）。
  - 披露错图里量名相同的比例：错图支持的是一般的依图性，不能单独隔离标签绑定。
- **启动记录**（START.json，前向之前写出）：各端点 model / config / tokenizer 的 SHA；MLM 头文件 SHA；题单和 1,000 张图实际字节的摘要；代码 SHA（probe、report、align）；错图映射；复现参照和容差。逐行明细（random.jsonl、numbers.jsonl）可以不用模型复算统计。

## 报告先核证据（PANEL-BIND-R1-01 / R1-02）

- 冻结题单来自 START：核对题单 sha 等于 START，并重新推出错图映射，必须等于 START 记录的映射。START 缺失时报告记为 incomplete（没有身份，不给结论）；START 截断或不符记为 invalid。两种情况都照样写出 report.json。
- 每个端点：summary、random、numbers 三个文件逐个读取。缺文件记为 missing / incomplete；空文件或截断记为 invalid，不会崩溃。
- random：(id, mode) 必须每个恰好一次，与题单 × {真图, 错图} 完全相同；family 和实际图片等于冻结映射；token 计数合法。由逐题计数重算数字、其余和总体的命中、分母和准确率，必须与 summary 完全一致，否则 invalid；复现判定只用重算值，并报告与参照的实际差值（不称逐位复现）。
- numbers：(id, mode) 同样完整唯一；family、图片、gold 等于冻结题单；token_right 非空且全为 bool；exact 是显式 bool，并且等于 all(token_right)；pred 是字符串。
- 只有通过全部检查的端点才进入统计和跨端点配对；其他已完成的端点照常保留。summary 和 report 都先写临时文件再原子替换。
- 口径（NOTE-01）："exact" 预先定义为数字跨度的全部 token 都正确（字段名 `*_exact_all_tokens`）；解码后的字符串与 gold 相等只作描述（`*_string_equal_descriptive`），不替代原口径。

## 状态和失败收尾

- 每个端点：completed / missing（没跑完或被停止）/ invalid（复现超出容差或行数不全）。总体：全部 completed 才是 completed，有 invalid 就是 invalid，否则 incomplete。已完成的端点照常报告。
- runner（`scripts/panel_bind/run.sh`）：只启动一次（`runs/panel-bind.started`），串行执行 START → 三个探针 → 报告。总上限 1,200 s，每个探针必须在截止前 120 s 结束（TERM，15 s 后 KILL），最后 120 s 留给不加载模型的报告。某个探针失败或被停止，该端点记为 missing，报告照样运行。启动前用 ps 完整命令行检查没有其他模型进程。

## 预算（按 D1 实测核算）

D1 总共 47.5 分钟：1,250 步约 2.0 s/步，约 41.7 分钟；其余约 5 分钟是 3 次验证、一致性检查和加载，每次验证（真图、换图各 63 批）约 1.3 分钟，约 0.6 s/批。每个端点要跑 random 126 批和 numbers 126 批，约 2.6 分钟，加上加载约 0.5 分钟；EXP26 没有局部视图，会更快。START 要对 3 个约 2 GB 的权重和 1,000 张图算哈希，约 0.5 分钟。合计约 **9–11 分钟**，目标 10–15 分钟，**上限 20 分钟**（含 120 s 报告）。

## 解释的边界（运行前写下，不看结果改阈值）

- 数字（特别是异值的两端完整数值）基本没变、改善主要在其余 token：下一方案应加强直接的数字 / 标签监督，不以混合准确率 0.932 为理由再盲加同配方遍数。
- D1 的两端读数已接近历史阶段 A，且依图有效：说明混合门槛与直接读数不同步，可以另立有直接读数依据的决策诊断；EXP74"未过 0.30"的结局永不改写，原 D2 也不绕过守卫启动。
- 数字在改善但明显落后历史参考：再讨论一次有边界的补充对齐。那是重新初始化优化器的额外训练，不能称为"精确的第二遍"（align 只保存合并后的模型，没有 optimizer / RNG 状态）。
- 历史参考在全遮下也弱：这个探针与阶段 B 能否学会的关系不足，不能把探针低分外推成路线不可学。

## 验证（CPU，已完成）

`scripts/panel_bind/test_panel_bind.py` 19 passed（R1 之后）：
- 报告证据核对 11 项：完整 → completed；缺 random → incomplete；保持行数不变的 shuffled 重复、错图映射不符、gold 不符、summary 与明细矛盾、exact 与 token_right 不一致、summary 截断、numbers 截断 → invalid。其余端点都保留为 completed，未核过的端点不参与配对，report.json 总会写出。另外 START 缺失 → incomplete，START 截断 → invalid，两者都写出报告。
- 原有：
- 真实 tokenizer 下 1,000 条 caption 全部精确解析；
- 3 种拒绝（缺句点、顺序错、不是数字）；
- tiny 模型上 random 模式与 align._evaluate 逐位相等，数字与其余的分母、命中数还原总数；
- numbers 模式的遮罩数等于两段数字的 token 数，错图是固定的循环映射；
- 统计（对调、同答、错一端等于另一端 gold、同值行）；
- 报告状态（missing → incomplete，复现超出容差 → invalid）；
- runner 的顺序、只启动一次和上限。
pyright 0。没有跑全仓测试。

## 结果（PANEL-BIND-RUN-01，2026-10-20 00:44:59–00:56:25；总体 invalid，D1 和 EXP26 两个端点有效）

- 起点：换图复现 0.4060 对参照 0.4128，差 −0.0068（分母 6,896 中的 47 个目标 token），超出 ±0.003，判 invalid，不计算 D1 − 起点，也不用描述值补出训练收益。原因尚未定位。
- D1 和 EXP26 的 random 复现与各自的训练报告完全相同。
- 异值 837 族，严格口径：
  - 两端都对：D1 196/837 = 0.234，EXP26 580/837 = 0.693；
  - 两端同答：D1 236/837 = 0.282，EXP26 7/837；
  - 对调：D1 37/837，EXP26 0；
  - 错的一端恰为另一端 gold：D1 236/1,032 = 0.229，EXP26 3/356。
  - D1 真减错 +0.234 [0.207, 0.264]；EXP26 +0.693 [0.662, 0.723]；D1 − EXP26 −0.459 [−0.493, −0.424]。
- **解释边界（PANEL-BIND-RESULT-01 / 02 更正）**：
  - 缺口同时包括数字读取和 A / B 分配，不能只归到其中一个。D1 有明确的依图收益，同答和串到另一端的错误明显偏多；但约 77% 的错端并不等于另一端的 gold，并且真图上 A、B 完整正确只有 0.435 / 0.412，所以"数字都能读出，只是放错了位置"超出了证据（更正上一份终点回执里的说法）。
  - 混合准确率 0.932 被非数字部分显著抬高。但混合真减错差 0.250 中，数字贡献 808/6,896 = 0.117，其余贡献 915/6,896 = 0.133；其余 token 里也有量名和单位，不能把这部分直接等同于"纯模板作弊"（更正"模板抬高"的说法）。EXP26 的数字真图 1,376/1,545 = 0.891，错图同为 268，直接数值差距确实存在。
  - EXP26 是不同初始化和训练历史的参考，不是单因素对照；错图只有 16.4% 量名相同，真减错不能全部称为纯标签绑定；全遮仍泄露数字的 token 长度，成绩是受约束的 MLM 读数，不是普通 decide 或发布成绩。

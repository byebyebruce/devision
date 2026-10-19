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

`scripts/panel_bind/test_panel_bind.py` 9 passed：
- 真实 tokenizer 下 1,000 条 caption 全部精确解析；
- 3 种拒绝（缺句点、顺序错、不是数字）；
- tiny 模型上 random 模式与 align._evaluate 逐位相等，数字与其余的分母、命中数还原总数；
- numbers 模式的遮罩数等于两段数字的 token 数，错图是固定的循环映射；
- 统计（对调、同答、错一端等于另一端 gold、同值行）；
- 报告状态（missing → incomplete，复现超出容差 → invalid）；
- runner 的顺序、只启动一次和上限。
pyright 0。没有跑全仓测试。

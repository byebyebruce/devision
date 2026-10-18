# exp47 方案：保持共同分量的对象残差差分对照（冻结 EXP45-G；CPU 准备完成，未运行，等评论家的模型许可）

日期：2026-10-18。依据：评论家 EXP47-PREP-01..03（EXP46 终点之后）。不训练，不改权重，不改 EXP46 的代码和产物；oracle 不是产品输入。

## 要区分的解释

EXP46 中，正确框没有让答对数增加；但在原图 tune 上，正确框对错误框的 gold 概率有微弱的正向差（+0.0038，对象方向对比 C +0.0075，9/12 个单元为正）。所以只能说有效的对象决策信号很弱，不能说分支完全没被使用。

本实验区分三种解释：

1. 有用的对象残差差分太弱；
2. 差分能传过去，但不是可用的对象数值；
3. 差分在融合或读出阶段被压掉了。

不扫倍率，不换样本。

## 设置（`scripts/exp47/diagnose.py`）

- **固定对象**：
  - G：`runs/exp45-g/stage2`（`c34b43dd…` / `2a5dfa64…`）；
  - 数据：EXP46 的 48 个单元 / 96 题（`ad68597e…`）、MANIFEST（`c877f70e…`）、EXP45 readout sidecar；
  - 参考输出：EXP46 已验收的 details（`4d7fe4a2…`）和 report（`1ec8da99…`），run.sh 启动前核对。
- **残差差分**：同一问题、同一 V 上，用 EXP46 的正确框和错误框掩码分别得到 r_T = out(attend_T)、r_O = out(attend_O)。
  - c = (r_T + r_O) / 2，d = (r_T − r_O) / 2；用的是真实残差张量，不是 EXP46 的 A/B 查询差。
  - β = max(1, min(1000, 0.1 × RMS(V + c) / max(RMS(d), 1e-12)))，在单元的两问上共同计算，不依赖 gold、概率或对错。
- **六种模式**：只替换模块返回的视觉 token。

  | 模式 | 视觉 token |
  |---|---|
  | baseline | V + r |
  | target_1 | V + c + d（= V + r_T） |
  | other_1 | V + c − d（= V + r_O） |
  | center | V + c |
  | target_gain | V + c + βd |
  | other_gain | V + c − βd |

- **保真**（任一项不通过，该单元就停止、不跑新模式，阈值不放宽）：
  - 不打补丁的前向对 G 的冻结记录和 EXP46 的 baseline；
  - 插桩 baseline 对不打补丁的前向；
  - target_1 / other_1 对 EXP46 的 oracle_target / oracle_other；
  - 以上都要求有限、最大差 ≤ 1e-5、预测一致。
  - 只在 with 块内替换，模式之间恢复；结束时核对模块权重没变。
- **逐题记录**：概率、预测、对错，以及 margin = P(本题 gold) − P(另一对象的 gold)。
- **逐单元记录**：RMS(V)、RMS(c)、RMS(d)、RMS(V + c)、β、是否触顶、是否零差分、施加后的相对差分。
- **沿路径**（target vs other，β 和 1 各一组；同一次前向，没有另起前向）：模块输出、编码器选项 marker 位置的隐藏状态（return_hidden，位置按 64 个视觉 token 偏移）、按题去均值后的选项 logits。每处都记半差 RMS 和共同 RMS。
  - 这些范数只用来定位响应在哪一段衰减，不证明信息可解码。
  - 不另算 QK，不保存整层激活。

## 读出与预写条件

- **对比量**：C_β = [P_target_gain(gold) − P_target_gain(另一对象 gold)] − [P_other_gain(gold) − P_other_gain(另一对象 gold)]，先在同图两问上取平均；C_1 用 target_1 / other_1 同样计算。
  - 按 fit / tune × 原图 / L，以及字段，报告 C_β、C_1、C_β − C_1 的均值，和正 / 负单元数；
  - 同时报告各模式的答对数、两问都对、同答，以及相对 baseline 的错→对、对→错。
- **继续条件**（只决定是否值得做小预算的训练对照，不是 v0.4 门槛）：
  - 原图 tune 上 target_gain 两问同时正确 ≥ 3/12 个单元，且比 other_gain 多 ≥ 2；
  - 原图 tune 的平均 C_β − C_1 ≥ 0.05；
  - 原图 fit 的该差值 > 0；
  - 原图 tune 三个字段中至少两个为正。
  - 只有 L 图有效，不算原图路线通过。
- **不满足时**：不加 region_weight，不扫 β。按 入口 → 选项位 → logits 的实际响应，判断下一份短训练方案的重点放在融合 / 读出，还是区域值表征。阴性不能单独排除可读性或干预分布外。

## 运行与预算（`scripts/exp47/run.sh`）

- **启动前检查**：没有其他模型进程；G 冻结；units sha；EXP46 已验收的 details / report sha；runs/exp47 原本不存在。
- **截止**：启动后 10 分钟，最后 60 秒用于无模型的 `--finalize`；一次有界运行，不重试。
- **前向次数**：每个单元 1 次不打补丁 + 6 次模式 = 7 次 batch 2，共 336 次 batch 2（672 次题目前向）。r_T、r_O 是在模块内多算两次注意力，不是额外的模型前向。EXP46 同样规模实测约 53 秒，估计 1–2 分钟。
- **收尾容错**：status.json 先写临时文件再改名（不会留下半写的文件）；finalize 缺目录时自动创建，并跳过半写的最后一行；中途停止时为 incomplete，并写出各模式的缺项数。
- **START**：绑定权重、输入核对结果、EXP46 details 的 sha 和脚本的 sha。

## CPU 验证（`scripts/exp47/test_exp47.py`，6 项）

1. 六种模式的公式：baseline 等于模块自身输出；target_1 = V + r_T，other_1 = V + r_O，center = V + c；gain 的 β 符合公式；交换 target / other 后 β 不变，结果等于另一个方向。
2. β = 1 时，gain 模式还原为端点。
3. return_hidden 不改变 logits；隐藏状态长度 = 文本长度 + 64。
4. tiny 上完整跑通：先用 exp46 自己的 tiny 流程生成参考输出，再跑 exp47 → complete；逐单元字段和沿路径的统计都有。
5. 篡改 EXP46 的 oracle_target 参考值后，target_1 保真失败，单元停止，各模式完成数为 0，incomplete。
6. finalize 能处理缺失的目录和半写的 status / details，给出 incomplete。

pyright 0。没有运行真实 G。

# 短复现：保存后的 N 端点错图准确率为什么与训练时不同（MLM-REPRO-PREP-01；CPU 准备完成，未运行模型）

日期：2026-10-20。worktree `devision-mlm-repro`，分支 `mlm-repro-prep`（从 6764043 拉出）。不重训 N，不改 ±0.003，不覆盖 N / report / compare，不补写成预先计划的归因。

## 症状

保存后的 N 端点（3ab0ab03…）经探针：真图 0.962587，与训练摘要相等；错图 0.669519，训练摘要（align_report a24ff10b…，step 1,250）是 0.680539，差 −0.011021（6,896 个目标中的 76 个命中）。起点端点在 PANEL-BIND-RUN-01 里有相似的症状。D1、EXP26、R 的两路都相等。训练时没有逐 token 记录，所以只能说"真图聚合准确率相等"。

## 设计（`scripts/mlm_repro/repro.py`）

在同一个进程里加载一次保存的 N 端点：同一 tokenizer、MLM 头、设备、图片顺序、遮罩（batch 16、每种模式 Random(1)、遮罩 0.5、错图为同批下一行）。依次跑三遍：

- **A**：align._evaluate，也就是训练时调用的那个函数；
- **B**：已验收的探针路径 probe.run_random；
- **C**：带记录的同一循环。每批记录 input ids / attention / targets / 实际图片张量的 SHA 和这一批的行。在**同一份 logits** 上做三种计分：
  - (1) align 的方式：先取 mask 位置，再在设备上 argmax；
  - (2) 探针的方式：先对整份 logits 在设备上 argmax，再在 CPU 上取 mask 位置；
  - (3) CPU float32 参考：把 mask 位置的 logits 拿到 CPU 再 argmax。
  - 三者的命中分别记录；只要三者有不一致的 token，就记录 gold、三个预测、CPU 上 top-2 的差和非有限值个数。
  - 每次只保留一批的 logits（约 16 × 41 × 50k 个 float），不缓存全量词表 logits。

## 第 1 轮修正（MLM-REPRO-R1-01 / R1-02）

- **身份先行（加载任何东西之前）**：复用已验收的 EXP75 START（runs/panel-number/START.json 1827f65d…），核对 N 的 model / config / tokenizer 树、缓存的 MLM 头文件、题单 SHA、1,000 张图的实际字节摘要和固定错图映射。任何一项不符：写 identity.json（ok false 和原因），exit 12，不加载模型。runner 里原有的 model / align_report 检查保留。identity.json 同时记录本次的代码 SHA（repro、probe、align）。
- **A / B 的实际调用证据**：A（align._evaluate）和 B（probe.run_random）运行时，在 align 模块上临时包装 _masked_batch / _logits_in_parts（运行结束后恢复，不修改 align.py 或 probe.py 源码）。每批记录实际 input ids / attention / targets / 图片张量的 SHA 和按 align 方式计的命中，与 C 逐批对齐；仍然只做 378 次前向。C 的不一致记录增加批内行号和 token 位置；align 方式的命中同时保留设备上和 CPU 上的计数。
- **逐段落盘与无模型收尾**：A、B 各自结束时立即写 batches.jsonl 和原子写入的 summary.json；C 逐批追加写入，最后写 summary。runner 无论正常、失败还是到限，都用剩余时间运行 `repro.py finalize`（不加载模型）：核对每段的批数（126）和 summary；三段都完整才逐批比较并给出 input_equal / differences，否则记为 partial，保留已完成的段，不给结论。总预算仍是 600 s：模型步骤在截止前 60 s 结束，finalize 在剩余时间内完成。

## 解释（运行前写下，不在其外声称根因）

- 输入相同但 A ≠ B：差异在计分或调用路径；哪一种与训练摘要相等就指向哪里。
- A = B = C，但都不等于训练摘要：保存后的端点现在的读数与训练时不同。训练时的未合并 LoRA、优化器和 RNG 状态没有保存，所以这里**不能**定位训练时与保存后之间的差异，只如实写"尚未定位"。
- (1)、(2)、(3) 在某些 token 上不同：说明"只取 mask 再 argmax"与"整份 argmax"或设备与 CPU 在这些 token 上有差别，并给出数量、gold 和 top-2 差；这只说明计分层面的现象，不自动推出 MPS 有问题。
- 不做的事：不缩小批次重跑（只有当同一实例的两条路径确实不一致时，才在之后另写方案缩到最小批次）；不跑 R 对照。R 的两路在训练和探针里都相等，这里要问的是 N 特有的偏差；如果 N 上三种计分和两条路径都一致，在 R 上做同样的事也不会得到新信息。若结果需要 R 的片段对照，另行申请。

## 预算与收尾

- 加载约 0.5 分钟；A、B、C 各 126 批（真图、错图各 63 批）。按探针实测约 0.85 s / 批，约 5.4 分钟；合计约 6 分钟。**上限 600 s**（runner 在截止前 60 s 结束，TERM，10 s 后 KILL）。
- runner（`scripts/mlm_repro/run.sh`）：
  - 只启动一次；runs/mlm-repro 必须不存在；ps / pgrep 检查没有其他模型进程；
  - 加载前核对 N 的 model 3ab0ab03… 和 align_report a24ff10b…，不符就 exit 12；
  - 输出 runs/mlm-repro/{summary.json, batches.jsonl}，summary 原子写入。
- 真实运行与 EXP76 都要用设备，顺序由评审决定，不同时运行。

## 验证（CPU，已完成）

`scripts/mlm_repro/test_mlm_repro.py` 18 passed（第 1 轮之后）：未改动的路径下 A / B / C 逐批对齐（input_equal，三种计分相同）；在 B 的入口改图片或改遮罩会被逐批发现（pictures / target_hash）；C 截断、缺 summary、缺 B 时报告 partial、保留已完成的段、不给结论；身份检查 6 种（正常，以及 START、config、题单、图片字节、MLM 头被改）；身份不符时不加载；不一致记录带行号 / token 位置；runner：身份不符 exit 12，正常、失败（exit 7）、超时（exit 124，在上限内）三种情况都会运行 finalize，只启动一次。原有的 3 项：
- tiny 模型上 C 的三种计分、A（align._evaluate）、B（probe.run_random）的准确率完全相等，没有不一致；
- 每批都有 input / 图片的 SHA；真图和错图批次的 input SHA 相同、图片 SHA 不同；
- 不一致记录的结构：命中、非有限值、top-2 差；
- runner：N 身份不符时 exit 12 且不调用模型；正常时调用一次；只启动一次；上限。

项目 pyright 0。没有跑全仓测试。

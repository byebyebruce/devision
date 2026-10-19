# v04-k 全评执行证据勘误（GPU-EVIDENCE-PREP-01 A/B；2026-10-20）

评审 GPU-EVIDENCE-01 / V04-GPU-EVIDENCE-01 的处理。本文件只追加说明；封存的输出、`RESULTS.json`、`IDENTITY.json`、
失败记录一字未改，也没有重测。全部核对只读已有文件（CPU），没有加载权重。

## 1. 结论

- `runs/v04-k/eval/` 的 94 份候选输出「check 0 problems」只说明**文件完整、哈希一致**，不说明每份输出在执行时没有出错。
  此前材料写的「全评有效」应读作「全评文件完整」。
- 两份输出在执行期间出现 MPS GPU 错误（InnocentVictim），**执行证据待补**：`test_gqa.mismatched` 和 `test_bind_gqa`（真图）。
  这两份以及依赖它们的比较，暂不作为有效的发布依据，也不解释成能力的退化或提升。
- 其余 92 份输出在本次审计范围内没有查到同类错误。但「日志里没有」不等于「执行可靠」，见第 4 节。

## 2. 受影响的输出（可从原文件复核）

日志 `runs/v04-k/logs/eval-full.log`（SHA 5e46aab5f9b00e8b3570c083abc205de8926c11fc4086b53c14e0c57a9cc4e02）。流水线逐个集合起子进程，
stdout / stderr 合并写进同一个日志，等上一个集合退出才启动下一个（`src/devision/train/pipeline.py`）。所以错误行之后的第一个 summary
属于出错的那个集合。

| 集合 | 运行时段（`runs/v04-k-full.out`） | 错误行 | 之后的 summary | 系统事件（`/Library/Logs/DiagnosticReports`） |
|---|---|---|---|---|
| test_gqa.mismatched | 10-19 17:26:55 – 17:32:20 | 11784 / 11788 | n 992，accuracy 0.5413306 | gpuEvent-WindowServer-2026-10-19-173146（MMU interrupt） |
| test_bind_gqa（真图） | 10-19 21:14:43 – 21:21:02 | 19803 / 19807 | n 1484，accuracy 0.6543127 | gpuEvent-WindowServer-2026-10-19-211652（MMU interrupt，6 份） |

封存文件（SHA 与 `RESULTS.json` fd5ec3b8… 中登记的一致；`IDENTITY.json` c14863e2… 未变）：

| 文件 | SHA-256 |
|---|---|
| test_gqa.mismatched.json | 476382883bc52a8e57b50f28134566c2159b18df7e591523309f76870ba28ec0 |
| test_gqa.mismatched.details.jsonl | 1979522336f110b888d22688e5aa5598172acb21b496e612fe5591ca2527e699 |
| test_bind_gqa.json | d8a6fdae31b2c5ef46f470027808437255d7920602f9af51dd594e7a839620ea |
| test_bind_gqa.details.jsonl | b985e5fea18d66d0a2f938218d776f34f97f2126cb9b9b60afde4f2be8e5644d |

**候选定位（未验证）**：

- 每份明细里只有一个请求的 latency 明显偏高（在已检查的 test_gqa、test_gqa.reversed、test_bind_gqa.mismatched 中没有同等离群值；其余集合未检查）：
  - test_gqa.mismatched 第 867 条，1336 ms，中位数 354 ms；
  - test_bind_gqa 第 475 条，820 ms，中位数 359 ms。
- 按请求耗时累计推算，这两个请求分别在约 17:31:45 和 21:16:54 前后，与系统事件 17:31:46、21:16:52 相差几秒。
- 这只是时间上的吻合，不能确定哪些题的输出被改变、改变了多少。EXP76 两次运行之间观察到连续 8 题的概率差异，时间与第一次的故障相合，归因待验证（见第 5 节），
  所以受影响的范围可能不止这一个请求。本文**不据此**剔除题目或重算分数。

## 3. 依赖这两份输出的结论（暂停引用）

- `runs/v04-k/posteval/pairs_vs_v03.json`（f7762241…）中的两项：
  - `test_gqa.mismatched`：−0.9，区间含 0；
  - `test_bind_gqa`：−2.2 [−3.9, −0.5]，0.677 → 0.654。
- 由它们算出的看图收益：
  - test_gqa = 真图 − 配错图；
  - test_bind_gqa = 真图 − 配错图。
- test_bind_gqa 的倒序翻转率：`order_sensitivity` 以真图明细为基准。
- 材料中引用了这些数字的地方：
  - `release/v0.4/notes.md` 中「held-out GQA binding test -2.2」；
  - `docs/training-log.md` v04-k 一节「29 集配对」里的 test_bind_gqa −2.2，以及「全评有效」的说法。
  这两处都已加注，指向本文。

**将来需要独立重测的最小完整集合**（本次未授权，不做）：

- 两个集合的完整输出：test_gqa.mismatched 全部 992 题、test_bind_gqa 真图全部 1484 题。
  同一冻结身份 c14863e2…，同一评测配置；不挑题，不只补跑错误附近的题。
- 一个干净集合作为复现对照：建议 test_gqa 真图（同一冻结身份）。如果同一设备能把它逐位复现成封存结果，会**增强**「两份受影响输出的新旧差异来自执行错误」的解释，
  但不能单凭这个集合通过就证明差异只来自 GPU 错误。
- 设备保持 MPS（在用户执行受控重启之后）。换成 CUDA 重测会和其余 92 份 MPS 输出混用不同设备，配对比较需要另行对照。
- 重测后重算第 3 节列出的比较，作为新文件另存，原封存文件保留。

## 4. 审计范围与结果（只看候选直接来源和这条链现有的文本日志）

匹配模式：`command buffer exited with error status|InnocentVictim|victim of GPU error|kIOGPUCommandBufferCallbackError`。

| 来源 | 文件 | 结果 |
|---|---|---|
| v04 全评 | v04-k/logs/eval-full.log | 2 处（上表） |
| v04 全评 runner | v04-k-full.out | 无 |
| 全评后统计（CPU） | v04-k/posteval/run.log | 无 |
| 收尾 / 打包 / 安装 | v04-k-finalize.out / .txt，v04-k-package，v04-k-install | 无 |
| PROTECT | protect.out，protect/*.txt，protect-k/logs/stage2.log | 无 |
| LIVE-K | livek.out，livek/*.txt | 无 |
| 校准 | v04-k-calib.out，v04-k-calib/*.txt | 无 |
| 合成确认 | v04-k-confirm.out，v04-k-confirm/*.txt | 无 |
| EXP71 读数组件 | exp71.out，exp71/*.txt | 无 |
| EXP75 | panel-number.out，exp75-panel-n/logs/n.log，exp75-panel-r/logs/r.log | 无 |
| MLM 复现 | mlm-repro.out | 无 |

- 这些 runner 用 `nohup … > X.out 2>&1` 启动，各步骤的 stderr 也写进同一个 `.out` 文件，所以「无」是检查过的结果，不是因为日志缺失。
- 没有查：exp30–exp73 的 PROTECT 上游训练链、更早的轮次（超出本轮范围）。

**系统事件与日志命中不重合**（只陈述事实，不推断根因）：

- 10-19 17:31 与 21:16 的系统事件，各对应一处日志错误。
- 另外 3 次系统事件，同一时段的运行日志里都没有错误行：
  - 23:55:03（BIF0 page fault）：EXP74 D1 训练中（23:42–00:29）。**D1 是 EXP75 R 的初始化，也就是 EXP76 起点的上游。**
  - 00:46:08 / 00:46:10：panel-bind 探针在读 start 的 random 面板（00:44:59–00:46:59）。
  - 02:51:48：EXP75 探针在读 N 臂的 random 面板（02:50:43–02:52:43）。N 臂已经判为 invalid。
- 反过来，EXP76 两次运行的日志错误（约 03:24、04:30）都没有对应的 ips。
- 所以「日志没有命中」和「没有系统事件」都不能证明执行可靠。上面 3 个时段是否影响了输出，属于未验证的边界。

## 5. 同一 checkpoint 两次 MPS 执行的逐题对照（已有文件）

EXP76 两次运行的 step0 监测，都从 R 起点评测同样 8 个集合，共 3,562 条记录：

- 3,554 条的预测和概率逐位相同；
- 只有 dev16 第 992–999 条不同。这是连续的一批，正好等于 eval_batch 8，概率最大相差 0.104，预测相同。
- 第一次运行的 dev16 在 03:19:16 – 03:26:04；第 992/1200 条按比例约在 03:24:50，与第一次运行日志里的错误时间相合。

观察到的情况：

- 在这两次运行的其余 3,554 条上，MPS 把同一 checkpoint 的评测逐位复现了。
- 观察到 8 题的概率差异（预测、gold 与输入字段相同），时间与第一次的 InnocentVictim 相合。这支持「这类错误可能不崩溃、不出 NaN，
  而是改变一批输出」的解释，但**不是因果证明**；哪一次的数值受了影响也未验证。

据此，任何出现过这类错误行的输出，都不能凭「事后还打印了 summary」就当作执行有效。

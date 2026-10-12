# deVision v0.2：Leanmcp/jevbench 图片子集实测

> 名称核对补充（2026-10-09 13:53:15 +08:00）：本次运行的是 **Leanmcp/jevbench v0.1 的 751 题图片子集**，不属于 [Benchmark Heaven 的 Image JevBench / JevImageBench](https://benchmarkheaven.com/image-jev-bench)。后者当前为 v0.3.0，使用另一套图片、题目和评分。上轮按历史调研选择数据时没有充分核对这个专有名称；本文成绩有效，但不能当作后者的成绩或排行榜分数。当前未运行 Benchmark Heaven 的基准。

记录时间：2026-10-09 13:48:13 +08:00。实际推理：2026-10-09 13:42:26–13:45:35 +08:00（含加载、校验和预热；正式请求约 184.16 秒）。

## 本次运行与主要结果

按用户要求实际运行 [Leanmcp/jevbench v0.1](https://huggingface.co/datasets/Leanmcp/jevbench/tree/v0.1) 的全部两个图片子集，共 751 题。模型为当前发布的 **deVision v0.2（v12-ext）**，不是尚在实验中的 R13。所有请求成功，未修改题目文字、state 对象、选项键或顺序；未进行训练或重新拟合温度。

“JevBench”有多个同名社区项目，本报告只对应上述仓库、版本和图片子集，不代表所有名为 JevBench 的基准，也不代表通用视觉能力总分。

| 子集 | 题数 / 不同图片数 | 答对 | 准确率 | 95% 区间，按图片分组 | NLL | ECE，10 等宽箱 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ScienceQA image | 500 / 487 | 318 | **63.60%** | 59.25%–67.60% | 0.7627 | 7.82% |
| VQA-RAD | 251 / 135 | 119 | **47.41%** | 39.77%–55.28% | 0.8517 | 21.82% |
| 按题数合并 | 751 / 622 | 437 | **58.19%** | — | — | — |

区间为 2,000 次图片组 bootstrap，seed=7。同一张图上的多道题共同重采样。上游原始 `scores.json` 的区间按 `family_id` 分组；这两个子集中的 family 是题目而非图片，所以本表另外给出按图片分组的区间。图片身份采用解码后的完整 RGB 像素哈希。

ScienceQA 选项数为 2/3/4/5，分别 144/123/227/6 题；均匀随机选择的期望准确率为 34.19%。按选项数，实测分别答对 95/144、59/123、162/227、2/6，不把仅 6 题的五选一结果概括为稳定能力。

VQA-RAD 的随机基线为 50%，全部答“否”为 133/251 = 52.99%。模型实际 **230/251 = 91.63% 回答“是”**，而 gold 中只有 118/251 = 47.01% 为“是”。混淆矩阵 TP=108、FP=122、FN=10、TN=11，AUROC=0.4802。在本套原始请求格式下，结果未显示可靠的放射影像判别能力。仅凭本次运行不能把原因单独归于视觉编码器、领域数据或问题位于 state 的布局。

Brier 沿用上游实现：ScienceQA 是各类别平方误差之和（0.4439），VQA-RAD 是二分类 `(P(true)-y)^2`（0.3115），二者口径不同，不汇总平均。两个集合均无 gold 概率为零的情况。准确率使用 choice 概率最大项和 noul 固定 0.5 阈值；未使用评分器的“在测试集上搜索最佳阈值”诊断结果替代正式成绩。

## 与同题公开结果的对照

使用作者发布的 [djev-0.1-images 逐题预测与评分](https://huggingface.co/datasets/Leanmcp/jevbench/tree/v0.1/predictions/djev-0.1-images)，没有运行该参考模型。用同一份上游评分函数重新计算其准确率、NLL、Brier、ECE，与公开文件一致（误差小于 1e-10）；两者的 751 个 case ID 完全对齐。

| 子集 | deVision v0.2 | 公开 djev-0.1 | 差值，百分点 | 配对差值 95% 区间，按图片分组 |
| --- | ---: | ---: | ---: | ---: |
| ScienceQA image | 63.60% | 82.80% | -19.20 | [-24.30, -14.31] |
| VQA-RAD | 47.41% | 78.09% | -30.68 | [-42.75, -17.65] |

这是同题的端到端模型比较。参考项目将 djev-0.1 标为 DiffusionGemma 26B-A4B；模型规模、原生图像处理和硬件不同，不是同规模架构消融或速度比较。没有 Laya Vision 的同题成绩，本次未构造相关对比。[上游框架说明](https://github.com/Leanmcp/jevbench)

## ScienceQA 数据重合边界

此次检查了 v0.2 实际初始化链中 v3、v6、v7、v8、v9b、v11、v12 所用文件内已知 ScienceQA/AI2D/TQA 来源的图片，以及 dev_mix、旧 ScienceQA benchmark、v10 ScienceQA 监测集。匹配使用原始文件哈希、解码 RGB 像素哈希，另复现本项目 quality=92 的 JPEG 存储编码；v10 还核对官方 test 的原始行号。方法、文件列表和逐题匹配见 `overlap.json`。

- **63/500 题**的图片匹配到至少一个实际使用过的训练文件；其中 60 题匹配到 v12 的 `data/ext/train_mix.jsonl`，63 题匹配到 v7/v8 的训练文件。这里只证实同图，不把不同问法或不同候选项说成同一道训练题。
- **117/500 题**匹配到既有评测/监测图片或题目：旧 `data/lv_bench/scienceqa.jsonl` 63 题、v10 监测图示题 52 题、`data/v5/dev_mix.jsonl` 2 题。这一组与上述训练匹配的 63 题不重叠。
- 剩余 **320 题**在本次检查范围内没有匹配，答对 223/320 = **69.69%**。这是事后切片，不能改称预先固定的独立测试集，也不能与参考模型的全量 82.8% 直接比较。
- VQA-RAD 在上述有限索引中未发现匹配；本轮未进行全来源、近重复或基座预训练污染审计，不据此保证它完全未见。

因此保留 500 题的原始基准分数，明确披露复用，**不将其称为完全未见数据上的泛化成绩**。本次没有删除、替换或补抽任何 benchmark 题目。

## 是非标签方向核对

上游评分器会在 `accuracy_if_polarity_flipped > accuracy_at_0.5` 时打印通用 POLARITY WARNING，本轮 VQA-RAD 触发了它。本次另从固定原始版本下载 test parquet，逐条核对 **251/251** 的原始 question、yes/no answer 和 benchmark gold，全部一致；yes/y 映射 true，no/n 映射 false。

也核对了本地 `laya.common.render_options` 的 `[false, true]` 语义顺序、`devision.infer.Decider._answer` 返回 `p[1]` 作为 noul，以及上游 `parse_answer` 的 `P(true) >= 0.5` 评分规则。未发现反转，保留原始 47.41% 成绩；未为了提高分数翻转输出。完整证据见 `polarity-verification.json`。

## 可复现信息与产物

- 数据仓库：`Leanmcp/jevbench`，tag `v0.1`，固定提交 `ed50fe0381714d1c0273cbe03cece755b25d586b`。
- 上游评分代码：`Leanmcp/jevbench`，提交 `b473da550f51e9727b050ae109f3cd9bc96a1f18`；原样保存 `common.py`、`score_runs.py`，仅在调用时指定本地结果目录。
- 模型：`runs/release/modelscope/`，名称 `devision-v0.2`；权重 SHA256：`07a3e8aacb986396d3975d12d672526c6d3b8ec240b9d9d503ef235c5c87c3ae`，与发布记录一致。
- 本地推理代码提交：`50df60737995564cf8d8f1da5fe087f6e690c89c`；涉及源码文件哈希另外记录在 `meta.json`，环境版本在 `environment.json`。
- 校准：原发布模型的题型及选项数温度，完整参数保存在 `meta.json`；本轮未校准。
- 设备：Apple M4，24 GiB，CPU FP32，2 个算子线程、1 个 inter-op 线程；逐题请求，预热 3 次。devision 0.2.1、laya 0.3.22、torch 2.14.0、transformers 5.17.0。
- 输入：原始 JSON state/question/criteria，单图走公开 `decide(image=...)`；原生 256×256 letterbox。0 请求错误，0 state token 截断，0 因 head 总预算再次压缩选项，0 候选项 token 碰撞。原生单选项 48-token 上限仍属于模型接口行为。
- 正式 751 个请求耗时 184.16 秒；CPU P50：ScienceQA 236 ms、VQA-RAD 226 ms。**同时存在 R13 的 MPS 工作负载，且只用两个 CPU 线程，不把这些数字当作独占设备性能结果。** 下载、加载、预热与统计不计入上述请求耗时。
- 本次未运行去图、错配图、换序对照，准确率不等于已证明对应的看图收益；未改动训练任务或对外上传结果。

本次实际运行命令（已有结果时脚本会拒绝覆盖）：

```bash
uv run --no-sync python -B -u runs/jevbench-v0.1-devision-v0.2/run_bench.py
uv run --no-sync python -B -u runs/jevbench-v0.1-devision-v0.2/analyze.py
```

原始数据、固定版本的评分代码：`data/jevbench-v0.1/`；另下载的 VQA-RAD 原始 parquet 使用默认 Hugging Face 缓存。评测适配脚本和产物保存在 `runs/`，未改动 `src/`、训练配置或依赖。

- [简要汇总](../../runs/jevbench-v0.1-devision-v0.2/summary.json)
- [上游口径完整评分](../../runs/jevbench-v0.1-devision-v0.2/scores.json)
- [751 条逐题预测](../../runs/jevbench-v0.1-devision-v0.2/predictions.jsonl)
- [原始 API 响应](../../runs/jevbench-v0.1-devision-v0.2/responses.jsonl)
- [运行身份与文件哈希](../../runs/jevbench-v0.1-devision-v0.2/meta.json)
- [图片重合核对](../../runs/jevbench-v0.1-devision-v0.2/overlap.json)
- [是非标签核对](../../runs/jevbench-v0.1-devision-v0.2/polarity-verification.json)
- [运行日志](../../runs/jevbench-v0.1-devision-v0.2/run.log)

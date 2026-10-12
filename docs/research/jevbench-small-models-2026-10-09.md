# JevBench 图片评测：小模型公开成绩核实

记录日期：2026-10-09（Asia/Shanghai）。范围：查阅发布者的仓库、模型卡、原始评分 JSON 和论文；未运行其他模型，未下载权重。本文为资料查询，不是验收或 critic。

## 结论与比较口径

对我们刚运行的 **Leanmcp/jevbench v0.1 图片子集，ScienceQA 500 题 + VQA-RAD 251 题**，本次检查其公开发布目录、评分文件和说明，**没有找到约 0.5B 视觉模型在同一套题上的已发布成绩**。这仅描述已核实的来源，不代表全网不存在。公开同题视觉结果来自 djev-0.1，其主干是 DiffusionGemma 26B-A4B，不属于 0.5B 级别。

另有名为 **Image JevBench（Benchmark Heaven）** 的不同基准；不能把它与上述 751 题或其准确率合并成同一排行榜。下文分开记录其公开信息。

## Leanmcp/jevbench：公开文件核实

HF 数据仓库 main 当前 revision 为 `ed50fe0381714d1c0273cbe03cece755b25d586b`。本次分别查询 main 和 v0.1 的 `predictions/`，均只有四个系统目录：

- `djev-0.1-images`
- `djev-0.1`
- `jev-1.13.0`
- `laya-421m-en`

来源：[main 目录 API](https://huggingface.co/api/datasets/Leanmcp/jevbench/tree/main/predictions?recursive=true&limit=1000)、[v0.1 目录 API](https://huggingface.co/api/datasets/Leanmcp/jevbench/tree/v0.1/predictions?recursive=true&limit=1000)。

`laya-421m-en` 是纯文本 ModernBERT-large 决策模型，不能当成 Laya Vision 的图片成绩。其 scores 文件只有 12 个文本切片，包括 `scienceqa_text`，不包含 `scienceqa_image` 或 `vqa_rad`。[发布说明](https://huggingface.co/datasets/Leanmcp/jevbench/blob/v0.1/README.md)、[Laya 原始 scores](https://huggingface.co/datasets/Leanmcp/jevbench/blob/v0.1/predictions/laya-421m-en/scores.json)

| 同题公开结果 | 参数口径 | ScienceQA image，500 题 | VQA-RAD，251 题 |
| --- | --- | ---: | ---: |
| djev-0.1 | DiffusionGemma 26B 总参数、4B 激活 | 82.80%（414/500） | 78.09%（196/251） |

以上直接读取发布者 `headline.value`，二分类阈值为 0.5，零解析错误。本轮未运行该参考模型；上轮已用原始逐题预测重算并核对评分，不能把重算评分称为独立运行模型。参数说明来自发布者 README。[原始 scores](https://huggingface.co/datasets/Leanmcp/jevbench/blob/v0.1/predictions/djev-0.1-images/scores.json)、[框架 README](https://github.com/Leanmcp/jevbench)

## 小模型在其他口径上的结果

以下仅说明存在相关任务成绩，**不能拿它们直接减去我们的 63.60% / 47.41%，据此判断强弱**。

| 模型 / 作者报告身份 | 规模 | ScienceQA | VQA-RAD | 与本次 751 题的差别 |
| --- | ---: | ---: | ---: | --- |
| SmolVLM 系列论文 Table 1 的 500M 变体 | 0.5B | 80.0% | 本次核对未见 | 论文采用 VLMEvalKit / OpenVLM 口径，非 JevBench 500 题；论文覆盖图像和视频版，未进一步核实这项分数与原始 `SmolVLM-500M-Instruct` 精确 revision 的映射 |
| `thaitea/laya-vision` 当前推荐版本 | 0.201B | 82.4%，官方 validation 有图子集 2,097 题 | 80.6%，另一套 62 题 | 两列均不是我们刚运行的题目集合 |
| `thaitea/laya-vision-modernvbert-250m` | 0.250B | 79.0%，官方 validation 有图子集 2,097 题 | 本表不引用 | 比 0.5B 更小；不同集合 |

SmolVLM 数据来自作者论文 [Table 1 与 §4.2](https://arxiv.org/html/2504.05299v1#S4)。论文说明使用 VLMEvalKit，500M 图像评测的最长边为 1920。原始 [SmolVLM-500M-Instruct 模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) 的 Evaluation 是图片；本次读取的卡片文字没有单列 ScienceQA / VQA-RAD 分数。因此本表将 80.0% 归于论文中的 500M 变体，不冒称是某个未核实 checkpoint 的结果。

Laya 来源：[201M 模型卡](https://huggingface.co/thaitea/laya-vision/blob/f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc/README.md)、[ModernVBERT-250M 模型卡](https://huggingface.co/thaitea/laya-vision-modernvbert-250m/blob/4081dc6e00284a0d7da2071baedb9f8bf2f9d451/README.md)。201M 卡片还列出旧 237M 版本 ScienceQA 82.8%、VQA-RAD 88.7%；仍分别为 2,097 / 62 题的其他口径，不能误配到 Leanmcp 的同名切片。

若要回答“与同体量模型相比怎么样”，最直接的后续工作是固定我们已用的 751 题、图片和提示语，实际运行一个明确 revision 的 SmolVLM-500M，并统一答案解析 / 概率抽取方法。本次只是查询公开资料，未执行该对照。

## Image JevBench（Benchmark Heaven）：独立核实

核对时间：2026-10-09 13:53:56 +08:00。

[发布者主榜](https://benchmarkheaven.com/image-jev-bench)当前标为 **JevImageBench v0.3.0**（Built 2026-10-07），54 measured / 52 accepted / 2 dated carries；题池 2,441 题、2,160 张图，抽取 300 public / 1,200 sealed。该项目与 Leanmcp/jevbench 的 751 题不是同一套；我们上一轮没有跑此榜单的基准。

当前主榜页面未找到标为 0.5B、500M 或 SmolVLM 的行；这是页面范围内的核对，不是全网不存在的证明。找到以下附近规模模型：

| 模型 | 标注规模 | Capability Score | Intelligence | Calibration |
| --- | ---: | ---: | ---: | ---: |
| JPT-0.8B | 0.8B | 46.1 | 20.4 | 71.7 |
| OmniJev 0.8B | 0.8B | 38.6 | 11.4 | 65.8 |
| PlayJev 0.8B | 0.8B | 25.6 | 7.6 | 43.6 |

本表来自主榜 v0.3.0 部分。Capability Score 是 Intelligence 与 Calibration 的平均值，**不是准确率，也不是同时计入速度与成本的 Composite**；不得与我们 58.19% 的准确率比较。页面同时含旧版 v0.1.5 存档，个别模型详情链接仍显示 v0.1.5，不能把不同版本的字段拼接。来源：[主榜](https://benchmarkheaven.com/image-jev-bench)、[JPT 的旧版详情页示例](https://benchmarkheaven.com/image-jev-bench/jpt_0p8b)。

同一主榜的评测状态表还将 Laya Vision 标为作者要求暂缓公布，未排名，不能据此填写 Laya Vision 的 Image JevBench 分数。[发布者状态表](https://benchmarkheaven.com/image-jev-bench)

名称更正：上轮按既有调研选择了 Leanmcp 数据，未充分核对用户提到的专名 Image JevBench。已在 [实验记录](../experiments/2026-10-09-jevbench-images-v0.2.md) 补充说明；原始运行结果和证据不变。本次未下载或运行 Benchmark Heaven 的测试集。

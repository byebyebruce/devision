# 图片版 JevBench 与相近公开基准调研

记录时间：2026-10-06 16:56:11 +08:00。补充核实：2026-10-06 16:58:30 +08:00。访问日期：2026-10-06。

问题：是否存在公开、带图片和标签、适合图片决策模型的 JevBench 或相似基准。本文为资料调研，不是项目验收；没有下载数据包或权重，没有运行基准。

## 结论

有。当前最直接的候选是 **Leanmcp/jevbench 的两个图片子集，共 751 道题**：ScienceQA 500 道 `choice`，VQA-RAD 251 道 `noul`。题目、答案、图片、来源版本和评分代码均公开。另一个 **Praveenrajus/jev-bench（Jevify）** 提供覆盖 POPE、A-OKVQA、AI2D 的视觉评测，作者报告共 4,244 道测试题，但图片通过公开上游重建，不是该 HF 数据卡的 22 个文本配置之一。

“JevBench”对应多个独立社区项目。本记录按仓库区分，不把同名项目称为彼此的官方视觉版本，也不把不同套件的分数混为同一排行榜。

## Leanmcp/jevbench：已公开图片与题目的视觉子集

- [数据仓库](https://huggingface.co/datasets/Leanmcp/jevbench)，本轮核对版本为 `v0.1`。
- [评测框架](https://github.com/Leanmcp/jevbench)。框架按统一输入评估结构化决策概率，含准确率、Brier、NLL、ECE 等指标。
- [问题文件目录](https://huggingface.co/datasets/Leanmcp/jevbench/tree/v0.1/cases)、[图片目录](https://huggingface.co/datasets/Leanmcp/jevbench/tree/v0.1/images)。

| 子集 | 题数 | 类型 | 内容 | 发布文件 |
| --- | ---: | --- | --- | --- |
| ScienceQA image | 500 | `choice`，选项数按题变化 | 带图科学选择题 | `cases/scienceqa_image.jsonl`、`images/scienceqa/` |
| VQA-RAD | 251 | `noul` | 放射影像的是非问答 | `cases/vqa_rad.jsonl`、`images/vqa_rad/` |

题数来自公开的 [scores.json](https://huggingface.co/datasets/Leanmcp/jevbench/blob/v0.1/predictions/djev-0.1-images/scores.json)；通过 HF 目录 API 分别核对到 500 和 251 个图片文件。这是文件数，不是经过去重的独立图片数量。本轮只读取公开文档、代码、目录元信息及每个视觉 JSONL 的第一条记录，没有下载完整题库或图片。

抽查记录包含 `state`、`question.type`、`question.instructions`、`question.criteria`、`gold`、`image_path`、原始 HF repo/revision 和许可。ScienceQA 首条实际有两个候选项，因此不能误称整个子集都是四选一。VQA-RAD 将答案为 yes/no 的题目转为布尔 gold。对 deVision 的接入工作预计主要是输入适配、图片路径解析和概率输出映射；这是接口阅读后的判断，尚未实现或验证。

[来源清单](https://github.com/Leanmcp/jevbench/blob/main/jev_bench/sources.json)固定了 ScienceQA 和 VQA-RAD 的上游 test split 与 revision，分别记录 CC-BY-SA-4.0、CC0-1.0。它隐藏答案、解析等字段。[构建器](https://github.com/Leanmcp/jevbench/blob/main/jev_bench/workspace/build_cases.py)从 ScienceQA 有图行抽样、从 VQA-RAD 筛出闭式是非问答。

使用边界：

- 只覆盖科学题和放射影像，不能单独代表通用视觉决策能力。
- HF 首页的 24,799 是多个系统的累计决策预测数，不是图片题数；自动 viewer 的 `train` 是图片目录自动识别结果，不代表原始来源训练划分。
- 公开视觉成绩来自 `djev-0.1-images`。该项目中的 Jev API 路径和 Laya 421M 文本模型没有参与图片部分，不能据此宣称已有与 Laya Vision 的同题比较。
- 框架提供选项换序分析，但当前图片构建子集不在 `PERMUTABLE` 中，公开视觉成绩的 `position_bias` 为 null；不能把文本部分的换序结果说成视觉部分已测。
- ScienceQA 图片题与其 `scienceqa_text` 来自不同题目集合，二者准确率之差不是同题去图实验。本任务不复核作者全部统计结论。

## Jevify / Praveenrajus/jev-bench：覆盖更广，可从上游构建

[HF 数据卡的 Vision 部分](https://huggingface.co/datasets/Praveenrajus/jev-bench#vision)公开了视觉评测方式与结果；[视觉转换代码](https://github.com/uspraveen/Jevify/blob/main/jevify/bench/adapters/vision.py)将公开上游转换为概率决策题。

| 来源 | 作者报告测试题数 | 类型 | 能力 |
| --- | ---: | --- | --- |
| POPE | 2,000 | `noul` | 图片里是否存在指定物体 |
| A-OKVQA | 744 | `choice`，四选一 | 视觉常识问答 |
| AI2D | 1,500 | `choice` | 科学图示理解 |
| 合计 | 4,244 | | |

作者公开预测和不带图片的记录清单，图片在运行时从 `lmms-lab/POPE`、`HuggingFaceM4/A-OKVQA`、`lmms-lab/ai2d` 加载。HF viewer 中的 22 个配置是文本数据，不能仅凭 Image 标签把它当作已经打包完毕的 4,244 题图片包。

划分口径需要保留：A-OKVQA 将原 validation 再划为校准/验证与测试用途；POPE、AI2D 也从原 test 中划出验证部分。这些是作者的派生划分，不是原官方 test 全量成绩。该项目支持概率质量测量，并公布温度拟合前后 ECE；如比较 deVision，应单列未经拟合和经独立校准后的结果。本轮未重建上游数据，也未核对作者派生划分按图片身份是否隔离。

## 相近项目核实

以下是名称相近或接口相似的候选；不能仅凭模型仓库公开或存在排行榜就认定基准图片已经公开。

| 项目 | 图片、问题、标签、评分器的公开情况 | 对 deVision 的适用性 |
| --- | --- | --- |
| Jev-Vision General track | 有公开数据转换脚本和评分器；脚本从公开上游获取图片、问题及标签，形成 1,500 题；本轮未实际重建 | 原生 `noul` / `choice`，接口接近；其中 NLVR2 是双图，其余四源是单图 |
| Jev-Vision Screen tracks | 问题、标签和评分器在仓库；作者文档仍把图像仓库标为 private staging | 是图片决策基准，但本次未核实完整图片包公开可下载 |
| jev-visual benchmarks | 随仓库附单张狗图片，围绕它重复八道事实题测吞吐 | 可作性能微基准，不宜当作视觉能力测评集 |
| Laya Vision evaluation suite | 有公开准备代码、上游来源及评估代码；准备后的数据描述为作者的 Modal volume 内容 | 可参考其视觉决策题转换方法；本次未确认存在统一打包公开的完整测评集 |

### Jev-Vision：一般图片与网页截图需要分开看

**General track** 的作者文档称其为 v2 draft。它从 POPE、MME、A-OKVQA、Food-101、NLVR2 各采样 300 题，总计 1,500 题。前两类为图片是非判断，A-OKVQA 为四选一，Food-101 为 20 选一；NLVR2 判断一句话是否符合两张图。按这一配比，单图部分为 1,200 题。它是社区项目内的视觉决策测试，不是本次已确认的 JevBench 官方视觉扩展。[基准说明](https://github.com/sseanliu/Jev-Vision/blob/main/bench/README.md)

构建脚本默认 seed=11，以流式上游数据的前 `4 × per_source` 条为候选池再抽样；保存图片、`items.jsonl` 和另一种评测行格式。脚本中未固定上游 dataset revision，而且个别来源失败会捕获异常后继续，因而“脚本公开”不等于本轮已经验证完整重建成功或复现作者完全相同样本。需要使用时应核对最终各来源计数并固定版本。[构建代码](https://github.com/sseanliu/Jev-Vision/blob/main/bench/build_general_items.py)

**Screen tracks** 的问题包括选目标 `ground`、是否跳过 `skip`、操作是否生效 `effect`、是否完成 `done`。`effect` 使用操作前后两张截图。v0 有 2,223 题、836 张截图；v1 candidate 与 pixel 分别为 2,141 和 2,003 题。作者基准说明当前把 v0/v1 图片位置写为 Hugging Face 私有暂存仓库；本轮打开对应地址未成功获取公开数据卡，因此只能确认问题文件和代码公开，不能宣布图片包已开放。[数据与任务说明](https://github.com/sseanliu/Jev-Vision/blob/main/bench/README.md)、[项目公开状态说明](https://github.com/sseanliu/Jev-Vision#layout)

评分器接受 `noul` 的 P(yes)，以及 `choice` 的选项与该选项置信度，报告准确率、ECE、选择性准确率；二分类还计算 AUROC 与正负类准确率。与我们输出概率的接口接近。[评分代码](https://github.com/sseanliu/Jev-Vision/blob/main/bench/eval_submission.py)

### jev-visual：测速度的固定工作负载

作者说明输入是附带的一张狗照片与固定上下文，八道可见事实问题重复扩展为 1、4、16、64 个决策，用来比较生成 JSON、逐题前向、共享图片上下文三条执行路径。这能帮助研究多问题推理吞吐，但图片与问题覆盖不足，不能据此评估对象识别、计数、空间关系等泛化能力。[Benchmark reproduction](https://github.com/hr98w/jev-visual/blob/main/benchmarks/README.md)

### Laya Vision：可参考的公开评估代码，尚非确认的一包式基准

作者模型卡报告完整套件为 34 个验证集、59,427 道题，并列出 A-OKVQA、带图 ScienceQA、重新按图片划分的 VQAv2 yes/no。这里的题数是作者特定 `full_eval` 运行的统计，不能直接解释为一个独立公开测试集的发布规模。[模型卡](https://huggingface.co/thaitea/laya-vision#full-evaluation-suite)

数据文档说明准备后的 JSONL 与图片保存在 `laya-datasets` Modal volume，公开代码提供来源和准备流程；`full_eval` 会运行多个数据组、游戏及延迟测试。因此它可以作为重建与接入参考，但本轮没有验证全部来源的下载权限、完整原始图片和最终划分可逐项复现。[数据说明](https://github.com/r33drichards/laya-vision/blob/main/site-docs/concepts/data.md)、[评估入口](https://github.com/r33drichards/laya-vision/blob/main/site-docs/how-to/evaluate.md)

其中 `evalsets.py` 明确把 POPE、VizWiz 可回答性转成 `noul`，把 CIFAR-10H、FER+ 转成 `choice`，另外还有图像质量、图文一致性 `score`。部分数据带人类投票分布，可评估概率质量；转换代码公开，但仍需分别取得上游图片和标签。EvalMuse 官方测试标签没有公开，代码使用训练数据重新划出验证集，这一点不能与独立官方测试混同。[转换实现](https://github.com/r33drichards/laya-vision/blob/main/laya/evalsets.py)

## 使用边界

- 推荐优先考虑题目、图片、标签与评分口径一起公开的候选，再决定是否接入；本次未实施转换或下载。
- 若选 Jev-Vision General track 的单图部分，应注明是 1,200 题子集，不与全量 1,500 题成绩直接比较。
- 若来源包含我们已训练或反复调参使用的数据，需要先按图片身份核对交集；公开外部题集不自动构成新的独立测试集。
- 本记录不对作者公布的准确率或性能作独立复现背书。

## 补充：专名 Image JevBench（2026-10-09 13:53:15 +08:00）

本轮进一步发现 [Benchmark Heaven Image JevBench](https://benchmarkheaven.com/image-jev-bench)，页面当前标题为 JevImageBench v0.3.0，发布池 2,441 题 / 2,160 图，使用 300 public / 1,200 sealed 抽样。它与上文 Leanmcp/jevbench 不是同一项目。本文件 10 月 6 日的检索没有覆盖这项，不能将上文推荐候选等同于该专名基准。2026-10-09 已跑的 deVision 751 题属于 Leanmcp 子集；历史实测原样保留。后续同名任务应使用完整项目名、链接及版本定位。详见 [小模型公开成绩核对](jevbench-small-models-2026-10-09.md)。

# ModelScope 图片决策数据集检索

检索日期：2026-10-06（Asia/Shanghai）。目标是找到可改造成 deVision“图片 + 问题 + 候选项 / 判断题 → 标签”的数据来源。本次是资料调研，不是数据验收；只读取公开数据卡、目录元数据及上游少量文本预览，未下载数据集、图片、权重，未安装依赖、运行训练或评测，不写 critic。

## 核实方法与规模口径

ModelScope 页面有时只返回前端空壳，因此本次用公开 API `https://modelscope.cn/api/v1/datasets/OWNER/NAME` 核实 `Code=200`、仓库身份和数据卡，再用 `.../repo/tree?Revision=master&Root=SUBSET&Recursive=true` 核实具体数据文件。下列链接均已核实，不是根据 Hugging Face 名称猜测的镜像地址。版本固定为检索时的 `master`；未对大文件做完整性或镜像一致性校验。

**行数、图片数、问答数不同。** The Cauldron 按图片聚合多轮问答，一行可以有多题。其 README 正文的源数据统计与 YAML 中实际发布行数也不完全一致；以下优先列 YAML 的 `num_examples`，不把源数据总量直接当成当前镜像可用题量。[Cauldron 镜像 README](https://modelscope.cn/datasets/AI-ModelScope/the_cauldron/resolve/master/README.md)

## 科学示意图、图表和几何

| 数据集与已核实的 ModelScope 入口 | 任务和标签 | 可用规模与划分 | 接入 deVision 的判断 |
| --- | --- | --- | --- |
| [ChartQA：swift/ChartQA](https://modelscope.cn/datasets/swift/ChartQA) | 真实图表的数值、比较和逻辑问答；`image/query/label`，答案为字符串列表 | train **28,299** 题、val **1,920**、test **2,500**；约 964 MB Parquet | 有训练划分，可补图表读数与比较。原生多为开放答案，转换 `choice` 需构造可靠且同类型的干扰项；其中真正的 yes/no 题可转 `noul`。不能把任意随机数当优质负例 |
| [FigureQA：The Cauldron 的 figureqa 子集](https://modelscope.cn/datasets/AI-ModelScope/the_cauldron) | 合成图表的颜色与对象绑定、最大/最小、大小比较等，答案 yes/no | 当前镜像 **100,000 行**，仅 train，4 个 Parquet、约 2.22 GB；一行包含多题。数据卡源统计约 132.7 万 QA，未全量重算镜像 QA 数 | **低转换成本**，最接近 `noul`。可补文字指定对象之后的视觉比较，但合成图表分布不能直接代表真实照片或科学图解的收益 |
| [InterGPS：The Cauldron 的 intergps 子集](https://modelscope.cn/datasets/AI-ModelScope/the_cauldron) | 几何图形题；已核实上游预览有四选一和正确字母 | 当前镜像 **1,280 行**，仅 train，约 24.9 MB；源 Geometry3K 的 3,002 道总题量不是这个子集的规模 | **低至中转换成本**：提取现成选项与正确字母，转 `choice`；可补角度、面积和几何识别。题目包含数学推导，不能全部归因于视觉理解 |
| [AI2D：The Cauldron 的 ai2d 子集](https://modelscope.cn/datasets/AI-ModelScope/the_cauldron) | 教材示意图、流程、结构和标记指代，多项选择 | 当前训练镜像 **2,434 行**，约 438 MB；每行可多题。另一个 [lmms-lab/ai2d](https://modelscope.cn/datasets/lmms-lab/ai2d) 镜像只有 test 文件，EvalScope 统计为 **3,088 题** | 现成选项可直接映射 `choice`。**项目已使用该来源，找到镜像不等于新增独立数据。** 要补训练应使用训练来源，不能把 lmms 的 test 当训练增量 |
| [TQA：The Cauldron 的 tqa 子集](https://modelscope.cn/datasets/AI-ModelScope/the_cauldron) | 教材图解问答，示例为带原生选项的选择题 | 当前镜像 **1,493 行**，仅 train，约 378 MB；源卡另列 6,501 QA，但不是本次全量核验结果 | 现成选项转 `choice` 成本低；项目已有 TQA，先查已有覆盖与重复。需要保留问题真正需要的上下文，不能因采用聊天格式就删掉题干 |
| [ScienceQA：swift/ScienceQA](https://modelscope.cn/datasets/swift/ScienceQA) | K-12 科学、社会科学和语言科学选择题；`question/choices/answer/image/hint` 等 | train **12,726**、validation **4,241**、test **4,241**，合计 21,208；**包含无图题**。Cauldron 的 scienceqa 为 4,976 行训练包，是不同格式和过滤口径 | `choice` 转换成本低。项目已有 ScienceQA，这里主要提供国内下载入口；图片决策训练应筛有图、视觉相关题，`lecture/solution` 不得作为泄露答案的输入 |

规模及字段来自 [ChartQA 镜像 README](https://modelscope.cn/datasets/swift/ChartQA/resolve/master/README.md)、[ScienceQA 镜像 README](https://modelscope.cn/datasets/swift/ScienceQA/resolve/master/README.md)、[Cauldron 镜像 README](https://modelscope.cn/datasets/AI-ModelScope/the_cauldron/resolve/master/README.md)；AI2D 的评测数量与用途另核对 [ModelScope 官方 EvalScope 文档](https://github.com/modelscope/evalscope/blob/main/docs/en/benchmarks/ai2d.md)。

上述镜像有 `image` 或 `images` 图像字段及对应 Parquet 文件；Cauldron 数据卡明确说明加载后为图像对象。**按发布格式，使用这些打包镜像不需要先去 COCO 等外站另找原图**，但本次没有逐行解码确认全部图片有效或完整。不要推广到 Cauldron 所有 50 个子集。ScienceQA 无图记录是数据本身的属性，不能当成漏下载。

其中 FigureQA 和 InterGPS 的任务与输出格式，还检查了 [Cauldron 上游 FigureQA 的首行文本预览](https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4/the_cauldron&config=figureqa&split=train&offset=0&length=1)、[InterGPS 首行文本预览](https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4/the_cauldron&config=intergps&split=train&offset=0&length=1)。这些预览是 Hugging Face 上游证据，不能替代 ModelScope 镜像逐样本校验。几何数据的原始关系由 [InterGPS 作者仓库](https://github.com/lupantech/InterGPS)确认：Geometry3K 是其发布的数据集，原始总量为 3,002 道几何题。

## 综合训练集合 The Cauldron

[AI-ModelScope/the_cauldron](https://modelscope.cn/datasets/AI-ModelScope/the_cauldron) 已核实存在，仓库 ID `21402`，API 标注约 168.9 GB。它是 Idefics2 使用的 50 个视觉语言训练数据集合集，只收训练划分，以 `images` 和多轮 `texts` 统一格式；它不是为 deVision 直接定义的概率监督协议。可只选有用子集，不必完整下载。另有 [HuggingFaceM4/the_cauldron 的 ModelScope 仓库](https://modelscope.cn/datasets/HuggingFaceM4/the_cauldron)，API 也确认存在，ID `127292`；本表主要核对的是前者。[仓库元数据](https://modelscope.cn/api/v1/datasets/AI-ModelScope/the_cauldron)

针对我们目前的数据覆盖，**新增方向优先考虑 FigureQA、ChartQA、InterGPS**；AI2D/TQA/ScienceQA 的镜像主要改善获取便利，不自动增加来源多样性。这个排序是任务匹配判断，尚无训练增益证据。导入前仍需与本项目已有训练、验证、校准及测试图片身份去重，并保留来源信息；Cauldron 标注“training only”不能证明与我们自行构建的 held-out 集不重叠。

本次尚未确认 ModelScope 上存在可直接替代我们全部 `choice/noul` 格式、并带有校准概率标签的专用图片决策训练集。现成选择题、yes/no 题可以转换为硬标签监督；硬标签不要求数据集本身带校准概率。

许可元数据有冲突：例如 ModelScope API 顶部泛写 Apache，但 `swift/ChartQA` README 的 YAML 为 GPL-3.0；ScienceQA 卡也存在顶部标签与正文非商业说明不一致。这里只记录冲突，不把镜像首页的通用标签当作各原始数据授权结论。


## 补充：计数、指代、综合决策与合成图解

下列仓库已通过 ModelScope 公开 API 核实 `Code=200` 和非空数据卡。PixMo-Count、CoSyn-400k、M3IT、Warehouse 还读取了仓库文件树，确认存在数据文件。本次不下载或重算全量内容；原作者报告规模与当前镜像实际完整性分别看待。

| 已确认的 ModelScope 数据集 | 监督内容与规模口径 | 划分与图片获取 | 转换为图片决策题的判断 |
| --- | --- | --- | --- |
| [allenai/pixmo-count](https://modelscope.cn/datasets/allenai/pixmo-count) | 图片、对象类别、计数；训练含目标点位置。训练来自 Detic 检测并过滤，验证/测试经人工核验，计数范围 2–10 | 文件树明确存在 train、validation、test Parquet；原图以 URL 提供，需另行抓取并按提供的 SHA256 校验 | **优先用于真实图片计数候选**。用 `count` 作为正确数字、同范围数值作干扰项，转 `choice`；验证和测试不混入训练。训练自动标注与人工核验的评测划分不能混为同等标签质量 |
| [allenai/pixmo-points](https://modelscope.cn/datasets/allenai/pixmo-points) | 人工指代表达 + 对象点坐标；既含一般指代，也含面向高数量计数的表达 | 数据卡有 train；原图 URL 需另下载，URL 可能重复，提供图片哈希 | **可补“文字指的是哪个对象”**。点位置可转绝对位置或明确指定对象后的关系题，但点标注不自动等于完整框、所有对象清单或可靠的“不存在”标签。多指代、歧义对象需过滤 |
| [allenai/CoSyn-400k](https://modelscope.cn/datasets/allenai/CoSyn-400k) | 合成图表、化学图、电路图、图解、文档、数学等，每图多 QA；上游论文报告 **40 万张图、270 万条指令数据** | 数据卡为 train/validation，说明划分主要用于训练而非正式 benchmark；图像直接打包。文件树已确认 chemical、circuit、diagram、document、chart 等训练 Parquet | **可补图解与文字/区域绑定的覆盖**。先选短答案、明确 yes/no、比较或计数题，再构造 `noul/choice`；QA 基于生成图片的代码而非对最终渲染图作答，需要抽查渲染后是否真的可见。论文整体数量不等于已验证镜像数量或可转换题数 |
| [allenai/CoSyn-point](https://modelscope.cn/datasets/allenai/CoSyn-point) | 合成图片、查询文字、答案点，坐标按 0–100 归一化；截图/文档等文字定位示例 | 数据卡有 train/validation，图片直接包含；非正式评测划分 | 可作为定位/绑定来源，转位置分类或区域选择需要设计明确的候选；不是现成 `choice` 答案概率集 |
| [allenai/pixmo-docs](https://modelscope.cn/datasets/allenai/pixmo-docs) | charts、diagrams、tables、other 四类合成图文 QA | train/validation、图片打包；原作者卡已推荐用 CoSyn-400k/CoSyn-point 作为改进版本 | 作为历史来源或补充，先看新版 CoSyn，避免把同家族数据简单叠加当成独立多样性 |
| [AI-ModelScope/M3IT](https://modelscope.cn/datasets/AI-ModelScope/M3IT) | 多数据集统一指令格式；卡列 `snli-ve` train 20,000、`clevr` train 30,000、`coco-itm` train 30,000，另有大量已有 VQA 来源 | 各子集有各自 train/val/test；卡示例使用 `image_base64_str`。文件树确认 `data/classification/snli-ve/` 有多版本训练/验证/测试 JSONL；未逐文件计数或复核 loader 选中版本 | **SNLI-VE 三类判别最接近决策题**：蕴含/矛盾/无法确定可转三选一。CLEVR 可补受控的关系、计数和属性组合。不要把已有 GQA/VQAv2/ScienceQA 的另一次汇编计作新增来源；不能把 `neutral` 直接改成“否” |
| [nv-community/PhysicalAI-Spatial-Intelligence-Warehouse](https://modelscope.cn/datasets/nv-community/PhysicalAI-Spatial-Intelligence-Warehouse) | NVIDIA 合成仓库 RGB-D 与区域 mask，空间关系、区域多选、距离、计数；卡报告 **499k 训练 QA、1.9k 验证 QA、19k 测试 QA、约 95k RGB-D 图对** | 文件树含 train/test 图片包及样本目录；卡说明 train/val 有 `normalized_answer`，test 不提供对应答案字段 | 左右和计数有研究价值，但 `<mask>`/`[Region n]` 需要对应区域输入。当前 deVision 不能直接理解这些占位符，必须先有明确的区域可视化/指代转换。公尺距离题还可能依赖深度信息；本次不建议直接混入普通单图判断题 |

数据卡/API 直接证据：

- [PixMo-Count 元数据](https://modelscope.cn/api/v1/datasets/allenai/pixmo-count)，仓库 ID `107120`；[文件树](https://modelscope.cn/api/v1/datasets/allenai/pixmo-count/repo/tree?Revision=master&Root=&Recursive=true)。
- [PixMo-Points 元数据](https://modelscope.cn/api/v1/datasets/allenai/pixmo-points)，ID `107006`。
- [CoSyn-400k 元数据](https://modelscope.cn/api/v1/datasets/allenai/CoSyn-400k)，ID `106532`；[CoSyn-point 元数据](https://modelscope.cn/api/v1/datasets/allenai/CoSyn-point)，ID `107681`。规模来源是 [CoSyn 作者论文](https://arxiv.org/abs/2502.14846)，不是本次下载后统计。
- [PixMo-Docs 元数据](https://modelscope.cn/api/v1/datasets/allenai/pixmo-docs)，ID `107309`，卡中直接给出新版推荐。
- [M3IT 元数据](https://modelscope.cn/api/v1/datasets/AI-ModelScope/M3IT)，ID `18812`。SNLI-VE 的三类含义与标签限制另核对 [作者仓库](https://github.com/necla-ml/SNLI-VE)，CLEVR 的任务来源核对 [作者项目页](https://cs.stanford.edu/people/jcjohns/clevr/)。
- [Warehouse 元数据](https://modelscope.cn/api/v1/datasets/nv-community/PhysicalAI-Spatial-Intelligence-Warehouse)，ID `105994`；图像、深度、区域和标准答案字段均来自其数据卡示例。

这些 ModelScope 数据卡/API 的 PixMo/CoSyn 顶部许可标签与正文也有差异：多处顶部为 Apache，而正文说明为 ODC-BY-1.0 并列生成来源条款。因此本记录不把首页标签视为准确的原始数据许可汇总。

## 对下一轮数据选型的建议（尚未实施）

按与本项目接口和能力需求的匹配程度，建议由数据执行者优先做小规模格式/标签抽查：

1. **FigureQA**：直接 yes/no，转换成本低，重点补按颜色或名称找到目标后比较的监督。应按图留出并保留成对问题，合成图表上的收益单独报告。
2. **PixMo-Count**：补来自真实新图的计数变化，成本主要是抓取原图、核验哈希和筛掉错误自动标注。用原始 train，保留验证/测试角色。
3. **CoSyn-400k 的 diagram/circuit/chemical/chart 子集**：补科学图解与文字密集图形的多样性。规模大，先选在实际输入分辨率下仍可辨的简短题，不把文字被缩小后不可读的数据当有效监督。
4. **M3IT 的 SNLI-VE / CLEVR 与 ChartQA、InterGPS**：作为不同能力方向的后续补充；其中开放答案转换要维护同义词、答案类型和有意义的干扰项。若只想先取得现成选择题，InterGPS 和已使用的 AI2D/TQA/ScienceQA 转换更简单。

这是基于任务结构的建议，不是已证明的训练增益排序。所有新增来源均应按图片身份与本项目训练、验证、校准和评测集合核对；不同集合中的同图或重打包图不能仅靠数据集名称判断隔离。本次不为任何数据集作全量质量背书，没有建立、修改或下载训练集，也没有启动评测。

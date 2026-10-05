# 数据扩充：Laya Vision 配方与候选来源

观察与记录日期：2026-10-04（Asia/Shanghai）。作者：Codex research 子代理。范围：只读官方模型卡、固定版本源码与数据作者文档；未下载数据或权重、未运行模型、未修改代码或配置。本文是外部来源核实与待执行建议，过滤后可用数量和训练收益均未验证。本地数据规模、运行状态由主评审另行核实。

## 判断

优先补真实照片中的空间关系、对象指代和复杂计数，同时保留已有科学插图覆盖。第一轮可以选 VSR 与 Visual7W telling，设置新增合计 1–2 万题的预算上限；这个数字不是保证能获得的无泄漏样本量。TallyQA 放第二步。先判断新监督能否改善独立图片上的目标能力，再扩大数据池。

Laya Vision 的优势包含已经完成的多模态预训练。下游决策题数量接近，只能补监督覆盖，不能等价替代前期图文融合训练。201M 从旧视觉决策 checkpoint 继续训练，其上游是 SmolVLM；ModernVBERT 分支从已经支持图文的 ModernVBERT 初始化。不能把我们的新投影层与文本 Laya 初始化视为相同起点。[201M 固定模型卡](https://huggingface.co/thaitea/laya-vision/raw/f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc/README.md)、[ModernVBERT 分支固定模型卡](https://huggingface.co/thaitea/laya-vision-modernvbert-250m/raw/4081dc6e00284a0d7da2071baedb9f8bf2f9d451/README.md)。

## 两个数据规模数字的口径

| 对象 | 已核实事实 | 不能据此声称 |
|---|---|---|
| Laya Vision 201M | 28,414 步、batch 64；卡中 1.82M 对应约 1,818,496 次训练抽样曝光。45% 的抽样是游戏帧，另外使用 19 个 Cauldron 闭式子集及四个 rubric 分级来源；继承更早 checkpoint | 182 万张独立通用视觉问答图片；全部属于当前 deVision 任务；这是它从预训练开始的全部数据 |
| Laya Vision ModernVBERT 分支 | 卡与实验记录为留出后 269,900 道训练问答；最多每子集 10,000 行、每行 4 个可用问答；子集均衡采样、单子集最多重复 4 遍；16,869 步、batch 32、两 epoch | 269,900 张独立图；各原始数据集全量；与 1.82M 曝光次数可直接相除 |

来源：[201M 固定模型卡](https://huggingface.co/thaitea/laya-vision/raw/f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc/README.md)、[ModernVBERT 固定实验记录](https://github.com/r33drichards/laya-vision/blob/9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c/site-docs/reference/results/modernvbert-cauldron.md)。训练原始产物未下载，以上是发布者记录，不是本地重跑复现。201M 文档未提供跨全部训练来源去重后的唯一图片总量。

## The Cauldron 的 19 个子集及转换边界

固定源码 `laya/cauldron.py` 的 `SUBSETS` 精确清单：

| 转换类别 | 子集 |
|---|---|
| 原生字母多选转 `choice` | `ai2d`, `iconqa`, `intergps`, `scienceqa`, `tqa`, `visual7w` |
| 原生选项列表转 `choice` | `aokvqa` |
| 图片中的候选面板、字母 A–H 转 `choice` | `raven` |
| 真假/是否问答转 `noul` | `figureqa`, `hateful_memes`, `nlvr2`, `vsr`, `vqarad` |
| 仅接收其中回答是 yes/no 的问答 | `clevr`, `dvqa`, `mapqa`, `ocrvqa`, `vqav2`, `chartqa` |

解析器按每个 turn 实际形式判定；上表是主要路径，不代表一个子集中的所有行都是同一题型。数值答案、开放回答、caption、代码直接跳过。字母多选要求选项唯一且答案能映射；ScienceQA 在 `Question:` 之前的上下文被保留为 state。源记录可含多图，我们不能无条件照搬。[固定转换源码](https://github.com/r33drichards/laya-vision/blob/9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c/laya/cauldron.py)。

**TallyQA 不在这 19 个子集中。** 它可以是我们另加的候选，不能引用这套配方声称对方已用它。源码说明也提到可能解析 PlotQA/TextVQA 的 yes/no，但两者不在 `SUBSETS`，不能据此扩充成已训练清单。The Cauldron 本身是 50 个数据来源的包装，不是 50 组独立新图片；既有 AI2D/TQA/ScienceQA/A-OKVQA 换个包装接入，应先去重。[Cauldron 数据卡](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron/tree/847a98a779b1652d65111daf20c972dfcd333605)。

## 高收益候选与适配要求

以下优先级是针对当前英文、单图、256×256、`noul`/`choice` 范围的建议，不是已验证效果。

| 候选 | 能补什么 | 转换与约束 | 先做什么 |
|---|---|---|---|
| **VSR** | 对两个指定物体的空间关系判断 | 原生图文陈述真假，直接适配 `noul`。随机 split 的训练集 7,680 条；零样本 split 的训练集 4,713 条，二者不能相加当新数据。图片来自 COCO，须按图片身份与现有全部评测隔离 | 首批接入。按 relation 分层，保留真假比例；明确采用哪套 split，不能交叉混用其 train/test |
| **Visual7W telling** | 更丰富的 what/who/where/how/why 等真实照片问题及指定对象问答 | 有人工构造的四个候选答案，适合 `choice`。telling 是文本答案；pointing 的答案是区域，不能把区域 ID 直接当成可理解的文字选项 | 首批接入 telling 的空间、属性、对象条件类题。按官方训练图片取样，并保留原生干扰项 |
| **TallyQA** | 带属性、动作、指代条件的复杂计数 | 原生答案是整数，不是现成多选。图来自 COCO/VG，问答部分来自 VQA/TDIUC/VG，不能把来源名变化当新增图。可用 `issimple` 区分简单/复杂 | 第二批。先采用固定、覆盖所选答案范围的计数候选集合；范围外题另行处理，不能截断标签。若构造是非或窄多选，需平衡真假及数字分布，避免候选集合透露正确数字 |
| **CLEVR / ShapeWorld** | 可控的属性组合、关系、计数、比较和反例 | CLEVR 有场景图与问题程序，yes/no 题直接适配；属性或数字题需按题型固定合法候选集合。ShapeWorld 支持随机生成图文一致性题，适配 `noul` | 作为诊断/课程实验的小份额。按场景与组合留出；合成分数上涨不能证明真实照片迁移 |
| **已有 AI2D / TQA / ScienceQA** | 科学示意图、图示关系和原生多选 | 已在本项目接入，优先补类别/图形覆盖，不重复计作新来源。TQA 有纯文本与图示题，部分需要教材上下文；ScienceQA 的 lecture/explanation 不可无区别当作可见输入，特别是答案解释 | 先沿用现有构建边界。抽查缩放后图片和必要上下文是否足以作答，并单独报告视觉区分题收益 |

候选事实来源：

- [VSR 作者 README，commit b27a0af](https://github.com/cambridgeltl/visual-spatial-reasoning/blob/b27a0af0ee1462d2b6b92c8c83e869d9254a241a/README.md)（2023-03-25）。它的 zero-shot 划分按概念留出；原生 split 仍须服从我们跨源的图片隔离要求。
- [Visual7W 作者工具说明，commit 2ba129b](https://github.com/yukezhu/visual7w-toolkit/blob/2ba129b58b3553a4dfc5defc619e3090770a9b70/README.md)（2019-10-08）：全部版本含 47,300 张 COCO 图、327,939 题，包含 telling 与 pointing，不能把该总数作为可直接训练的 telling 数量。
- [TallyQA 作者说明，commit 46cdc64](https://github.com/manoja328/TallyQA_dataset/blob/46cdc649ec79c3dcc2720ff227ad07d7ee51da6f/README.md)（2024-02-19）：全数据约 287K 题、165K 图，19K 人工复杂题；这些均不是去掉本项目留出图片后的可用训练规模。
- [CLEVR 作者主页](https://cs.stanford.edu/people/jcjohns/clevr/)与[生成器固定版本 f0ce2c8](https://github.com/facebookresearch/clevr-dataset-gen/tree/f0ce2c81750bfae09b5bf94d009f42e055f2cb3a)（2019-03-22）：原始训练集 70,000 图、699,989 题，有组合泛化版本 CoGenT；本次未生成数据。
- [ShapeWorld 作者说明，commit e720bf4](https://github.com/AlexKuhnle/ShapeWorld/blob/e720bf46e57fc01326d04d639fa6133d9c12158f/README.md)（2021-04-19）：可指定属性、关系、量化和逻辑生成条件。旧依赖在当前环境是否可直接运行未验证。
- [AI2D 作者项目页](https://prior.allenai.org/projects/diagram-understanding)、[TQA 作者项目页](https://prior.allenai.org/projects/tqa)、[ScienceQA 作者仓库，commit 2cbf831](https://github.com/lupantech/ScienceQA/tree/2cbf8318e07b9ece895bb2ae605e71e38d623264)（2024-09-19）。TQA 按课程划分，并区分图示与非图示题；只留单图须保证问题仍完整。

本项目历史已经尝试过小规模合成绑定题及扩大合成量，未解决真实图片绑定；见 [实验记录](../experiments/2026-09-30-rlcd-plateau.md) 与 [数据质量分析](data-quality.md)。因此本次不建议把扩大合成数量当作已经成立的修复。若执行者选择重试，应明确新变量，例如组合留出、带条件计数或场景图校验的关系题，并固定真实图外部验证，不能原样重复旧失败设置。

## 不宜首批照搬的来源与验收口径

- NLVR2 需要成对图，超出当前单图接口；把两图拼接会改变每图有效分辨率和任务分布，应单独立项。
- OCR-VQA、DVQA、ChartQA 等要在 **实际 256×256 预处理结果**上抽查文字与图表标记可读性。仅因为解析器能得到 yes/no，不代表样本适合当前输入；不宜先用它们凑总量。
- rubric `score` 数据和游戏操作超出本阶段目标。也没有依据仅为补来源数量引入医学等专业域。
- 新来源进入前先冻结开发、校准及最终测试图片清单；跨 COCO/VG 使用身份映射，插图保留精确像素与必要近重复核验。Visual7W/TallyQA/VSR 的新标注不等于新图片。
- 记录原始问题数、过滤后唯一题数、唯一图数、每图题数、类别与语义答案分布、训练抽样曝光次数。不要只报告 JSONL 行数；不要将不相关开放答案随机拼成容易排除的假多选。
- 首轮由执行者完成两个来源的低量级转换、人工分层抽查和固定预算对照；目标切片准确率、真实图相对错配图收益及旧任务退化一起判断。只有数据量变化、模型整体分数略涨不足以确认目标能力改善。

未核实部分：新源与本项目图片/问题的实际重叠量、256 输入可读性、过滤后可用数量、转换成本和训练收益；没有检查发布方全部预训练图片身份，不能声称外部模型在这些公开集合上完全未见过相关图片。各来源及底层图片条款应随原始数据保存；Cauldron 包装不统一改变原始许可。

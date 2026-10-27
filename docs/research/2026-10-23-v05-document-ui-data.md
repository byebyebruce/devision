# v0.5 文档与 UI 训练数据可用性

日期：2026-10-23（Asia/Shanghai）。仅查询官方说明、HF API metadata/tree、GitHub metadata；未下载数据主体、未运行模型、未修改实验脚本或运行配置。文中大小为压缩文件字节，不是解码内存。HF 仓库发布者的 metadata 是数据格式的一手证据，但镜像发布者不等于原始数据作者，许可不能跨来源自动推定。

## 当前建议

**先做 ScreenQA 的小规模人类标注 UI 目标绑定实验。** 可匿名取一个 train shard 和一个 validation shard，共约 260 MB；有真实截图、原始问题及人类标注的答案区域。需要筛选单一目标、排除重复文字/重叠框，然后生成带候选标记的 choice；不生成虚假的答案文本。DocVQA 的小 shard 也可下载，但这两份常用镜像没有 OCR 字段，不能立即实现“同页 OCR 候选”的方案。FUNSD 原始包则保留字段关系，是低成本文档诊断备选，尚需检查包内数据许可。

## 1. ScreenQA：推荐的第一项 UI 训练源

原始发布者是 [Google Research ScreenQA](https://github.com/google-research-datasets/screen_qa)，不是 ScreenSpot。原版由人类标注，官方按截图划分 train/validation/test，保证同图多问属于同一 split。`ground_truth` 含多个 raters 的 `full_answer` 与 `ui_elements`，元素含 `text`、像素 `bounds=[left,top,right,bottom]`、`vh_index`。官方 dataset 为 CC-BY-4.0，截图来自 RICO；应保留 RICO 的原始来源信息与适用条款。官方原版约 68,951 / 8,614 / 8,419 题；不要将 **ScreenQA Short** 混入本轮，因为官方明确其答案由模型自动生成。

可用包装为 [bevaya/RICO-ScreenQA](https://huggingface.co/datasets/bevaya/RICO-ScreenQA)，HF API 返回 `gated=false`，固定 revision：

```text
44c8508ea528b4d4e035cc06e28d7baf8db09ec6
```

metadata schema：`screen_id:string, question:string, ground_truth:list[{full_answer,ui_elements:list[{bounds,text,vh_index}]}], file_name:string,image:Image`。包装元数据行数 68,980 / 8,618 / 8,427，与原版 README 略有不同；因此先报告为该固定版本的 ScreenQA-derived 子集，不能默认为与原版逐题相同。

[固定版本 tree](https://huggingface.co/api/datasets/bevaya/RICO-ScreenQA/tree/44c8508ea528b4d4e035cc06e28d7baf8db09ec6?recursive=true&limit=1000)核实的最小下载：

| 路径 | bytes | 固定下载 URL |
| --- | ---: | --- |
| `data/train-00000-of-00017.parquet` | 146,636,833 | [train 0](https://huggingface.co/datasets/bevaya/RICO-ScreenQA/resolve/44c8508ea528b4d4e035cc06e28d7baf8db09ec6/data/train-00000-of-00017.parquet) |
| `data/validation-00000-of-00003.parquet` | 113,566,505 | [validation 0](https://huggingface.co/datasets/bevaya/RICO-ScreenQA/resolve/44c8508ea528b4d4e035cc06e28d7baf8db09ec6/data/validation-00000-of-00003.parquet) |

合计 **260,203,338 bytes**。全数据 metadata download_size 3,324,988,803 bytes，故初始只取指定 shards，避免自动下载全部 split。分片不是独立随机样本，可能截断同屏多问组；必须记录覆盖和筛选率。

**无模型负标签的转换方案（待实现）：**

1. 按 `screen_id` 汇总可见同屏元素，不向模型输入 OCR 文本或 gold 框坐标。只用 human `ground_truth`，对 rater 目标歧义、多区域组合答案直接排除。
2. 原问题保持不变，增加“Which marked region contains the answer?”；将正确区域和其他同屏已标注元素画为 A/B/C/D，选项仅为 marker 名称。替代任务是显式区域定位，不冒称原版自由回答分数。
3. 正目标用全部 raters 可接受目标的并集；负候选不能与任一正目标重叠、不能同文本/alias；目标若不是唯一可解释区域则剔除。只保留至少三个足够分离的候选框的样本。下载后实际可用数量未知，不能先承诺上万题。
4. 标记字母随机置换并保留映射，训练/验证同规则；不固定 A 为正，不把目标独有大小/颜色当捷径。原始问题会有 yes/no/count 等非定位问题，此类不强行转换。
5. 开训前人工审计至少 50 个“问题–所有候选–gold”组合，检查同屏 target 唯一性与标记可见性；统计图片 hash 跨 split 和跨既有评测重叠。验证增加错图、marker 换序，并按图片聚类比较。

该转换依赖真实人类区域证据，但原始 QA 未标全屏所有可接受语义答案，故“自动过滤后完全无噪声”仍不能保证。若保留率低，切换下一节保留完整 DOM 候选的 SeeClick，不用随机无关答案补数量。

## 2. SeeClick：同屏元素候选更完整，但需自建 train/validation

[原作者说明](https://github.com/njucckevin/SeeClick/blob/main/readme_data.md)给出 web `img_filename,url,elements[{instruction,bbox,data_type}]`；bbox 是归一化四边。它是爬取的元素文本/DOM 标签，并非人工多轮问答。页面提醒各数据源沿用原许可，不能用 repo 代码许可替代截图内容许可。

可按 shard 小量获取的 [moondream/seeclick](https://huggingface.co/datasets/moondream/seeclick) 是模型发布者的重打包，不是 SeeClick 原作者 HF 仓库。HF API 返回 ungated，revision：`3c2cabe59ede641ee7cd2f7f76b1f955d12e88e1`。schema 为 `image, elements:list[{instruction,bbox:float32[],data_type}], qa:list[{question,answer}]`。**本方案只用 elements，不使用来源未核实的 qa。** 仅 train 271,121 屏，没有官方 validation。

[固定 tree](https://huggingface.co/api/datasets/moondream/seeclick/tree/3c2cabe59ede641ee7cd2f7f76b1f955d12e88e1?recursive=true&limit=1000)的 shard 0 为 **497,464,605 bytes**，可取 [train-00000-of-00255.parquet](https://huggingface.co/datasets/moondream/seeclick/resolve/3c2cabe59ede641ee7cd2f7f76b1f955d12e88e1/data/train-00000-of-00255.parquet)。全量约 130 GB，不建议全量下载。

转换：从同屏 `elements` 中选唯一 instruction 的目标，候选为同屏其他非重叠元素，截图加标记，问“Click the element labelled {instruction}; which marker?”。这是页面自带文本定位，目标关系由原始 DOM 框提供，不生成自然语言推理 gold；过滤重复 instruction、父子框、多重匹配和不可见元素。先按原始页面 URL/domain 划分最稳，但该包装 schema 不保留 URL，至少按截图内容 hash 分组切出 10% dev，并披露同站模板泄漏尚未排除。保留原始 test-only ScreenSpot 完全独立。发布材料应补全原始 SeeClick 图片许可链；此镜像 card 本身没有 license 字段，不把它写成已核实 Apache-2.0。

不优先使用 `pierretokns/seeclick-web-commercial-mlx`：虽然 card 自称 Apache-2.0 和 train/validation/test（约 465 MB），其示例只有单个 click coordinate，不含完整同屏候选；“commercial safe”是镜像作者声明，无法替代原许可核查。

## 3. DocVQA 镜像：匿名可取，缺同页 OCR

### HuggingFaceM4/DocumentVQA

实际存在的名称是 [HuggingFaceM4/DocumentVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA)，`HuggingFaceM4/DocVQA` 本次匿名 API 返回 401，不能据此认定为合法下载源。可用源 revision `a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4`，ungated；metadata 有 39,463 train / 5,349 validation / 5,188 test。

schema：`questionId,question,question_types,image,docId,ucsf_document_id,ucsf_document_page_no,answers`；**没有 OCR tokens/boxes**。最小 [train 0](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA/resolve/a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4/data/train-00000-of-00038.parquet) **61,005,816 bytes** + [validation 0](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA/resolve/a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4/data/validation-00000-of-00017.parquet) **45,950,315 bytes**，合计 **106,956,131 bytes**。这些是同一 split 的局部分片，不能宣称全量基线。大小与字段来自 [HF tree](https://huggingface.co/api/datasets/HuggingFaceM4/DocumentVQA/tree/a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4?recursive=true&limit=1000) 与 [repo API](https://huggingface.co/api/datasets/HuggingFaceM4/DocumentVQA)。

镜像 card 没有 license 字段。[原作者数据页](https://site.docvqa.org/datasets)要求通过 RRC 登录并阅读条款；不能因 HF ungated 就宣称已核实商业许可。可读官方格式和原问答 gold，但没有完整 OCR 就不满足“同页 OCR 候选”要求。从同页其他问答答案拿负例也不总能证伪，不能直接把这种方案作为无歧义 gold。

### lmms-lab/DocVQA

该路径现重定向 [lmms-lab-encoder/DocVQA](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA)，revision `539088ef8a8ada01ac8e2e6d4e372586748a265e`，ungated；card 标 Apache-2.0，但这是包装者声明，仍需原 DocVQA 图片许可链。tree 有 `DocVQA/train-00000-of-00012.parquet` 等，**card configs 只声明 validation/test**。因此直接 `load_dataset(...,'DocVQA')` 不能假定得到 train，必须显式 parquet 文件列表。train shard0 **616,602,754 B**；validation shard0 **114,817,746 B**。DocVQA 子集仍没有 OCR；同仓 InfographicVQA schema 虽有 `ocr:string`，不能混为 DocVQA OCR。

## 4. FUNSD：保留字段关系的极小原始包

作者 [官方项目](https://guillaumejaume.github.io/FUNSD/)有 199 个扫描表单；[原始格式文档](https://raw.githubusercontent.com/guillaumejaume/FUNSD/master/description.html)明确 `form[{id,text,box,label,words,linking}]`，question→answer 的 linking 可以给出字段绑定 gold。按唯一 question 且唯一 answer link 筛选，choice 候选来自同页其他 answer 实体；过滤所有链接正例、重复文字、部分/组合答案。可用于区别“能读字”和“能按字段绑定”的低成本诊断，不代表一般文档泛化。

作者 GitHub commit 为 `8905aca92b3181853307e2880e3d6f71dee8e9f3`；metadata 显示原始 [dataset.zip](https://raw.githubusercontent.com/guillaumejaume/FUNSD/8905aca92b3181853307e2880e3d6f71dee8e9f3/dataset.zip) **16,838,830 B**。未下载 zip；包内数据许可仍待检查，网页 repo `LICENSE.md` 实际是网站模板许可，不能误当数据许可。该包只有原 train/test，应从原 train 按文档 hash 固定切 dev，原 test 不用于训练或选方案。

[nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) 包装 revision `7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5`，train149/test50，train parquet **12,254,116 B**，字段 `words,bboxes,ner_tags,image`；**丢失 entity linking**。因此不选它来训练“字段→答案”关系，不从 BIO 标签猜链接。

## 5. TextVQA：第一方入口存在，当前不作为小预算首选

[facebook/textvqa](https://huggingface.co/datasets/facebook/textvqa) revision `30a47cfa557c996f64903f01250799d453be6215`，HF card 为 CC-BY-4.0，repo 只有 README 与旧加载脚本。脚本指定 `https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_{train,val,test}.json` 和 `https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip`。这些是作者托管的官方入口；本次没有验证 zip 实际大小，因此不承诺初始 <=3 GB。脚本字段含 image/question/answers（10个人类答案），**不含 OCR**。不执行旧 arbitrary-code dataset loader，也不下载 test 图片。

## 执行门槛与限制

第一项建议的 ScreenQA pilot：先下载指定 260 MB，再确认图片解码、字段范围及 train/dev 图片隔离；只筛保守 human 单目标样本，输出带 split/hash/target/candidates 的 manifest，检查50题并记录排除率；若有效样本不足或 gold 歧义明显，停止构建并转 SeeClick elements 单 shard。先在256输入下确认目标可见，不能把看不清的小字题硬灌入训练后用 loss 下降作成功依据。所有 derived 评测保持固定候选，公开指标只声称对应子集，不与原版生成式分数混比。

尚未验证：数据主体完整性、gold逐题质量、实际可用题数、下载吞吐、镜像与原版逐题等同、原始数据与已有训练/评测的图片重复、SeeClick包装的URL映射，以及模型是否能在当前分辨率读取这些内容。

## 2026-10-25 07:04 +08:00：FUNSD原始包实际获取

已从上述作者固定commit下载dataset.zip，实际16,838,830 bytes，检查zip共824项、含原始实体linking，包内没有LICENSE文件。未将网站模板许可作为数据许可，没有上传或再分发数据。本轮开始exp98：只解析原training_data，按RGB像素哈希隔离train/dev、保守保留唯一question-answer链接和同页四个单链接答案；不从丢失关系的HF BIO包装构造gold。开发集仍为原train派生monitoring，原test不解析。像素审计、实际可用数量与模型结果以[exp98](../experiments/2026-10-25-exp98-funsd-binding.md)及其原始产物为准，不在此预先声称已通过。

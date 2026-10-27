# v0.5 外部基准获取与真实场景训练数据

记录日期：2026-10-23（Asia/Shanghai）。范围：官方页面、发布者仓库和小型 HTTP 元数据查询。未下载数据主体、未运行模型、未提交榜单；以下分数与方法均为发布者陈述，不是本地独立复现。

## 当前决策

ImageJevBench 仍可作为发布目标，但不能以目前已核实的公开文件直接跑“公开 300 题”。官方 results 下载是系统级汇总，不是题集。先补可访问的外部基线并推进独立训练数据，不把榜单接入阻塞当作增加旧域训练预算的理由。

此外，不能把“场景贴近实际”理解为图片全部真实：官方 core 题池包含真实与合成项目。来源：[ImageJevBench 主榜与方法](https://benchmarkheaven.com/image-jev-bench)。

## ImageJevBench：可核实和未核实的边界

官方 v0.3.0 结果 API 当前标注 built_at `2026-10-22T11:45:25.707594+00:00`，source_sha256 `b7e1ba2ee58baba1f442924db9d4760934425e2061fdee88dc94e45d78150bb5`；题池 2,441，其中 real_hf 713、synthetic 1,728；public 300 / sealed 1,200。core 包括 choice、noul、score，sealed 构成为 600 / 300 / 300。发布者定义 Capability 为 Intelligence 与 Calibration 的均值；只测 choice 不能称完整官方 Capability。冻结 admission cap 为 USD 0.3183703881 / 1,000 decisions 和 adjusted p50 1.2329566404 秒。上述为 API 的官方规则，未本地重算。来源：[版本化原始 JSON](https://benchmarkheaven.com/api/decision-results/imagejevbench/v0.3.0/json)。

[官方 data card](https://benchmarkheaven.com/jev-models/data)明确仅提供 system-level measurements，不包含封存题或逐题预测；CSV / JSON 的一行代表一个系统，不应当作为训练或评测样本。当前可用结果入口为上述版本化 JSON，以及同页 CSV 链接。主榜 HTML 超过网页读取工具大小上限；使用搜索抓取的官方页文字作交叉核对，尚未找到可验证的 core public 300 图片、题目、gold、manifest 完整下载。此处是证据缺口，不断言发布者没有公开过。

可进一步检查的发布者代码仓库：[model-market-comparison](https://github.com/fstandhartinger/model-market-comparison)。本次已取得 GitHub recursive tree，找到 `public/image-jev/examples/manifest.json` 与 `lib/image-jev-public-examples.mjs`，但尚未找到 core 300 的完整题集。注意不要误拿另一项目 Leanmcp/jevbench 的 751 道图片题替代本基准；仓库历史说明见 [2026-10-09 调研](jevbench-small-models-2026-10-09.md)。

[正式提交页](https://benchmarkheaven.com/submit)提供免费普通队列，无账户要求；需要模型名、至少一个 GitHub/HF/公共 HTTPS API 链接、联系邮箱和所选 benchmark，代码会接受审核。自托管 GPU loopback timing 在排名中按 `2 × measured + 150 ms` 调整；提供 hosted endpoint 可改测真实网络请求。加急付费不是准入必要条件。本轮没有提交或发送消息。正式被接受上榜受维护者队列与审核影响，不能在本地保证日期。

建议保留两条完成状态：①本地候选达到预写质量门槛；②正式榜单接受。接入前需补：core public 项目的版本与下载方式、完整评分代码和输入 schema，尤其是 score 类接口与本项目接口的对应关系。不能用 ECE 或准确率自行猜出官方 Capability。

### 已找到 8 个公开示例，可做接入冒烟

发布者的 [example manifest](https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/public/image-jev/examples/manifest.json)明确 `status=public_examples_only`、`count=8`，提供 asset URL、来源、revision 与图片 SHA-256。[示例定义代码](https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/lib/image-jev-public-examples.mjs)包含 question、options、correctLabel。可下载 `https://benchmarkheaven.com` 加各 `asset` 路径，做格式与推理冒烟；不得将 8 题结果称为公开 300 或正式分数。其中 UI 示例在截图上画了 A–E 点击标记，任务是选择标记；这提示训练时可以使用相同接口，但应来自独立训练截图。FinQA 示例为表格重渲染图，并非原始文档页。

## 最低成本可落地外部基线：RealWorldQA

第一方 [xai-org/RealworldQA](https://huggingface.co/datasets/xai-org/RealworldQA)只有 `test`，765 行，字段 `image`、`question`、`answer`，许可卡标注 CC-BY-ND-4.0。它没有官方 validation，且同时包含原生选择题与自由回答题；不应为了直接接 decision 接口而默默给自由回答题生成负例，再宣称是原版全量分数。

直接核对 [官方 tree API](https://huggingface.co/api/datasets/xai-org/RealworldQA/tree/main?recursive=true&limit=1000) 得到：

| 文件 | 大小（bytes） | 下载地址 |
| --- | ---: | --- |
| test shard 0 | 337,443,936 | [parquet 0](https://huggingface.co/datasets/xai-org/RealworldQA/resolve/main/data/test-00000-of-00002.parquet) |
| test shard 1 | 340,887,239 | [parquet 1](https://huggingface.co/datasets/xai-org/RealworldQA/resolve/main/data/test-00001-of-00002.parquet) |

合计约 678 MB（十进制）。建议固定 revision 后缓存下载；两份是同一 test split 的分片，不能把一份叫 train、另一份叫 validation。先完整下载再统计能无歧义解析的原生 MC 数量，保留原始 question/answer、索引、图片 hash 和答案映射；报告为 `RealWorldQA native-MC subset`，明确样本数与原版的差别。该 subset 是评测用，不进入训练。

## 训练数据候选与任务适配

以下是可选训练方向，不是已经构建的数据。大多数真实文档/图表/UI 数据原生并非 MC；必须明确改造，不冒充原版评测。

### 优先：ChartQA 原生 Yes/No，再扩充有证据的 choice

作者 [ChartQA 仓库](https://github.com/vis-nlp/ChartQA)提供 train/val/test、图片及问答，完整版本带底层表格和部分图形注释。问答结构为 `imgname, query, label`，训练分 human / augmented；注释有噪声，不能无条件作为完美 gold。作者指向的 [HF 数据仓库](https://huggingface.co/datasets/ahmed-masry/ChartQA)说明完整包需手动下载 zip，不能简单假定 `load_dataset` 就能加载其完整图像结构。现有 HF viewer 显示 `imgname/query/label/type/image`；作者 card 标注 GPL-3.0。

建议最小实验先选 train 中原生 Yes/No 问题，直接转为 noul，无需伪造答案；保留独立图片级 val/test。第二步从同图表格可证伪的数值/类别构造难负例，并固定种子、避免“错误选项长度或数字范围暴露标签”。训练混入既有任务回放，主要看真实图表 held-out NLL / accuracy 和真图–错图差值。原生数值题转 choice 后只能称 ChartQA-derived，不能与官方开放回答 relaxed accuracy 直接比较。

### 文档：DocVQA，数据可用但正式入口有登录

[官方 DocVQA 页](https://site.docvqa.org/datasets/docvqa)描述为真实文档图片上的答案文本跨度。下载在 RRC challenge portal；[官方数据格式页](https://site.docvqa.org/datasets)要求登录并阅读条款，提供 `questionId, question, image, docId, ucsf_document_id, ucsf_document_page_no, answers, data_split`。本次未验证匿名可下载的完整第一方镜像，因此暂不把它设为马上能开训的数据依赖。

适配方案：保留真实文档和问题，候选从同页其他字段中选，过滤 aliases、子串等可能同时正确的负例；按 document 身份隔离，避免同文档不同页泄漏。不能仅把正确答案和随机无关文本配对，否则模型可能靠文本猜测。低分辨率阅读困难必须通过错图对照与固定图像预处理实验确认。

### UI：SeeClick 的独立训练源，ScreenSpot 留作测试

[SeeClick 官方数据说明](https://github.com/njucckevin/SeeClick/blob/main/readme_data.md)提供 RICO/Widget Captioning/RICOSCA，以及 web screenshots。训练 schema 为 `img_filename, instruction, bbox`，web 为图片、URL 和 elements（instruction/bbox/data_type）。完整 web 图片约 130 GB，但作者提供 [10,000 图 subset](https://box.nju.edu.cn/f/813897fc4edc440a9e12/)；本次只验证链接存在，未确认包大小或成功下载。需沿用各原始数据许可，不把代码许可视为所有图片许可。

适配是“指令→候选元素”的 choice，需要可靠同屏候选框及互斥目标；单独给一张截图随机四象限会改变任务。主任务仍是选择目标元素，不在本轮引入 bbox 输出接口。[ScreenSpot 官方介绍](https://github.com/njucckevin/SeeClick)明确是 GUI grounding evaluation benchmark，不能把测试图片拿来训练后再报告外部提升。

## 推荐执行顺序与验收

1. 获取固定 revision 的 RealWorldQA，构建原生 MC 只评测集，记录样本量、图片 hash 与解析排除原因；与 v0.4 固定配置逐题比较。
2. 先做 ChartQA 原生 Yes/No 的小预算监督混合实验，保持模型结构固定，验证视觉依赖、旧任务保持与概率质量；通过后再投入更大规模文档/UI 数据。
3. ImageJevBench 同步解决 core public 下载与 score 接口，拿到材料前状态标记“接入待核实”；不阻塞以上低成本实验，但也不将辅助结果冒称已达到主榜目标。

未验证：任何新模型分数、官方主榜重跑、core 300 全量访问、训练源全量标签与泄漏、DocVQA 账户条款、UI subset 实际下载、CPU 延迟。本文不代表已取得发布资格。

## 追加核查：`public/core/items.jsonl` 线索（2026-10-23）

结论仍是：**目前确认可直接获取的是 8 个公开示例，core 300 题接入仍阻塞；没有取得完整官方题集。** 这不是证明 300 题永远不可获得，而是本次公开发布渠道的核查结果。

- manifest 的 `hash_fields` 确实提到 `public/core/items.jsonl`，但它是解释 hash 的相对文件引用，未给部署主机或 archive 下载 URL。不能仅凭这个字符串认定静态网站存在同名文件。
- 直接 HTTP 请求 `https://benchmarkheaven.com/core/items.jsonl` 和 `https://benchmarkheaven.com/public/core/items.jsonl` 均得到 **404 Not Found**；GitHub raw 的 `main/public/core/items.jsonl` 也返回 404。只核查了该公开路径线索，没有探测封存数据或绕过认证。
- 重新取得发布者 GitHub recursive tree，API 返回 `truncated=false`；image 相关可见文件是页面、aggregate JSON、8 个 examples 与测试等，没有找到该 `public/core` 目录。已读 [.gitignore](https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/.gitignore)，其中没有为该目录提供下载/生成说明。
- [GitHub Releases 页](https://github.com/fstandhartinger/model-market-comparison/releases)显示尚无 releases，因此没有可用的版本化数据 archive 资产。已读 [package.json](https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/package.json) 与 [旧版 build-imagejev 脚本](https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/scripts/build-imagejev-v015.mjs)，没有发现获取/构建 public 300 的命令；旧脚本消费汇总结果，不能用它还原题目。
- 官方 [v0.3 release 测试](https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/test/imagejev-v03-release.test.mjs)明确标注 aggregate-only，并检查结果行不带 gold/predictions；[example 测试](https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/test/image-jev-public-examples.test.mjs)仅校验 8 项公开选择。

不再耗费训练窗口反复试探路径。解除阻塞所需的最小新增材料是：发布者明确给出的 **v0.3 core public 300 下载链接/归档与 manifest**，或负责人已有的合法本地副本。现有 8 项可以作为格式冒烟，正式训练路线仍由独立真实数据上的固定外部评测决定。

## 2026-10-25 06:05 +08:00：提交入口重新核实

本轮实际读取[官方提交页](https://benchmarkheaven.com/submit)：普通免费队列仍按接收顺序处理，所有提交需代码审核，维护者决定评测对象与时间；模型链接、联系邮箱和benchmark选择仍为必填。没有提交或付款，不能保证2026-10-26 09:00前正式接受上榜。主榜页面本轮读取仍触及4MB工具上限，未刷新领先分数，也未因此声称完整public300不存在。public300归档缺口仍沿用此前路径与仓库核查证据，本轮未重复404探测；正式Capability和五用途验收未执行。

## 2026-10-25 07:04 +08:00：正式提交条件再次核对

本轮实际读取[官方提交页](https://benchmarkheaven.com/submit)：普通队列免费、按接收顺序处理，所有提交需要代码审核，维护者决定评测对象与时间；需要模型可访问链接、联系邮箱和benchmark选择。没有执行提交、付款或外发消息。按当前尚未通过正式主指标的证据，不将发布授权理解为必须上传未达标权重；周一09:00前被接受上榜无法保证。public300归档缺口沿用历史核查，本轮没有重新尝试题集下载，不声称本轮证明题集不存在。

## 2026-10-25 16:05:51 +08:00：固定版本接入归档（exp107）

实际获取官方v0.3.0原始结果及发布者commit `4365ca7ec2ee12ea952079185cbf68f37bbf7ff1` 的完整recursive tree（43,352条，未截断），同commit manifest仍为8题public_examples_only；tree未发现public/core路径，未取得public300，不外推不存在。官方聚合source hash与built_at同本文原记录。原响应/哈希/报告：`runs/exp107-official-access/`，实际exit0。当前decide拒绝score的既有测试通过，因此实现符合现有规格，完整正式题型兼容仍待明确官方契约。下一项优先分析本次固定tree的schema/执行代码，不重复猜路径。普通提交免费且审核时间不受本地控制，未发送提交。

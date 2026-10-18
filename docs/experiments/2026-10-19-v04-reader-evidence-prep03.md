# v0.4 读取链证据：确认来源与冻结协议、路由校准验证、reading_combination 接入（V04-READER-EVIDENCE-PREP-03；只用 CPU，不是模型许可）

日期：2026-10-19。依据 V04-READER-EVIDENCE-R2-ACCEPT-01 和盘点（docs/experiments/2026-10-19-v04-reader-evidence-inventory.md）。本文件只是设计，没有改任何运行链或绑定文件，没有加载权重。EXP57 的 fit 失败和组合资格仍待处理；EXP72 的结果不会自动给另一组权重资格；昂贵的全评（约 8.6 小时，按实际 controls 复算约 515 分钟）要等 EXP73 的保护结果之后再定。

## 1. 新读取链的独立确认：来源与冻结协议

### 1.1 来源可得性结论

- **本地真实来源**：没有。原始 ScienceQA 三个划分里的 876 道温度 / 磁力题全部已经暴露：train 533 题用于训练和开发，validation 182 题是每轮都评的 bench_lv_scienceqa，test 161 题是 dev_sqa（见盘点 B）。
- **外部真实来源**：尚未核到可取得、未暴露的匹配来源（R3 NOTE-04 收窄：本轮没有对 IXL、AI2D、TQA 做来源审计，下面两点是未核实的判断，不是穷尽结论）。
  - ScienceQA 的题出自 IXL 的 K-12 题库，原图的许可和获取方式我们没有核到。
  - AI2D、TQA 等公开图解集，凭印象没有这两种「两罐粒子 + 数值表」「两对磁铁 + 距离标注」的格式；没有逐个审计。
  - 所以**真实确认的缺口保留**：交付记录要明确写「新读取链没有在未见过的真实图上独立确认」。如果之后获得了来源（例如负责人取得授权的新题），再按 1.2 的协议执行。
- **合成泛化确认**（可做，但只能称合成确认，不能称未见真实来源确认）：由与开发无关的独立渲染器绘制。

### 1.2 冻结协议（真实来源和合成来源都适用）

1. **先冻结组合**：候选组合（决策端点 + reading_combination，见第 3 节）和全部规则锁定、登记哈希之后，才生成或打开确认集的答案。
2. **抽样规则先写**：
   - 温度 60 张、磁力 60 张；类别的枚举和金标规则见 5.3（原文「温度六类」有误：生成规则只有四类）；
   - 每张一道比较题，再加该图全部读数题；按 sha256(来源 id) 固定抽样，不按模型表现挑选。
   - 数量依据：120 张图合计在 0.8 左右的准确率下，95 % 区间半宽约 ±0.07；每个域 60 张时约 ±0.10。分域照实给区间，不把「大体能用」当统计判定。
3. **去重与暴露隔离**：
   - 真实来源：对已有 876 道 ScienceQA 题和全部训练图做感知哈希（pHash，汉明距离 ≤ 6 只是候选近重复，逐张人工确认后才剔除）；题干是通用模板，相同题干不算重复，不按题干删题；
   - 合成来源：数值组合与 data/real_layout 全部划分、data/exp68 / exp65 / exp70 的 canonical 组合不相交；
   - 确认集只登记哈希，不进入任何训练、开发、校准或方法选择。
4. **合成渲染器的独立性**：
   - 另写一个渲染器（不复用 real_layout 的 temp_render / force_render），使用不同的字体族（例如 DejaVu Sans / Liberation Serif）、画布尺寸、罐与磁铁的形状、标注位置和单位排版；
   - 参数在看任何模型成绩之前冻结并登记；生成后人工目视抽查 20 张，只看绘制是否正确、不看模型输出。
5. **读出**：通过实际评测入口（第 3 节）跑；同时跑配错图和选项倒序。报告准确率、看图收益（真图减错图）、倒序翻转率，温度和磁力分开，并逐题记录读数→表→state。
6. **不缩减原确认项**：confirm_property / confirm_chartqa（普通路径）全部保留；它们证明旧能力，不证明新读取链。

## 2. 路由路径的概率验证（与确认集严格隔离）

- **现状**：全量校准集 data/r13b/val.jsonl（3,892 题）全部是普通路径，路由路径（温度 / 磁力读取后决策）的概率从来没有被校准或验证过。3,892 题的校准不缩减，照原流程做。
- **角色隔离**：
  - 确认集（第 1 节）不参与任何温度拟合，也不用于选温度；
  - rl_confirm（600 行生成数据）保留它「同一生成器、未见过的数值组合」的确认角色，也不拿来拟温；
  - 新增的路由校准集与确认集、开发集按组合和图片两层不相交。
- **新增路由校准集**（生成数据；只是设计）：
  - 用 data/real_layout 生成器，以新的 seed 生成「rl_calib」划分：温度和磁力各 300 族、两视图（共 1,200 张图），canonical 组合与已有全部划分以及确认集不相交；
  - 只用比较题，经实际评测入口走路由路径得到概率。
- **验证方案**：
  1. raw（温度 1）与 fitted（3,892 题拟合出的题型 / 选项数温度）两套概率，都在 rl_calib 上算 NLL、ECE（15 个等宽分箱）和可靠性曲线；
  2. 两折交叉：在 rl_calib 的一半上为路由路径单独拟合温度，在另一半上看 NLL / ECE 是否改善（族为单位分折）；
  3. 判定：如果 fitted 温度在路由路径上的 ECE 不差于 raw，就沿用同一套温度；否则把「路由路径另设温度」作为一项交付改动另行送审（需要在 config 里按路由分桶，decide 实际使用前必须验收）；
  4. 真实图上的概率只能用已暴露的 dev_sqa 作描述，不作校准依据。
- 预计请求（R3 NOTE-03 更正）：1,200 张图各 1 个请求（raw / fitted 只是后处理，共用同一次前向，但两个视图都要跑），按 0.43 秒约 8.6 分钟，另加加载和收尾。不为排程减数据。
- **本节的判定规则以 5.3 为准**（原第 3 条只看 ECE，已被 R3-03 取代）。

## 3. reading_combination 接入发布身份和实际评测入口（最小设计）

### 3.1 现状（只读核对）

- 发布评测的实际入口是 devision-eval（src/devision/train/cli.py eval_main → Decider.load），只走普通 decide。devision.load 和 devision-serve 也只加载 Decider。也就是说，**当前发布流程根本不经过读取链**：温度 / 磁力题在发布评测里会被当作普通题。
- check_eval.py 的 IDENTITY.json 是 tree_hashes(checkpoint)，即 checkpoint 目录下的全部文件。
- bundle 的 reading.json 已经绑定了 policy / grid / templates / domains / corrector 规则和组件、决策端点的 sha256（评审已指出）；代码文件不在其中。

### 3.2 设计

（R3-01 / R3-02 指出本节只换加载对象不够，且「全部规则已核」不实：Reading.load 只核 policy / grid / corrector。补正见第 5 节；两处冲突时以第 5 节为准。）

1. **打包**：把读取组件作为 checkpoint 的子目录 `<checkpoint>/reading/`（reading.json、selector.pt、head.pt、corrector.pt），由 write_bundle 生成。这样 check_eval 的 tree_hashes(checkpoint) 不用改就会把整个组合纳入 IDENTITY。同一份权重的 off / on 是两个不同的 checkpoint 目录（不同的 reading/），筛查和全评证据无法互借。
2. **加载**：新增一个入口 `devision.load(path, reading="auto")`：
   - checkpoint 里有 reading/ 时，返回 Reading.load(path, path/reading)（经验证的入口：先核 reading.json 里的全部 sha256 和规则，再加载 Decider 和组件）；
   - 没有 reading/ 时，与现在完全一样返回 Decider；
   - reading="off" 可以显式关闭（只用于对照）。
   - devision-serve 和 devision-eval 都通过它加载，所以 API、评测、发布用的是同一个对象。
3. **评测入口**：evaluate_records 对每个请求调用这个对象的 decide：
   - 普通题 → Reading.decide 直接转给 Decider（已有行为，夹具已证明与 Decider 相同）；
   - 路由题 → 读取实际展示的图（配错图对照时就是配错的那张图）→ 表 → 普通 decide；
   - 选项倒序 → 只改 criteria 的顺序，路由按题目文字，读数不受影响。
   - 汇总的 run 字段新增 `reading`: {reading.json 的 sha256, 是否带校正器}；check_eval 的内容检查要求它等于 IDENTITY 里 reading/reading.json 的哈希；没有 reading/ 的 checkpoint 必须记为 null。
4. **代码身份**：
   - 规则已在 reading.json 里；代码（src/devision/reading.py、image.py、infer.py、train/evaluate.py）由发布 wheel 的哈希绑定：check_eval freeze 时额外记录当前安装的 devision 版本和 wheel（或源码树）哈希，seal / check 时核对一致；
   - 评测必须在冻结记录的那个安装上跑。
5. **安装核验**（scripts/release/hf.py verify 的扩展）：
   - 在干净的虚拟环境里从 wheel 安装，用发布的 checkpoint（含 reading/）跑固定的 smoke：普通题 2 道、温度 1 道、磁力 1 道，再加一道配错图；
   - 要求读数字符串、表、state 和答案概率与源 checkpoint 在 CPU 上的结果一致；
   - 另核 `reading="off"` 时与 Decider 完全相同。
- **在哪一步加载和验证**：
  - Reading.load 在加载 Decider 之前核对 reading.json 绑定的全部哈希和规则；
  - check_eval freeze 记录 checkpoint 树（含 reading/）和安装哈希；
  - 每个集合跑完后 seal 核对；
  - check 在发布前做最终核对；
  - verify 在干净安装里复现。
- 以上都还没有实现。实现属于发布代码改动（src/devision/__init__.py、train/cli.py、train/evaluate.py、scripts/release/*），需要单独送审，并且要在昂贵全评之前完成，否则全评测到的不是读取链。

## 4. 时间与资格

- 原定 2026-10-18 的期限已经错过，不改历史承诺。约 14–16 小时只是带前提的资源估算，不是新的交付保证，也不包括获取真实确认来源的等待。
- 顺序建议：EXP73 保护结果 → 裁定候选组合和 EXP57 fit 失败的处理 → 实现并验收 3.2 → 路由校准验证（第 2 节）→ 全评 → 合成确认（第 1 节）→ 打包与安装核验。真实确认缺口在交付记录中写明。

## 5. R3 补正（V04-EVIDENCE-R3-01–03；仍只是设计，实现另送）

### 5.1 统一入口（R3-01）

核对的事实（v04-evidence-prep，3887c7f 上的代码）：

- train/evaluate.py:98–108 把图片作为 state 里的图片片段传给 decide，不传 image=；
- reading.py:391–393 在 image=None 时整题退回普通 decide，:396–398 又拒绝带图的 state；
- 所以只换加载对象，eval 会静默评到普通路径；而 HTTP 用 state 图片片段时也会绕过读取，两种图片方式不再等价。
- Reading 没有 predict / cfg / model；eval_main:164–169、evaluate.py:269–271、serve.py:45 都要用 cfg。
- Reading.load（reading.py:292–301）只核 policy / grid / corrector，没有比对 templates / domains；HEADER 和表的写法不在 reading.json 里。

设计：

1. **一个共享的请求解析**：在 infer.py 把 Decider.decide 开头的那段抽成 `resolve_request(state, questions, image) -> (text_state, picture 或 None)`。内容不改，只是抽出来：
   - 题目校验；
   - _split_state；
   - 双图冲突拒绝；
   - _image_arg。
   Decider.decide 改成「resolve_request → 内部 _decide(text_state, questions, picture)」，行为不变。
   Reading.decide 也先调同一个 resolve_request，再按解析出的那张图决定路由：
   - 有图且有支持的题 → 读这张图；
   - 否则 → 交给 decider._decide。
   这样顶层 image= 和 state 图片片段在路由之前就已经是同一个对象，不再有「state 带图就拒绝」或「image=None 就退回」两套规则。HTTP 的规则不变：
   - serve 仍先用 files=False 解析顶层 image；
   - state 图片片段只接受 base64 / url（_load_image 不读文件）；
   - 冲突和坏图仍抛 InvalidRequest，返回 422。
   eval 和 serve 都不再各自拼规则。
2. **公开接口兼容**：Reading 加：
   - `predict`（等同 decide）；
   - `cfg` / `model` / `tok`：返回内部 Decider 的同一个对象，不是副本；
   - `use_raw_temperatures()`：Decider 也加同名方法。它把这个对象实际使用的全部温度置 1，包括 5.2 中路由温度一旦进入 config 后的那一项。
   eval_main 的 --no-temperature 改为调用这个方法，不再直接改 cfg 字段。
   - 因为路由题的内部决策用的就是 self.decider.cfg，所以原始温度对普通题和路由题同时生效；夹具会验证这一点。
   - /health 返回 model_name 和 reading 状态（5.1 第 5 条）。
3. **加载与 Hub**：
   - infer.py 抽出 `resolve_checkpoint(path, revision, token) -> 本地目录`，就是 Decider.load 现有的那段 snapshot_download，Decider.load 也改用它；
   - `devision.load(path, device, revision, token, reading="auto")` 先把 path 解析成本地目录，再检查 `<目录>/reading/`，所以 Hub 上的 repo id 和 revision 也一样走读取链；
   - reading 参数三个取值：
     - auto：有 reading/ 就走读取链，没有就和现在完全一样，返回 Decider；
     - on：没有 reading/ 就报错；
     - off：只加载 Decider，并标明 mode=off。
   - devision-eval、devision-serve 加 --reading（默认 auto）和 --revision，都经 devision.load 加载。
   - `import devision` 仍不加载 torch，reading 只在需要时导入。依赖方向不变：reading 属于推理模块，只引用 infer / image / model；serve 和 train 只引用推理模块。
4. **规则与代码常量一致**：
   - write_bundle 新写 `header`、`table_format`（serialise 的版本名）和 `rules_sha256`（policy / grid / templates / domains / header / corrector 的规范 JSON 的哈希）；
   - Reading.load 逐项比对 templates（按 JSON 往返后的列表比较）、domains、policy、grid、corrector，以及存在时的 header / table_format，与代码常量不一致就拒绝；
   - 旧 bundle（如 EXP71 的 bundle_on）没有 header，仍可经 Reading.load 用于原实验；但发布入口 devision.load 要求这三项都存在，所以发布用的 bundle 必须按 5.2 新写；
   - 夹具：改动 reading.json 里的一个模板就会被拒绝。
5. **记录实际执行的模式，而不是磁盘上有什么**：
   - decide 接受可选的 diagnostics；Decider 也接受，并记 route=ordinary；
   - 每个请求记录：
     - 每题的 route（read / ordinary）；
     - 读数、表和 state；
     - 被读图片的像素哈希：sha256(RGB tobytes) 加尺寸；
   - devision-eval 在逐题明细里写 route，并按它实际发送的文件计算同样的像素哈希，逐请求核对两者相等，以此证明读的是实际展示的那张图，配错图对照时就是配错的那张；
   - 汇总写 `reading = {mode: on|off|none, reading_json_sha256, rules_sha256, corrector, routed_questions, read_questions}`；
   - check_eval 只在以下条件全部满足时，才把一个结果算作 on 组合的证据：
     - checkpoint 带 reading/；
     - mode=on，reading_json_sha256 等于 IDENTITY 里 reading/reading.json 的哈希；
     - 每道 domain_of 不为空的题 route=read（routed_questions == read_questions）；
     - 像素哈希逐请求一致。
   - off 或回退执行的结果不能当 on 的证据。

5.1 的 CPU 验证计划（tiny 模型 + 假读取组件，不加载真实权重）：

- 正式的 evaluate_records 入口上跑普通、读取、混合三种请求，每种都跑真图 / 配错图 / 倒序：
  - 读取题 route=read；
  - 配错图时像素哈希等于配错的那张图；
  - 倒序只改选项顺序，读数相同。
- 顶层 image= 与 state 图片片段：读数、表、state、概率逐项相同。
- 公开接口：
  - predict 可用；
  - /health 返回 reading 状态；
  - 双图请求和坏图返回 422；
  - HTTP 不读服务器本地文件（沿用 tests/model/test_http_image.py 的断言）；
  - use_raw_temperatures 后普通题和路由题的概率都等于温度 1 的结果。
- 没有 reading/ 的 checkpoint 经 devision.load 得到的对象与 Decider 的答案逐位相同；Hub 路径用本地假 snapshot（替换 snapshot_download）验证会检查解析后目录里的 reading/。
- 改一个模板后加载被拒绝；mode=off 的汇总交给 check_eval 时被拒绝作为 on 证据。

### 5.2 身份生命周期（R3-02）

核对的事实：

- train/cli.py:185、:233–254：calibrate --out 只复制固定的 CHECKPOINT_FILES，没有 reading/，随后重写 config 的温度；
- reading.json 绑定整个 config.json（reading.py:295、:297）；
- scripts/release/hf.py:28、:325–336 同样只复制固定文件，还改写 model_name；
- rlcd.calibrate:701–703 直接用 model 前向，不经过读取。

确定的顺序（每一步只允许列出的那些差异）：

1. **来源**：
   - 核 EXP71 的 START / registry 和 bundle_on 三个组件、EXP57 端点三项的哈希，写入 `runs/v04-candidate-meta/source.json`（不放进候选目录本身）；
   - 原实验产物只读，不改。
2. **候选目录**：在新目录 `runs/v04-candidate/` 克隆 EXP57 端点的推理文件，即 CHECKPOINT_FILES。原目录不动。
3. **普通温度**：
   - 在候选目录上用既定的 3,892 题跑 calibrate --in-place；
   - 普通题经 Reading 时原样交给 Decider，这一点已有夹具证明，所以沿用 rlcd.calibrate 的直接前向是对的；
   - 同时把 config 的 model_name 定为发布名。之后不再在构建时改名。
4. **路由温度的选择**（5.3）：
   - rl_calib 的概率只能来自已验证的完整入口：在候选目录之外写一个临时 bundle（`runs/v04-candidate-meta/calib_bundle/`，绑定第 3 步的 config），经 Reading.load 这个已验证的入口和 evaluate_records，以 use_raw_temperatures 跑出原始概率，route / 像素哈希按 5.1 第 5 条核对；
   - 拟合时用 log p 当 logits：温度 1 的 softmax 输出取对数，与 logits 只差一个常数，所以拟合出的温度不变；
   - 不用 rlcd.calibrate 去处理 rl_calib，也不另写读取前向；
   - 只有 5.3 的分支选中「另设路由温度」时，才往 config 写 `temperature_by_route`（新字段；decide 只把它用在路由题上；旧 config 没有这个字段，行为不变；这个分支需要另行送审实现）；
   - 写完 config 才算最终定稿。临时 bundle 绑定的是定稿前的 config，按设计会被第 5 步拒绝，只用来产生拟温数据，不作为任何结果的证据。
5. **重新绑定**：
   - 对定稿后的候选目录用 write_bundle 生成 `<候选>/reading/`，组件从 EXP71 的 bundle_on 原样复制；
   - 脚本 `scripts/release/rebind.py` 核对并写 `runs/v04-candidate-meta/lineage.json`：
     - 三个组件的哈希等于 EXP71 的记录；
     - model.safetensors、tokenizer、encoder、vision 等于 EXP57；
     - config 与 EXP57 的 JSON 差异只能是 {temperature, temperature_by_options, model_name, temperature_by_route}，任何其他差异都拒绝；
     - 规则等于代码常量。
   - 温度变了，所以概率证据必须属于这个新的身份：EXP71 / EXP73 的成绩只作为开发和保护的依据，不当作新身份的全评。
6. **冻结**：check_eval freeze 记录 tree_hashes(候选)，即整个候选目录（推理文件加 reading/；source / lineage 在目录外，另记它们的哈希），同时记录代码身份：
   - 运行身份 = 评测进程在跑完之后，sys.modules 里所有 devision.* 模块的「模块名 → 实际文件路径 + 内容 sha256」，加上 `devision.__version__` 和 Python 版本；
   - freeze 用同一解释器、同一安装跑一个小探针，记下参考身份；每个评测汇总写入它自己进程的运行身份，seal / check 要求按模块名逐个相等；
   - 只看没被加载的 wheel 文件哈希不算数。
7. **正式评测**：只在冻结后的候选上跑，经 devision.load(reading="on")。
8. **构建**：
   - hf.build 遇到带 reading/ 的候选时，整树复制推理文件（含 reading/），不改 config；
   - 改为核对 config 里的 model_name 等于发布名，不相等就拒绝，因为名字已经在第 3 步定好；
   - 构建完成后核对输出里的推理文件和 IDENTITY 逐个相同；
   - 没有 reading/ 的旧发布（v0.2 / v0.3）保持原有流程。
9. **干净安装核验**：hf.py verify 在新虚拟环境里从 wheel 安装，跑固定 smoke（普通 2 题、温度 1 题、磁力 1 题、配错图 1 题），要求：
   - 运行身份按模块名的内容哈希等于 freeze 的参考身份；
   - 读数、表、state 和概率与源候选在 CPU 上的结果一致；
   - reading="off" 时与 Decider 完全相同。
- **校准入口的边界**：calibrate 遇到带 reading/ 的 checkpoint，不论 --out 还是 --in-place 都拒绝，并提示「先校准，再重新绑定」。这样不会静默丢掉 reading/，也不会让 reading.json 绑着旧 config。

5.2 的 CPU 验证计划（tiny）：

- 正向流程：校准副本 → 定 model_name → 重新绑定 → freeze → 发布副本。用 devision.load 能以 mode=on 加载；组件不丢；lineage 记下的 config 差异只有允许的那几个字段。
- 以下情况要被拒绝：
  - 绑定旧 config 的 bundle；
  - 换掉一个组件；
  - config 多出一个不允许的差异；
  - 未登记的 off 输出；
  - 对带 reading/ 的 checkpoint 跑 calibrate。
- build 前后推理文件的哈希不变；model_name 与发布名不符时 build 拒绝。
- 运行身份：改动一个被导入模块的一个字节后，seal 拒绝。

### 5.3 结局规则先写定（R3-03）

**路由校准**（rl_calib，1,200 张图 = 600 族 × 两视图，每张图一道比较题；统计单位是族，同一族的两个视图一起重抽；bootstrap 用 seed 0、2,000 次）：

- 三套概率：raw（温度 1）、ordinary（第 3 步的普通拟温）、routed（路由单独拟温，每个域一个温度）。
  - routed 只在样本外评：按 sha256(族 id) 分两折，在一折上拟合、在另一折上评，两折的结果合起来。
  - 三套概率都来自同一次完整入口前向的原始概率，身份都记录在案。
- **主判据是样本外 NLL**，温度 / 磁力分开报告，也报合并结果：
  - Δ_R = NLL(routed) − NLL(ordinary)；
  - Δ_O = NLL(ordinary) − NLL(raw)；
  - 两者都是按族配对的差值和 95 % 区间。
- 辅助指标：ECE（15 个等宽分箱）和可靠性曲线，只用来描述，不单独改变分支。
- 容许幅度预先定为 **0.02 nats / 题**（约为 ln 2 的 3 %，在看到任何成绩之前写定）。
- 三个分支：
  - 另设 Δ_B = NLL(routed) − NLL(raw)（R4 补正）：直接用同一批样本外逐题差值、按族配对重抽得到区间，不能把 Δ_R 和 Δ_O 两个区间的端点相加。
  1. **另设路由温度**：以下三条同时成立。
     - 合并后 Δ_R ≤ −0.02 且区间上界 < 0；
     - 每个域的 Δ_R 点估计 ≤ +0.01；
     - 合并后 Δ_B 的区间上界 ≤ +0.02，即新温度不比 raw 明显差。
     - 成立时，用全部 rl_calib 拟合出的每域温度写入 config（5.2 第 4 步），再重新绑定、冻结；
     - 这项改动需要另行送审实现，才能进入产品。
  2. **保留现行温度**：分支 1 不成立，并且：
     - 合并后 Δ_R 的区间下界 ≥ −0.02，即路由单独拟温最多带来不到 0.02 的改善；
     - Δ_O 的区间上界 ≤ +0.02，即现行温度不比 raw 明显差。
  3. **明确退化，校准未通过**：分支 1 不成立，并且合并后 Δ_O 的区间下界 > +0.02，即现行温度在路由路径上明确比 raw 差，超过容许线。
     - 现行温度不能作为路由路径上可发布的温度，这项校准资格到此停止，记为未通过；
     - 文件保留，只用于诊断；
     - 下一步怎么选温度，另行裁定（例如另一种拟合方式或改用 raw），并重新按本节规则验证。
     - 不自动放行发布。
  4. **证据不足**：其余所有情况，包括区间太宽、无法排除退化、跨过容许幅度。
     - 产品配置不改，但这项校准仍是未完成的交付项，不自动放行发布；
     - 交付记录写明「路由路径的概率校准未得到验证」，不能宣称概率校准完成；
     - 区间跨 0 不等于已证实等效。
- 反例（文档层面，能直接从规则推出，不是实测）：
  - ordinary − raw = +0.10 [+0.09, +0.11]，routed − ordinary = −0.03 [−0.04, −0.02]，每个域都同向：
    - Δ_B 直接重抽约 +0.07，上界 > +0.02，所以不走分支 1；
    - Δ_O 下界 +0.09 > +0.02，所以走分支 3：明确退化，校准未通过；
    - 不能标为校准合格，也不能保留 ordinary 当作可发布的温度。
  - Δ_O = [−0.01, +0.05]，Δ_R = [−0.06, +0.01]：
    - Δ_R 上界不小于 0，不走分支 1；
    - Δ_R 下界 < −0.02 且 Δ_O 上界 > +0.02，不走分支 2；
    - Δ_O 下界 ≤ +0.02，不走分支 3；
    - 所以走分支 4：证据不足。
  - 「ECE 下降但 NLL 升高 0.05」→ Δ_R > 0 → 不走分支 1；剩下的分支只按 Δ_O 和 Δ_R 判定，ECE 不参与。
  - 只有 ECE、没有 NLL → 结果不完整，没有结论。

**合成确认**（120 张图，独立渲染器）：

- **用途**：「新读取链能泛化到另一种绘制风格」的合成支持证据。
  - 它**不是**发布资格的独立确认，填补不了真实来源缺口；
  - 也不替代、不降低正式门槛：开发迁移（EXP71，按 68:114 规则）、保护（EXP73 等）和全评。
- **同题基线**：v0.3（普通 Decider，不带读取）在同 120 张图、同样的题、同样的配错图上作答。这需要一次单独的模型许可（约 240 个请求，2–3 分钟）。
- **主要判定**（每张图的一道比较题，以图为单位按图配对，bootstrap 用 seed 0、2,000 次）。形式上借用 68:114，但口径不同：这里的看图收益是候选自身「真图减错图」，而 68:114 用的是相对 v0.3 的看图收益增量。这是另行预写的合成支持判据，不能替代正式迁移门槛：

  - 候选减 v0.3 的准确率差 ≥ +10 点，且区间下界 > 0；
  - 并且候选的看图收益（真图准确率减配错图准确率）区间下界 > 0。
- 次要分项：
  - 读数题（温度每图 4 题、磁力每图 2 题）的完全一致率，按图聚类给区间，不把同一张图的多道读数当作新增的独立样本；
  - 两个域分开描述，每域 60 张，区间约 ±0.10。
- 预写结局：
  - **合成确认通过**：主要判定成立。只能写「合成确认通过」，不能写成真实确认。
  - **未通过**：结果完整有效，但主要判定不成立。当前读取组合不获准发布，交付记录写明原因。
  - **无结论**：有缺失（incomplete）或无效（invalid）。与未通过同样处理，当前读取组合不获准发布。
  - 未通过或无结论只否决当前组合，不产生任何别的发布许可（R4 补正）：
    - 去掉读取链的 off 是另一个组合，必须独立满足候选资格和对应证据，不能借 on 的保护或全评；
    - 只去掉读取链也不能宣称完成了「有明显进步的 v0.4」；
    - 没有合格的替代候选时，保留 v0.3，不以研究包代替目标版本。
  - 合成确认通过也不解决 EXP57 的 fit 资格，真实来源缺口照旧保留。
- 所有选择都在确认成绩揭示之前完成：组合冻结（5.2 第 6 步）之后才生成确认集答案。
- **类别和金标**（来自 scripts/data/real_layout.py 的事实定义，由独立渲染器重新实现；数值范围相同，只换绘制方式）：
  - 温度共四类，即 temp_scene 的 case，每类 15 张：
    - same mass：两样本质量相同、速度不同；
    - same speed：速度相同、质量不同；
    - both differ：两项都不同，且 M·V² 不相等；
    - equal：两项都相同。
    - 金标：M·V² 较大的那个样本温度较高；相等时选 TEMP_SAME。
    - 前三类按 sha256(图 id) 平衡 A / B 哪个是答案。
  - 磁力共三类，即 force_scene 的 case，每类 20 张：
    - distance：磁铁都是大的，距离不同；距离较近的那一对磁力较大；
    - size：距离相同，其中一对里有一块小磁铁；两块都是大磁铁的那一对磁力较大；
    - equal：距离和大小都相同；两对磁力相同。
    - 措辞四种（magnitude greater / smaller、strength stronger / weaker），金标按措辞翻转，与 force_items 相同。
  - 已知限制：读取表里只有距离，没有磁铁大小，所以 size 类的比较题在读取链上得不到大小信息。它照样计入主要判定，因为那就是产品的真实行为，同时单独报告。

### 5.4 实现顺序与边界

- 实现放在一个从 develop 拉出的新分支，不在 devision-exp72 / devision-exp73 的工作区里做，也不改它们绑定的文件。
- 顺序：5.1 和 5.2 的代码与 tiny 夹具 → 送审。验收之后才谈任何模型运行：
  - rl_calib 的生成与读出；
  - 合成确认（独立渲染器加 v0.3 基线）；
  - 全评。
  每一项都要单独申请许可。
- EXP57 的 fit 失败和组合资格仍待处理；EXP73 的保护结果仍是全评的前提。

### 5.5 实现注意（R4 建议，不扩大实验）

- 运行身份：
  - 按模块名和内容比较，路径只用来追溯；
  - 必须覆盖的模块集合预先固定，即正式评测进程实际载入的全部 devision.* 模块；只比较两边的交集不算数，探针少载一个模块要能被发现；
  - CPU 夹具同时覆盖正式 eval 和干净安装两种入口。
- log p 当作 logits 的前提是原始概率有限且严格为正：
  - 遇到 0 或非有限值时明确拒绝，不能静默夹紧后仍声称等价；
  - 或者改从已有前向直接输出稳定的 log-prob；
  - 用 tiny 夹具验证。
- 合成确认的统计权重：以图为单位，每张图一道比较题，权重相同。

### 5.6 5.1 / 5.2 的 CPU 实现（分支 v04-reading-entry，基线 3f22b8d；不加载真实权重）

- 基线说明：develop（a9022cd）缺少读取链代码（reading.py 的校正器等在 exp 分支上），所以从 v04-evidence-prep 拉分支；它与 develop 只差 develop 上的一篇 exp57 结果文档。
- 已实现：
  - infer.resolve_request / resolve_checkpoint / picture_identity；
  - Decider.use_raw_temperatures 与 diagnostics；
  - Reading 的 predict / cfg / model / tok / use_raw_temperatures、先统一解析再路由、route 和像素记录、规则比对（check_rules；发布入口 full_rules）；
  - write_bundle 写入 header / table_format / rules_sha256；
  - devision.load(reading=auto|on|off)；devision-eval / devision-serve 的 --reading / --revision；/health 返回 reading 状态；
  - evaluate 记录 route / reading_domain / 像素哈希，逐请求核对，汇总写 reading；devision-eval 记录 runtime_code；
  - 带 reading/ 时 calibrate 拒绝运行；
  - scripts/release/rebind.py（clone / bind，lineage）；
  - check_eval：freeze 记录 reading 和代码参考，参考来自新进程的探针；content 检查 on 证据和代码身份（实际载入的模块必须都在参考里、内容相同，必需模块必须载入）；结果目录落在绑定的 checkpoint 之内时 freeze 拒绝；
  - hf.build 整树复制、不改 config、核对推理文件。
- 实现时发现：只评测的 YAML 中，checkpoint 的上一级没有 run.json 时，结果会写进 checkpoint 目录本身，会改变被冻结的树。所以候选必须放在带 run.json 的轮次目录下（例如 runs/v04-candidate/stage），freeze 已对这种情况拒绝。
- 夹具：tests/model/test_reading_entry.py（12 项）、tests/release/test_reading_lifecycle.py（7 项）。
- 尚未实现：
  - 干净安装核验（hf.py verify 在新虚拟环境中从 wheel 安装，再按模块名比对运行身份）：比对函数 code_problems 已有，安装流程本身未接；
  - temperature_by_route：只在 5.3 选中分支 1 后另行送审；
  - rl_calib 用 log p 拟合时的有限性检查：属于拟温工具，跑之前随工具一起实现。

### 5.7 实现第 2 轮（V04-ENTRY-R1-01–03）

- R1-01：新增 devision.reading.bundle_state，判定 bundle 是 none / complete / incomplete。只有完全没有 reading/ 才是普通模型；有 reading/ 但缺清单或缺组件就是 incomplete。devision.load（auto / on）、check_eval freeze 和带读取组合的 build 都用这个判定，遇到 incomplete 一律拒绝；显式 off 照常加载，并标明 mode=off、bundle=incomplete。
- R1-02：check_eval 的 load_data 按源题的 instructions，用 domain_of 算出每题的期望域。reading_problems 对每对已匹配的「记录—源题」逐题核对：
  - 明细里的 reading_domain 要等于期望域；
  - 应读取的题 route=read，其他题 route=ordinary；
  - routed / read 计数从源题重算，不用明细自报的数。
- R1-03：
  - rebind.py source 核对固定来源：EXP71 的 START / registry / report 哈希固定在代码里（PINS），给定的 bundle 和决策目录必须正是 START 记录的 bundles.on，然后写 `<round>/reading_lineage/source.json`；
  - clone 和 bind 都从 source.json 出发，并重新核对源文件；lineage 记录 source.json 的哈希；
  - freeze 把 source / lineage 的哈希写进 IDENTITY，preflight / seal / check 每次重新核对；
  - build 先读 `<round>/eval/IDENTITY.json`，核对候选和记录都与冻结时一致；构建后逐个推理文件对照 IDENTITY 检查。
- 干净安装核验（PENDING-01）仍是独立的后续项。
- 第 3 轮（V04-ENTRY-R2）：rebind.verify_source 是唯一的只读来源核验，核两件事：
  - 固定运行文件在磁盘上的实际哈希等于 PINS；
  - bundle 和决策目录正是该 START 的 bundles.on。
  source 写入时、clone / bind 使用 source.json 时、lineage_problems（freeze / preflight / seal / check / build）每次都调用它。check_source 另外要求记录的 pins 等于规定来源，记录自己的 bundle_on 等于 START.bundles.on。

# exp56 方案：实时自动读取接入普通推理（零训练；CPU 准备完成，未运行，等评论家的模型许可）

日期：2026-10-18。依据：评论家 EXP56-PREP-01–03（EXP53 / EXP55 阳性分支）。模式说明见 `docs/spec/reading-mode.md`。

## 实现（`src/devision/reading.py`，可选模式；默认推理不变）

- **组件**：选择器、字符头、区域规则、框规则、严格解码、千位格式，从实验脚本原样迁入推理模块。模块不导入 scripts / train / 评测数据。
  - CPU 夹具用随机权重核对：Selector、CharHead 与原类的 state_dict 互通，输出逐位相同；region、pixel_box、decode、fmt 在若干输入上与原函数相同。
- **路径**：真实图片 → `visual_inputs`（每个请求一次，全图加 12 视图）→ 前 256 个 K/V 转 float16 作为选择器的 patch；问题词嵌入取自决策模型冻结的输入嵌入，转 float16 → 选择器 → 框规则 → region 取 K/V（float16 后转 float）→ 字符头 → 严格解码 → fmt → 表 → `decide(文字 + 表, 受支持的题, image=None)`。
- **路由**：只看自然问题文本（6 个读数模板加 2 种比较问句）。读取集合是该领域的全部模板，与题号和选项顺序无关。其他题走带图的普通 decide；没有受支持的题或没有图片时，就是普通 decide。
- **默认和响应结构**：默认 Decider、HTTP 和发布包都不变；响应只有 model / answers / usage，诊断信息走可选的 diagnostics 字典。
- **读数包**（`write_bundle`）：复制 selector.pt / head.pt，reading.json 用 sha256 绑定它们，以及决策 checkpoint 的权重、config、tokenizer 目录、策略和网格；`Reading.load` 逐项核对，不符就拒绝，并要求 checkpoint 有 12 / 0.25 / T 的局部视图。
- **CPU 夹具**（tests/model/test_reading.py，5 项）：
  - 组件与原实现相同；
  - 路由：单个读数题、只有比较题、同图多题都读该领域；题号 / 选项顺序改变时读取集合不变；未知问题不读取；大小写和空白不影响；
  - 未知请求和无图请求与普通 decide 相同；混合请求中不受支持的题与单独用普通 decide 的答案相同；
  - 题号 / 选项顺序改变时，读到的框和表逐项相同；
  - 只访问问题的 instructions / type / criteria 三个字段（带 id、kind、gold、image 字段的题目字典核对过）；
  - 响应只有 Jev 字段；保存 / 加载后答案相同；读数包文件被改时拒绝；没有局部视图的 checkpoint 拒绝。

## 运行（`scripts/exp56/exp56.py`、`run.sh`）

- **start**（无模型）：
  - 核 EXP53 的 selector（8f10f1fd，registry）、EXP51 的 head_T（0b548142，registry）、EXP55 的 START 和 registry 中全部文件、EXP50-T 的实际权重、题单 / rl_tune、EXP53 的 details_auto；
  - 构建 runs/exp56/bundle；START 记录 HEAD、9 个文件的 sha、读数包的全部 sha 和基线文件。
- **live**（MPS）：
  - **P**：600 张原图，每张图一个请求（原题全部，image = 图片），通过 `Reading.decide` 这条产品路径；保存框、原始解码、表、实际 state、输出和每个请求的耗时（reads.jsonl）。
  - **E / D**（只在诊断 runner 里做干预）：E 把实时表的值全部改为 unreadable，D 用固定孪生图的实时表，经普通文字 decide。
  - 每个条件完成后登记。
- **report**（无模型）。

## 预写判据

- **读取保真**：1,800 道读数题的实时解码字符串与 EXP53 相同的比例 ≥ 99%，读取准确率之差的绝对值 ≤ 0.01。
- **决策保真**：实时 P 表文本与 EXP55 P 输入逐字相同的请求，预测相同，概率最大差 ≤ 1e-5（对照 EXP55 的 details_P）。容差沿用 EXP55 实测为 0 的惯例，没有理由放宽。
- **使用门槛**：照用 EXP54 / 55 的 G / P / P − E / P − D。
  - G 取 EXP55 冻结的 details_G（核 registry，本轮不重跑）；
  - read_right 仍是 EXP53 冻结的 1,470 题；
  - 供体计数去掉分隔符后比较。
- **分支**：
  - 保真不满足时记"实时路径没有复现"，不换权重、不调模板；
  - 保真满足时，按 EXP54 的分支（G 和 P 都过且依赖内容时，进入比较监督和迁移 / 保护筛查的准备）。
- **其他**：比较题只报告；外部迁移不因整合成功自动放行；不跑 G、fit / L 或新确认集。

## 预算（估计）

| 阶段 | 请求数 | 估计 |
|---|---|---|
| start | 无模型 | < 0.5 分钟 |
| 加载 | 决策模型约 2 GB，读数包约 9 MB | 约 0.5 分钟 |
| P | 600：每次视觉编码 13 张图，约 0.25 秒（EXP50 评测实测额外开销）；选择器加字符头最多 4 次读取，可忽略；文字 decide 约 0.16 秒（EXP55 实测 6.1 个请求 / 秒） | 600 × 约 0.45 秒 ≈ 4.5 分钟 |
| E / D | 1,200 次文字请求 | 约 3.3 分钟 |
| 报告 | | 约 1 分钟 |
| **合计** | | **约 10 分钟** |

- 硬上限 30 分钟，最后 120 秒留给报告；watch 截止比外层早 35 秒，宽限 10 秒。
- 内存：决策模型约 2 GB，每张图的 K/V 3,328 × 768 × 4 B ≈ 10 MB；峰值预计 ≤ 4 GB。
- 每个请求的 P50 / P95 耗时和读取耗时都写入报告。CPU 部署时延另行批准后再测，不与 MPS 运行争资源。

## CPU 验证

- tests/model/test_reading.py：5 项；scripts/exp56/test_exp56.py：3 项（start 拒绝错误组件；读取保真规则用 EXP53 自身字符串时复现，改 2% 时判失败；真实 watch 的短链在截止时停止并 finalize）。
- 全仓 `uv run pytest -q` 281 项通过，pyright 0 错误。

## 状态

CPU 准备完成，**未运行**。等评论家的 EXP56-RUN-01。v0.4 仍需真实迁移、保护、全量校准与评测、独立确认、CPU / API / 安装和远端核验；新组合的全评时间要实测，不承诺今晚 23:59 交付。

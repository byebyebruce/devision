# Jev（TypeSafe AI）API 调研

- 调研日期：2026-09-30
- 一手来源：TypeSafe 官方文档 `docs.typesafe.ai`（通过 `https://docs.typesafe.ai/llms.txt` 索引，逐页下载 `.md` 原文）、官方博客、官方 SDK 仓库 `github.com/typesafe-ai/*`；Laya 部分用 Convai/作者的 GitHub README、`laya/serve.py` 源码与 HF 模型卡。
- `systemonemodels.org` 自称“independent site, not affiliated with TypeSafe AI or any model vendor”，因此只当作 Laya 的索引页使用，不作为 Jev 事实来源（[来源](https://systemonemodels.org/models/laya/)）。
- 文中 API 字段名和代码保持原文；标注“（推断）”的内容是我自己的推导，文档里没有直接写。

---

## 1. Jev 是什么

### 1.1 概念

- Jev 是 TypeSafe 的旗舰模型，也是第一个 “System One model”：接收 *state* 和一组带类型的 *questions*，直接返回结构化答案，不生成文本，调用方也不用解析输出（[Introduction](https://docs.typesafe.ai/introduction.md)）。
- 官方博客的原话：“a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out”（[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。
- 发布日期为 2026-09-15，作者是创始人 Diogo Almeida，目前处于 early access（[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。
- 博客提到新的模型架构、“parallel sampler”，训练方法叫 RLCD（Reinforcement Learning for Calibrated Decisions）；采样方式是“Parallel. Generates all outputs in a single query”，而 LLM 是逐 token 顺序生成（[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。
- FAQ 里只有一句：“Jev is neither small nor an LLM”。架构细节和参数量都没有公开（[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)，FAQ 折叠内容，从页面 HTML 中提取）。
- RLCD 的输出约定是：不生成文本，只返回决策和概率；“Higher probability should correspond to a greater chance that the answer is correct”（[AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)）。
- Jev 不针对客户做 fine-tune 或 LoRA，所有账户共用同一套权重；领域适配只能通过 `state`、`instructions`、`criteria` 完成（[Models](https://docs.typesafe.ai/models.md)）。
- 训练数据全部自产，不用客户数据训练（[博客 FAQ](https://typesafe.ai/blog/introducing-system-one-models-and-jev)，[Models](https://docs.typesafe.ai/models.md#data-handling)）。

### 1.2 输入和输出

- **输入**：`state` 只能是文本，可以是字符串、JSON 对象或数组（[State](https://docs.typesafe.ai/concepts/state.md)）。
- **输出**：每个 question 对应一个带类型的答案（`choice` / `score` / `noul`）以及概率。Choice 和 Score 还带 `confidence`（[API reference](https://docs.typesafe.ai/api.md)）。
- 答案只会落在调用方给出的选项或等级里，“never a value outside them”（[Primitives](https://docs.typesafe.ai/primitives.md)）。

### 1.3 图像和多模态：不支持（重点）

- Models 页的规格表写的是：“**Input: Text only. String, JSON object, or array of text values. No image, audio, or video input.**”（[Models](https://docs.typesafe.ai/models.md)）。
- 同一页的补充说明：“Pre-process non-text inputs (images, audio, video, binaries) into text or structured fields before sending them as `state`.”（[Models](https://docs.typesafe.ai/models.md)）。
- State 页原文：“Jev accepts text only… Images, audio, and video are not supported (yet).”（[State](https://docs.typesafe.ai/concepts/state.md)）。System One 页的说法相同（[System One](https://docs.typesafe.ai/concepts/system-one.md)）。
- 博客里 Doom demo 的说明：“The demo is on structured state as a data structure with text, not on images (yet…)”（[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。
- 结论：Jev 1.13 只接受文本；“yet” 暗示以后可能支持，但没有给出时间表。

---

## 2. API 格式

### 2.1 端点和认证

| 项 | 值 | 来源 |
|---|---|---|
| 评估端点 | `POST https://api.typesafe.ai/v1/systemone` | [API](https://docs.typesafe.ai/api.md) |
| 认证 | `Authorization: Bearer <API_KEY>`；在 console 创建 key，环境变量为 `TYPESAFE_API_KEY` | [API](https://docs.typesafe.ai/api.md)，[Python SDK](https://docs.typesafe.ai/sdk/python.md) |
| Content-Type | `application/json` | [API](https://docs.typesafe.ai/api.md) |
| 模型列表 | `GET https://api.typesafe.ai/v1/models`，返回 `{"models":[{name, description, release_date}]}`；目前只列出别名，但版本 ID 也可以直接用 | [Models](https://docs.typesafe.ai/models.md) |
| 默认 base URL / 默认模型 / SDK 超时 | `https://api.typesafe.ai` / `jev-latest` / 每次 HTTP 操作 10.0 秒 | [Python constants](https://docs.typesafe.ai/sdk/python/api/constants.md) |

### 2.2 请求体

| 字段 | 类型 | 必填 | 说明 | 来源 |
|---|---|---|---|---|
| `state` | `string \| object \| array` | 是 | 要评估的内容 | [API](https://docs.typesafe.ai/api.md) |
| `model` | `string` | 是（HTTP 层；SDK 默认填 `jev-latest`） | 例如 `"jev-latest"`、`"jev-1.13.0"` | [API](https://docs.typesafe.ai/api.md) |
| `questions` | `map<string, Question>` | 是 | key 由调用方自定，答案按同一 key 返回。**key 不会发给模型，不参与推理** | [API](https://docs.typesafe.ai/api.md) |

Question 的公共字段是 `type` 和 `instructions`。`instructions` 可以是 string、object 或 array：把问题放在一个字段、引用数据放在其他字段，然后在问题里用反引号写字段路径，例如 `` `potential_duplicate` `` 或 `` `items[3]` ``（[API](https://docs.typesafe.ai/api.md)，[Primitives](https://docs.typesafe.ai/primitives.md)）。

**Noul（是/否）**

- `type: "noul"`
- `instructions`：必填
- `criteria`：可选，形如 `{ "true": <string|object|array>, "false": <...> }`，分别描述“是”和“否”各指什么

来源：[API](https://docs.typesafe.ai/api.md)

**Choice（单选）**

- `type: "choice"`
- `instructions`：必填
- `criteria`：必填，类型为 `map<option, string|object|array|null>`。value 为 `null` 表示该选项不需要额外描述
- **每个 Choice 最多 255 个选项**

来源：[API](https://docs.typesafe.ai/api.md)

选项名和描述**都会发给模型**（[Choice](https://docs.typesafe.ai/primitives/choice.md)）。

**Score（有序等级）**

- `type: "score"`
- `instructions`：必填
- `criteria`：必填，是一个有序数组，从低到高排列
- 至少 2 级，**API 最多接受 10 级**

来源：[API](https://docs.typesafe.ai/api.md)，[Score](https://docs.typesafe.ai/primitives/score.md)

等级编号就是数组下标，从 0 开始。“Every level is evaluated separately. The model doesn't see a level's number or its neighbours”（[Score](https://docs.typesafe.ai/primitives/score.md)）。

> SDK 与 HTTP 文档不一致：JS SDK v0.6.0 的 `types.ts` 把 `instructions` 标成可选（`instructions?: EntryType`，可以为 `null`），`EntryType` 也允许 `null`；它的 `ScoreCriteria` 是“at least two”的元组，并且允许 `null` 等级（[types.ts](https://github.com/typesafe-ai/typesafe-sdk-js/blob/v0.6.0/src/types.ts)）。HTTP 文档则写 `instructions` required。以 HTTP 文档为准。服务端是否接受 `null`，未核实。

### 2.3 响应体

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 实际执行的版本 ID，例如 `"jev-1.13.0"`，即使请求用的是别名 |
| `answers` | `map<string, Answer>` | 每个问题一个答案，key 与请求中的 key 相同 |
| `usage` | `{input_tokens: int, output_tokens: int}` | token 用量 |

来源：[API](https://docs.typesafe.ai/api.md)，[Models](https://docs.typesafe.ai/models.md)

各类答案的字段：

- **Noul 答案**
  - `type: "noul"`
  - `noul: number`：取值 0–1，表示答案为“是”的概率
  - **没有 `confidence`**
- **Choice 答案**
  - `type: "choice"`
  - `choice: string`：概率最高的选项
  - `probabilities: map<option, number>`：覆盖所有选项，和为 1
  - `confidence: number`：取值 0–1
- **Score 答案**
  - `type: "score"`
  - `score: number`：概率加权期望 Σ i·p_i，可以落在两级之间
  - `legend: map<"0"|"1"|..., description>`
  - `probabilities: map<"0"|"1"|..., number>`：key 是字符串形式的等级下标，和为 1
  - `confidence: number`
  - Python SDK 会把 `probabilities` 和 `legend` 的 key 转成 int

来源：[API](https://docs.typesafe.ai/api.md)，[Score](https://docs.typesafe.ai/primitives/score.md)

### 2.4 原文示例（Quickstart，逐字复制）

请求（[Quickstart](https://docs.typesafe.ai/introduction/quickstart.md)）：

```json
{
  "state": "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
  "model": "jev-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this",
      "criteria": {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": "Pricing or account questions"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated the customer appears",
      "criteria": [
        "Calm, just stating facts",
        "Frustrated but civil",
        "Very angry, strong language"
      ]
    },
    "is_urgent": {
      "type": "noul",
      "instructions": "The message conveys urgency or time-sensitivity"
    }
  }
}
```

响应：

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "confidence": 0.78,
      "probabilities": {
        "technical": 0.85,
        "sales": 0.0,
        "billing": 0.15
      }
    },
    "frustration": {
      "type": "score",
      "score": 1.0,
      "confidence": 1.0,
      "legend": {
        "0": "Calm, just stating facts",
        "1": "Frustrated but civil",
        "2": "Very angry, strong language"
      },
      "probabilities": {
        "0": 0.0,
        "1": 1.0,
        "2": 0.0
      }
    },
    "is_urgent": {
      "type": "noul",
      "noul": 1.0
    }
  },
  "usage": {
    "input_tokens": 392,
    "output_tokens": 65
  }
}
```

Score 被“夹在两级之间”的例子（[Score](https://docs.typesafe.ai/primitives/score.md)）：

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "bug_severity": {
      "type": "score",
      "score": 1.43,
      "confidence": 0.35,
      "legend": {
        "0": "Cosmetic; no impact to functionality",
        "1": "Broken or degraded feature, but workaround exists",
        "2": "Blocking issue; no workaround exists"
      },
      "probabilities": {
        "0": 0.0,
        "1": 0.57,
        "2": 0.43
      }
    }
  },
  "usage": {
    "input_tokens": 332,
    "output_tokens": 18
  }
}
```

Noul 带 `criteria` 的请求（[API](https://docs.typesafe.ai/api.md)）：

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?",
      "criteria": {
        "true": "Explicitly time-sensitive",
        "false": "No urgency expressed"
      }
    }
  }
}
```

结构化 `instructions`（[API](https://docs.typesafe.ai/api.md)）：

```json
"instructions": {
  "potential_duplicate": {
    "name": "John Smith",
    "location": "Oakland, California",
    "last_employer": "Google"
  },
  "question": "Is the resume for the same person as `potential_duplicate`?"
}
```

### 2.5 错误码与重试

- 错误码（[API](https://docs.typesafe.ai/api.md)）：
  - `401`：key 缺失或无效
  - `422`：请求体校验失败，响应 body 会指出出错的字段
  - `429`：超出限流
  - `529`：服务过载
- 429 和 529 应该用指数退避重试（[API](https://docs.typesafe.ai/api.md)）。
- SDK 默认会自动重试，并遵守 `retry-after` 头（[Models](https://docs.typesafe.ai/models.md)）。
- JS SDK 的默认重试策略：
  - 最多重试 2 次
  - 退避从 500ms 开始，上限 5000ms，jitter 0.25
  - 重试的状态码：408、429、500–599
  - 遵守 `Retry-After` 和 `retry-after-ms`，上限 60 秒

  来源：[types.ts](https://github.com/typesafe-ai/typesafe-sdk-js/blob/v0.6.0/src/types.ts)

### 2.6 SDK

| 语言 | 安装 | 调用方式 | 来源 |
|---|---|---|---|
| Python | `pip install typesafe-sdk`（可选 `[http2]`） | `TypeSafeClient` / `AsyncTypeSafeClient`，`client.system_one(state=..., questions={...})`；问题对象为 `Noul`、`Choice`、`Score`、`NoulCriteria`；便捷访问器 `response.nouls[...]`、`.choices[...]`、`.scores[...]` | [Python SDK](https://docs.typesafe.ai/sdk/python.md)，[源码](https://github.com/typesafe-ai/typesafe-sdk-python) |
| JS/TS | `npm install @typesafe-ai/sdk`（Node ≥ 20） | `new TypeSafeClient().systemOne({state, questions})`，辅助函数 `choice()`、`noul()`、`score()`；答案类型由问题推断 | [JS SDK](https://docs.typesafe.ai/sdk/javascript.md)，[types.ts](https://github.com/typesafe-ai/typesafe-sdk-js/blob/v0.6.0/src/types.ts) |

- JS SDK 默认禁止在浏览器中使用；必须设置 `dangerouslyAllowBrowser` 才能开启（[types.ts](https://github.com/typesafe-ai/typesafe-sdk-js/blob/v0.6.0/src/types.ts)）。
- 另有面向编码 agent 的 skill，仓库为 `github.com/typesafe-ai/skills`（[Agent skill](https://docs.typesafe.ai/agent-skill.md)）。

---

## 3. 问题类型的语义与限制

### 3.1 三种问题类型

| 类型 | 适用场景 | 语义要点 |
|---|---|---|
| Choice | 无序的固定集合 | 相对判断，回答的是“哪一个”。选项列表可能覆盖不全时，建议加 `other` 或 `none of the above`；每增加一个选项“costs a few tokens” |
| Score | 可以分级描述的谱系 | 各级分别评估，模型看不到等级编号，也看不到相邻等级，所以描述要写“情境”，不要写“程度”；`score` 是期望值 |
| Noul | 清晰的是/否判断 | 绝对判断。`noul` 表示“是”的概率，不是程度量表；0.5 的意思是“是和否各半”，不是“中等” |

来源：[Primitives](https://docs.typesafe.ai/primitives.md)，[Choice](https://docs.typesafe.ai/primitives/choice.md)，[Score](https://docs.typesafe.ai/primitives/score.md)，[Noul](https://docs.typesafe.ai/primitives/noul.md)

### 3.2 执行语义

- 同一请求里的所有问题共享同一个 `state`，并行评估，彼此独立。
- 一个问题的答案不会成为另一个问题的上下文。
- 增加问题几乎不影响延迟，也不会造成 context-rot。

来源：[Introduction](https://docs.typesafe.ai/introduction.md)，[Primitives](https://docs.typesafe.ai/primitives.md)

### 3.3 不保证结构不变性

- 同一个问题分别用 Noul 和二元 Choice 来问，数值可以不同：Noul 为 0.22，Choice `yes` 为 0.01。
- `P(q)` 与 `P(not q)` 之和可以是 1.19。
- 文档明确说这些不变性“aren't guaranteed”。

来源：[Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)

### 3.4 限制

| 项 | 值 | 来源 |
|---|---|---|
| Choice 选项数 | 最多 255 | [API](https://docs.typesafe.ai/api.md)，[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| Score 等级数 | 2–10 | [API](https://docs.typesafe.ai/api.md) |
| 上下文 | 每个请求 64k tokens（`state` 加上所有问题）；`state` 加上最长的单个问题不超过 32k tokens | [Models](https://docs.typesafe.ai/models.md) |
| 每个请求的问题数 | **没有写明硬上限**，只受上面的 64k 预算约束 | [Models](https://docs.typesafe.ai/models.md) |
| 语言 | 主训练语言是英语；其他语言（包括 CJK）可以处理，但准确度较低，需要自测，并更依赖 confidence | [Models](https://docs.typesafe.ai/models.md#language-support) |
| 已知弱项 | 字面化理解、数学和计数、日期比较、多跳间接推理、大量无关 state、对抗内容、指令与 criteria 矛盾、文本生成 | [jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) |
| 不要用 score 做数值插值 | “score levels are weak in numerical calibration” | [jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) |

博客补充说明：更高基数的选择（Wikiracing）在应用层做成两阶段——先独立打分，再显式做 Choice（[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。

---

## 4. 校准、置信度、延迟与价格

### 4.1 校准

- 官方声明：“Calibrated: higher confidence means higher accuracy. More consistent: returns similar answers for similar inputs.”（[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）
- 校准是在一组预测上衡量的，对单个答案不作保证（[System One](https://docs.typesafe.ai/concepts/system-one.md)，[AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)）。
- 官方**没有公布** ECE、Brier 等校准指标，也没有可靠性曲线。
- 官方刻意不公布 public benchmark 成绩（[博客 FAQ](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）。

### 4.2 `confidence` 的算法

文档的说法：

- `confidence` 是从 `probabilities` 的分布形状推导出的统计量：全部概率集中在一个选项时为 1.0，分布越平越低。
- 文档交互示例对 3 个选项使用 `(3 × largest probability − 1) / 2`，并注明这是 “approximate”。

来源：[Confidence](https://docs.typesafe.ai/confidence.md)

推广到 n 个选项：

- 示例组件代码写的是 `(count * peak - 1) / (count - 1)`，并截断到 [0, 1]（[Confidence](https://docs.typesafe.ai/confidence.md) 页面源码）。
- Laya README 也把 Jev 的算法描述为 `(n·p_max − 1)/(n − 1)`（[Laya README](https://github.com/NandhaKishorM/laya#self-hosting-http-server-jev-compatible)）。

（推断）用官方示例验算：

- p=0.85, n=3 → 0.775，响应里是 0.78
- p=0.88 → 0.82，响应里是 0.81
- p=0.95 → 0.925，响应里是 0.92
- p=0.57 → 0.355，响应里是 0.35

误差都在四舍五入范围内，因此基本可以确定生产环境用的就是 `(n·p_max − 1)/(n − 1)`。但文档没有正式声明，只称之为 “approximate”。

### 4.3 其他

- 官方建议把 confidence 分成三段——自动执行 / 谨慎执行 / 转人工，阈值按风险高低设定（[Confidence](https://docs.typesafe.ai/confidence.md)）。
- 如果按某个版本调好了阈值，应当固定版本 ID，不要用别名（[Models](https://docs.typesafe.ai/models.md)）。

### 4.4 延迟

- 官方声明端到端 **70ms–500ms**，在 System One 类查询上比前沿 LLM 快 40–200 倍。
- 官方评测“run from our laptops on the West Coast”。
- 首页的 “193.6x faster, 444.6x cheaper” 来自 workflow evals，官方自己说“on the higher end”。

来源：[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

第三方测得 p50 为 236–276ms，引自 Laya HF 卡，属于二手信息（[HF card](https://huggingface.co/convaiinnovations/laya)）。

### 4.5 价格与限流

| 项 | 值 | 来源 |
|---|---|---|
| 输入 | $0.042 / Mtok（$42 / Btok） | [Models](https://docs.typesafe.ai/models.md)，[博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| 输出 | 免费 | 同上 |
| 限流 | 100K tokens/s，40 requests/s；“adjusting dynamically… can change without notice”；企业版可提高 | [Models](https://docs.typesafe.ai/models.md) |
| 数据 | 不用客户数据训练；企业版提供 ZDR | [Models](https://docs.typesafe.ai/models.md)，[Legal](https://docs.typesafe.ai/legal.md) |

### 4.6 模型版本

- 当前只有 `jev-1.13.0`。
- `jev-latest` 和 `jev-preview` 都指向它，目前没有 preview 版本。

来源：[Models](https://docs.typesafe.ai/models.md)

---

## 5. Laya / laya-serve 与 Jev API 的对应关系

Laya 是 Convai Innovations 的开源权重模型（Apache 2.0）。它是一个 encoder 加决策头：ModernBERT-large，共 421M 参数；多语言版基于 mmBERT-base，共 322M 参数。每个选项在自己的 `[MASK]` 位置打分，再在该问题的所有选项上做 softmax；另有一个 act/escalate 头（[HF card](https://huggingface.co/convaiinnovations/laya)）。

### 5.1 兼容点

- `laya-serve`（`pip install "laya[serve]"`）暴露同样的 `POST /v1/systemone` 协议。README 称答案 payload 与 Jev “schema-identical”：`choice`/`score`/`noul` 答案加上 `{input_tokens, output_tokens}`，现有 Jev 客户端只需改 `baseUrl`（[README](https://github.com/NandhaKishorM/laya#self-hosting-http-server-jev-compatible)）。
- HF 卡称它接受 Jev API 支持的所有问题形态，会忽略未知字段，问题格式错误时返回 422（[HF card](https://huggingface.co/convaiinnovations/laya)）。
- 设置 `LAYA_API_KEY` 后要求 `Authorization: Bearer <key>`；未设置时**不做认证**（[README](https://github.com/NandhaKishorM/laya)）。

### 5.2 差异

| 维度 | Jev | laya-serve | 来源 |
|---|---|---|---|
| `model` 字段 | 必填，取值如 `jev-*` | 只在值为 `english` / `multilingual` / `typed-decisions` 时生效，否则按脚本和语言自动路由 | [README](https://github.com/NandhaKishorM/laya) |
| 额外请求字段 | 无 | `task`、`lang`、`lang_guess`、`min_confidence`、`max_len`、`head_max_len` | [serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py) |
| 批量端点 | 无 | `POST /v1/systemone/batch`，传 `states` 数组（最多 64 个），返回 `results` 和 `total_usage` | [README](https://github.com/NandhaKishorM/laya) |
| 额外响应字段 | 无 | 顶层 `routing`；答案可带 `answer_confidence`（= max p），以及 `low_confidence`（设置了 `min_confidence` 时出现）；响应头 `X-Inference-Time-Ms` | [serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py)，[README](https://github.com/NandhaKishorM/laya) |
| `confidence` 定义 | `(n·p_max−1)/(n−1)` | 1 − 归一化熵，**阈值不能从 Jev 直接迁移过来** | [README](https://github.com/NandhaKishorM/laya) |
| Choice 选项数 | ≤ 255 | 服务端硬上限 100（`MAX_CHOICE_OPTIONS`，超出返回 413）；实际还受 `head_max_len` 的 token 预算约束，超过约 20 个选项时描述会被截断 | [README](https://github.com/NandhaKishorM/laya)，[serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py) |
| Score 等级 | 2–10 | `MAX_SCORE_LEVELS = 32`；**`null` 等级返回 422** | [serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py)，[README](https://github.com/NandhaKishorM/laya) |
| 每个请求的问题数 | 未写明 | `MAX_QUESTIONS = 64`；`MAX_STATE_CHARS = 50000`；body ≤ 2 MiB | [serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py) |
| 上下文 | 64k / 32k | 512（英文版，`head_max_len` 192）或 1024（多语言版，256），多语言版最多可调到 8192 | [HF card](https://huggingface.co/convaiinnovations/laya) |
| 错误码 | 401/422/429/529 | 400/401/413/422/500/503（busy，带 `Retry-After`） | [serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py)，[README](https://github.com/NandhaKishorM/laya) |
| `GET /v1/models` | 有 | 源码里没有看到，只有 `GET /health` | [serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py) |
| Noul 扩展 | 无 | 可选的 `labels`，用来改模型看到的 false/true 文本；可选的 `option_order` | [README](https://github.com/NandhaKishorM/laya#decision-primitives) |
| 图像 | 不支持 | 不支持（纯文本 encoder） | [HF card](https://huggingface.co/convaiinnovations/laya) |

- `output_tokens` 在 laya 的批量端点固定为 0（[serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py)）。
- 质量方面：README 自称是“a fast base to specialise, not a zero-shot decision engine”，出厂状态偏过度自信，需要自己拟合 temperature（[README](https://github.com/NandhaKishorM/laya)）。

---

## 6. 对“视觉 + 决策模型、输出 Jev 同款格式”的设计启示

以下是依据上文事实做的推断。

### 6.1 输出头必须产出的内容（逐问题）

**Noul**

- 输出一个标量 `noul` ∈ [0, 1]，含义是 P(yes)。
- 不输出 confidence。
- 可以用一个二分类 logit 加 sigmoid，或者对 [false, true] 两个槽做 softmax 后取 true 的概率（Laya 的做法）。

**Choice**

- 在请求时才给出的 K 个选项（K ≤ 255）上输出分布 `probabilities`，和为 1。
- `choice` = argmax。
- `confidence` = `(K·p_max−1)/(K−1)`，截断到 [0, 1]。
- 因为选项集合是请求时动态定义的，输出头不能是固定维度的分类器。它必须把“选项名 + 描述”编码后逐个打分，再做 softmax，例如 Laya 的做法：每个选项一个 `[MASK]`（[HF card](https://huggingface.co/convaiinnovations/laya)）。
- 选项描述的类型可能是 string、object、array 或 null，要能处理。

**Score**

- 在 L 个有序等级（2 ≤ L ≤ 10）上输出分布，key 为 `"0".."L-1"`。
- `score` = Σ i·p_i。
- `legend` 原样回显 criteria。
- `confidence` 的公式与 Choice 相同，n = L。
- 官方说明各级是分别评估的，模型看不到编号。因此结构上可以复用 Choice 的逐选项打分头，只在后处理中计算期望（[Score](https://docs.typesafe.ai/primitives/score.md)）。

**顶层**

- 顶层结构为 `{model, answers:{<qid>: ...}, usage:{input_tokens, output_tokens}}`。
- qid 不能作为模型输入，因为 Jev 的语义就是“key 不发给模型”。

### 6.2 多问题

- 同一个 state 编码一次，多个问题相互独立并行评估。
- 要避免问题之间互相串信息，每个问题单独做 cross-attention 或单独打分。
- 如果要严格模仿“答案互不影响”，那么增删问题不应改变其他问题的结果。

### 6.3 校准

- 所有概率都必须经过校准训练：用 proper scoring rule，例如 log-loss 或 Brier，再加上 temperature scaling。
- 要评估 ECE。
- 确定性很重要，Jev 声称“extremely consistent”。

### 6.4 图像输入是新增能力

- Jev 的 `state` 只接受文本。官方给的路径是先把图像转成文本或结构化字段再发送（[Models](https://docs.typesafe.ai/models.md)）。
- 要做视觉版，有两种选择：
  - 在 `state` 中定义自己的图像扩展，例如 `{"image": {"url"|"b64": ...}, ...}`。这会破坏与 Jev 请求格式的兼容，但**响应格式可以完全兼容**。
  - 走“视觉 → 结构化文本 state → Jev 兼容模型”的级联方案，这样请求和响应都兼容。
- 注意 Laya 的 422 策略：遇到未知字段时，它会忽略顶层的未知字段；但如果把图像塞进 `state`，Jev 会按文本处理。

### 6.5 限制与错误码对齐

建议对齐的限制和错误码：

- 选项 ≤ 255
- 等级 2–10
- 校验错误用 422，限流用 429，过载用 529
- 模型名别名机制，响应里返回版本 ID
- `GET /v1/models`

### 6.6 延迟目标

参照官方的 70–500ms 端到端延迟；Laya 在 T4 上约 33ms。

---

## 7. 未核实 / 缺失

- **问题数上限**：文档没有给出每个请求最多多少个问题，只有 64k token 预算。
- **校准指标**：官方没有公布 ECE、Brier 或可靠性图；Laya README 里的 Jev 数字是第三方数据，Laya 作者自己也声明“never measured here”（[HF card](https://huggingface.co/convaiinnovations/laya)）。
- **`confidence` 精确公式**：官方只称 “approximate”；我根据示例验算出的 `(n·p_max−1)/(n−1)` 属于推断。
- **概率精度**：示例都保留两位小数（出现 0.0、1.0）；API 是否总是四舍五入到 0.01，未核实。
- **`output_tokens` 的含义和计费**：输出免费，但示例里有 18–65 个 output tokens，这些 token 代表什么，文档没有解释。
- **错误响应 body 的 JSON 结构**：只说了 “JSON body describing what went wrong”，没有给 schema。
- **`instructions` 可空性**：HTTP 文档写 required，JS SDK 写 optional/nullable，服务端实际行为未验证。
- **Score 的 `null` 等级在 Jev 上是否允许**：JS SDK 允许，HTTP 文档写的是 `array<string|object|array>`，与 SDK 不一致，未验证。
- **重复选项名、空 criteria、Choice 只有 1 个选项**等边界情况：文档没有写。
- **架构、参数量、streaming、批量端点、webhook**：官方都没有公开，也没有看到 Jev 有 batch 端点。
- **多模态路线图**：只有 “not supported (yet)”，没有时间表。
- **未读的页面**：博客提到的 playground 分享链接（需要 early access）；[workflow evals 站](https://evals.typesafe.ai/)；[How to build with TypeSafe](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md)；Python `responses` 页全文；各 cookbook。
- **页面加载情况**：所有目标页面都成功加载（HTTP 200），没有需要换替代源的失败页面。`docs.typesafe.ai/api` 除了 WebFetch 以外，还通过 `/api.md` 原文核对过。博客 FAQ 的答案是折叠内容，我从页面内嵌 JSON 中提取。

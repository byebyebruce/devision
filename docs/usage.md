# 使用指南

deVision 是一个 HTTP 接口（也可以在 Python 里直接调用）：给一张图和几道英文题，返回每道题的答案和校准过的概率。请求和响应格式与 Jev `/v1/systemone` 相同，唯一的扩展是 `state` 里可以放一张图。

## 安装

```bash
pip install "devision[serve] @ git+https://github.com/byebyebruce/devision"   # 推理 + HTTP 服务
pip install "devision @ git+https://github.com/byebyebruce/devision"          # 只在 Python 里推理
```

| 安装项 | 包含 |
|---|---|
| `devision` | 推理：`devision.load`、`Decider.predict` / `decide`、`devision-eval` |
| `devision[serve]` | 加上 HTTP 服务和 web demo（`devision-serve`） |
| `devision[train]` | 加上训练（`devision-align`、`devision-train`，依赖 peft、SwanLab） |

在本仓库里开发用 `uv sync`，开发依赖组已经包含上面所有内容。

## 在 Python 里调用

```python
import devision

decider = devision.load("runs/stage2-cocoqa")   # 本地 checkpoint 目录，或 Hugging Face 仓库 id
# devision.load("user/repo", revision="v0.1", token="hf_...")   # 固定版本 / 私有仓库

result = decider.predict(            # 和 decider.decide(...) 完全相同，predict 是 Laya 的叫法
    state=[{"type": "image", "url": "https://example.com/kitchen.jpg"}],
    questions={
        "has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
        "room": {"type": "choice", "instructions": "Which room is this?",
                 "criteria": {"kitchen": None, "bathroom": "has a sink and a toilet", "bedroom": None}},
    },
)
```

请求不合法时抛出 `devision.InvalidRequest`（HTTP 服务里对应 422）。

## 启动 HTTP 服务

```bash
devision-serve --checkpoint runs/stage2-cocoqa --port 8000
```

| 参数 | 作用 |
|---|---|
| `--checkpoint` | 本地目录或 Hugging Face 仓库 id |
| `--revision` | 仓库 id 时固定的 commit、tag 或分支 |
| `--host` / `--port` | 默认 `127.0.0.1:8000` |
| `--threads` | CPU 推理线程数 |
| `--no-demo` | 只开接口，不挂 web demo |

接口：

- `POST /v1/systemone`：做决策，见下文。
- `GET /health`：返回 `{"status": "ok", "model": "devision-0.1"}`。
- `GET /`：web demo，可以上传图片、填题目、看概率（`--no-demo` 关闭）。

只跑 CPU，Mac 上单题大约 0.2 秒；同一请求里的多道题共用一次图片编码。

## 请求

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": [
    {"type": "image", "url": "https://example.com/kitchen.jpg"},
    "customer says the order arrived damaged"
  ],
  "questions": {
    "has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
    "room": {
      "type": "choice",
      "instructions": "Which room is this?",
      "criteria": {"kitchen": null, "bathroom": "has a sink and a toilet", "bedroom": null}
    }
  }
}'
```

- **`state`**：字符串、对象或数组。数组里最多一张图：`{"type": "image", "url": "https://..."}` 或 `{"type": "image", "base64": "..."}`；其余元素是文字背景信息。不带图时等同于纯文本的 Laya。
- **`questions`**：对象，键是你自己起的题目 id，可以一次问多道。
  - `noul`：是非题，`instructions` 写要判断的陈述或问题；`criteria` 可选，用 `{"false": "...", "true": "..."}` 改写两个选项的含义。
  - `choice`：单选题，`criteria` 的键是选项（2–255 个），值是 `null` 或一句说明，帮模型区分相近的选项。
  - `score` 不支持，返回 422。

## 响应

```json
{
  "model": "devision-0.1",
  "answers": {
    "has_fork": {"type": "noul", "noul": 0.03},
    "room": {
      "type": "choice",
      "choice": "kitchen",
      "probabilities": {"kitchen": 0.81, "bathroom": 0.07, "bedroom": 0.12},
      "confidence": 0.72
    }
  },
  "usage": {"input_tokens": 180, "output_tokens": 0}
}
```

- `noul`：陈述成立的概率。
- `probabilities`：每个选项的概率，和为 1；`choice` 是概率最高的选项。
- `confidence`：按 Jev 的公式 `(n·p_max − 1)/(n − 1)`，0 表示和均匀猜一样，1 表示完全确定。
- 错误：请求格式不对、图片取不到或解码失败、选项放不进模型输入时，返回 422 和 `{"error": "..."}`。

## 怎么用得好

**用概率做阈值，没把握的转给人或更强的模型。** 概率是校准过的：说 80% 时大约八成是对的。

```python
p = result["answers"]["has_fork"]["noul"]
if p > 0.9:
    handle_yes()
elif p < 0.1:
    handle_no()
else:
    escalate()   # 人工，或者更强的模型（系统 2）
```

阈值请在自己的数据上定：看不同阈值下自动处理的比例和错误率。

**适合问的：**

- 图里有没有某样东西：有没有人、有没有破损、有没有某种物品。
- 是什么、属于哪类：房间类型、动物种类，选项不宜太多。
- 整体判断：室内还是室外、东西看起来是不是坏了。

**不要问的：**

- 某个东西在哪里、在另一个东西的哪一边。这类题现在是随机水平。
- 数数、读图中的小字（输入只有 256×256）。
- 非英文问题、一次多张图、`score` 打分题。

**题目写清楚，选项互不重叠。** 选项意思接近时在 `criteria` 里加一句说明。

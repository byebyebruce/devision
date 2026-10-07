# deVision

**deVision = Decision + Vision**：看图做决策。输入一张图片和若干英文问题，输出 Jev 格式的答案（`noul` 是非题 / `choice` 选择题）和校准过的概率。SigLIP2 看图，Laya 的 ModernBERT 决策头打分，不生成文字；没有 GPU 也能跑。

模型在 Hugging Face：[`lukbit/devision`](https://huggingface.co/lukbit/devision)（当前 v0.2）。

## 安装

```bash
pip install "devision @ git+https://github.com/byebyebruce/devision"            # 推理 + HTTP 服务
pip install "devision[train] @ git+https://github.com/byebyebruce/devision"     # 再加训练
```

需要 Python 3.11+。在本仓库里开发用 `uv sync`。

## Python

```python
import devision

decider = devision.load("lukbit/devision")   # Hugging Face 上的最新版；也可传本地 checkpoint 目录
# 固定版本 revision="v0.2"；设备默认 auto（CUDA > MPS > CPU），可指定 device="cpu"

result = decider.predict(            # 与 decider.decide(...) 相同，predict 是 Laya 的叫法
    image="kitchen.jpg",
    state="",                        # 文字背景；没有就传 ""
    questions={
        "has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
        "room": {"type": "choice", "instructions": "Which room is this?",
                 "criteria": {"kitchen": None, "bathroom": "has a sink and a toilet", "bedroom": None}},
    },
)
print(result["answers"]["has_fork"]["noul"])        # 陈述成立的概率
print(result["answers"]["room"]["choice"])          # 概率最高的选项
print(result["answers"]["room"]["probabilities"])   # 每个选项的概率，和为 1
```

- `image`：本地路径、http(s) URL、Data URI、Base64、`bytes` 或 PIL 图片。
- `state`：文字背景，字符串、对象或数组，和 Jev 一样。
- 请求不合法时抛出 `devision.InvalidRequest`。

## HTTP 服务

```bash
devision-serve --checkpoint lukbit/devision --port 8000   # 浏览器打开 http://127.0.0.1:8000 是 web demo
```

常用参数：`--revision v0.2` 固定版本，`--device cpu|cuda|mps`（默认自动），`--no-demo` 只开接口。

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "image": "https://example.com/kitchen.jpg",
  "state": "customer says the order arrived damaged",
  "questions": {
    "has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
    "room": {"type": "choice", "instructions": "Which room is this?",
             "criteria": {"kitchen": null, "bathroom": "has a sink and a toilet", "bedroom": null}}
  }
}'
```

```json
{
  "model": "devision-v0.2",
  "answers": {
    "has_fork": {"type": "noul", "noul": 0.03},
    "room": {"type": "choice", "choice": "kitchen",
             "probabilities": {"kitchen": 0.81, "bathroom": 0.07, "bedroom": 0.12}, "confidence": 0.72}
  },
  "usage": {"input_tokens": 180, "output_tokens": 0}
}
```

- 请求：`image` 是 http(s) URL、Data URI 或 Base64（不读服务器本地文件）；`choice` 有 2–255 个选项，`criteria` 的值可以加一句说明帮模型区分；不支持 `score`。
- 响应与 Jev 相同：`noul` 是陈述成立的概率；`confidence = (n·p_max − 1)/(n − 1)`。请求不合法返回 422。
- 速度：Mac CPU 上单题约 0.15 秒；同一请求里多道题共用一次图片编码。

## 适合做什么

用概率做阈值，没把握的转给人或更强的模型；阈值请在自己的数据上验证。

- **适合**：有没有某物、有几个（用选择题问）、是什么类别、颜色材质、室内室外、照片里的上下左右（用选择题问）。
- **不适合**：示意图、图表、地图上比较两处；相对位置的是非题；图中小字（输入 256×256）；非英文、多张图。

## 效果

v0.2 与 laya-vision 201M 在同一批题上逐题比较（差值 95% 区间按图片重采样）：

| 评测集 | 题数 | laya-vision | deVision v0.2 | 差值 |
|---|---|---|---|---|
| POPE random / popular / adversarial | 3 × 3,000 | 0.836 / 0.819 / 0.777 | **0.891 / 0.868 / 0.791** | +5.5 / +4.9 / +1.4 |
| VQAv2 是非题 | 4,887 | 0.717 | **0.725** | +0.8 [−0.9, +2.3] |
| A-OKVQA | 1,138 | 0.598 | **0.626** | +2.7 [−0.6, +6.0] |
| ScienceQA（带图题） | 2,097 | **0.824** | 0.766 | −5.8 [−7.9, −3.6] |

完整结果和每一轮的训练记录见 [`docs/training-log.md`](docs/training-log.md)，模型结构和设计见 [`docs/spec/vision-decision-mvp.md`](docs/spec/vision-decision-mvp.md)。

## 训练

```bash
uv run devision-pipeline configs/example.yaml    # 用示例数据跑通一遍，CPU 上一两分钟
bash scripts/data/build_all.sh data              # 从公开数据集生成训练数据（几十 GB、几个小时）
uv run devision-pipeline configs/scratch.yaml    # 从头训练
```

每一轮训练是 `configs/` 里的一个 YAML，再跑同一个 YAML 是续跑。数据格式见 [`examples/train/`](examples/train/)，发布版 v0.2 的配置是 [`configs/v12-ext.yaml`](configs/v12-ext.yaml)。

## 许可

代码和权重为 [Apache-2.0](LICENSE)。训练数据各有条款，部分非商用（如 ScienceQA），请按用途自行核对。

<p align="center"><img src="https://raw.githubusercontent.com/byebyebruce/devision/master/assets/logo.png" alt="deVision" width="200"></p>

# deVision

**deVision = Decision + Vision**：看图做决策。输入一张图片和若干英文问题，输出 Jev 格式的答案（`noul` 是非题 / `choice` 选择题）和校准过的概率。SigLIP2 看图，Laya 的 ModernBERT 决策头打分，不生成文字；没有 GPU 也能跑。

模型在 Hugging Face：[`lukbit/devision`](https://huggingface.co/lukbit/devision)（当前模型 **v0.3**）；ModelScope：[`baibaibai000/devision`](https://modelscope.cn/models/baibaibai000/devision)。

## 安装

```bash
pip install devision            # 推理 + HTTP 服务
pip install "devision[train]"   # 再加训练
```

需要 Python 3.11+。在本仓库里开发用 `uv sync`。

## Python

```python
import devision

decider = devision.load("lukbit/devision")   # Hugging Face 上的最新版；也可传本地 checkpoint 目录
# 设备默认 auto（CUDA > MPS > CPU），可指定 device="cpu"

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
devision-serve   # 浏览器打开 http://127.0.0.1:8000 是 web demo
```

仓库内运行 `uv run devision-serve`。默认加载 `lukbit/devision`，也可用 `--checkpoint` 指定其他模型仓库或本地 checkpoint 目录。

常用参数：`--device cpu|cuda|mps`（默认自动），`--no-demo` 只开接口。

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
  "model": "devision-v0.3",
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
- 速度：Mac CPU 上单题 P50 约 0.175 秒、P95 约 0.190 秒（4 线程、FP32、200 道单题请求，排除预热）；同一请求里多道题共用一次图片编码。

## 适合做什么

用概率做阈值，没把握的转给人或更强的模型；阈值请在自己的数据上验证。

- **适合**：有没有某物、有几个（用选择题问）、是什么类别、颜色材质、室内室外、照片里的上下左右（用选择题问）。
- **不适合**：示意图、图表、地图上比较两处；相对位置的是非题；图中小字（输入 256×256）；非英文、多张图。

## 效果

v0.3 与 laya-vision 201M 在同一批题上逐题比较（差值 95% 区间按图片重采样；POPE 仅比较公开汇总分，没有配对区间）：

| 评测集 | 题数 | laya-vision | deVision v0.3 | 差值 |
|---|---|---|---|---|
| POPE random / popular / adversarial | 3 × 3,000 | 0.836 / 0.819 / 0.777 | **0.880 / 0.859 / 0.805** | +4.4 / +4.0 / +2.8 |
| VQAv2 是非题 | 4,887 | 0.717 | **0.725** | +0.8 [−0.7, +2.4] |
| A-OKVQA | 1,138 | 0.598 | **0.612** | +1.4 [−1.8, +4.7] |
| ScienceQA（带图题） | 2,097 | **0.824** | 0.789 | −3.5 [−5.6, −1.4] |

相对 v0.2，CountBench +16.7、镜像关系 +2.9、ScienceQA +2.3 个百分点。v0.3 是负责人接受取舍后的采用：四项旧能力保护未通过，VSR 召回下降、A-OKVQA 可能下降，计数对选项顺序更敏感。这些集合已多轮用于实验决策，不是新的独立测试；详见 [采用说明](release/v0.3/adoption.md)。

完整结果和每一轮的训练记录见 [`docs/training-log.md`](https://github.com/byebyebruce/devision/blob/master/docs/training-log.md)，模型结构和设计见 [`docs/spec/vision-decision-mvp.md`](https://github.com/byebyebruce/devision/blob/master/docs/spec/vision-decision-mvp.md)。

## 训练

```bash
uv run devision-pipeline configs/example.yaml    # 用示例数据跑通一遍，CPU 上一两分钟
bash scripts/data/build_all.sh data              # 从公开数据集生成训练数据（几十 GB、几个小时）
uv run devision-pipeline configs/scratch.yaml    # 从头训练
```

每一轮训练是 `configs/` 里的一个 YAML，再跑同一个 YAML 是续跑。数据格式见 [`examples/train/`](https://github.com/byebyebruce/devision/tree/master/examples/train)，发布模型 v0.3 的配置是 [`configs/r15-vision.yaml`](https://github.com/byebyebruce/devision/blob/master/configs/r15-vision.yaml)（从 v0.2 继续训练，不是从零训练）。

## 许可

代码和权重为 [Apache-2.0](https://github.com/byebyebruce/devision/blob/master/LICENSE)。训练数据各有条款，部分非商用（如 ScienceQA），请按用途自行核对。

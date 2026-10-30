<p align="center"><img src="https://raw.githubusercontent.com/byebyebruce/devision/master/assets/logo.png" alt="deVision" width="200"></p>

# deVision

**deVision = Decision + Vision**：看图做决策。输入一张图片和若干英文问题，输出 Jev 格式的答案（`noul` 是非题 / `choice` 选择题）和校准过的概率。SigLIP2 看图，Laya 的 ModernBERT 决策头打分，不生成文字；没有 GPU 也能跑。

模型在 Hugging Face：[`lukbit/devision`](https://huggingface.co/lukbit/devision)（当前模型 **v0.5**）；ModelScope：[`baibaibai000/devision`](https://modelscope.cn/models/baibaibai000/devision)。

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
  "model": "devision-v0.5",
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
- **不适合**：复杂示意图、密集图表和地图上的细节比较；相对位置的是非题；图中小字（输入 256×256）；非英文、多张图。

## 效果

v0.5（r24）相对 v0.4，在同一批题上按图片配对重采样：

| 评测用途 | v0.4 | v0.5 | 准确率差（百分点，95%区间） |
|---|---|---|---|
| 手机购物界面状态 | 21.4% | 68.5% | +47.0 [+42.8, +51.2] |
| 手机 / 网页 GUI | 38.6% | 78.2% | +39.7 [+36.8, +42.7] |
| 手机界面状态 | 25.3% | 52.4% | +27.1 [+24.0, +30.1] |
| 第一人称场景 | 34.7% | 71.7% | +37.0 [+35.4, +38.7] |
| 图表 ChartQA | 30.7% | 58.9% | +28.2 [+25.7, +30.9] |
| CV-Bench | 55.1% | 60.8% | +5.7 [+4.0, +7.3] |
| POPE | 86.5% | 86.1% | −0.5 [−1.1, +0.1] |

其余24个旧能力集合未检出显著准确率退步，不代表已证明等效。计数与ScienceQA未显著改善。负责人于2026-10-30接受校准取舍发布：NaturalBench NLL显著增加0.029，部分集合ECE也升高。完整数据及采用说明见 [v0.5验收表](release/v0.5/accept_vs_v0.4.md) 和 [发布取舍](release/v0.5/adoption.md)。这些集合已参与多轮实验决策，不是新的独立测试。

以上五个用途是项目评测协议，不等同于ImageJevBench的五个官方领域；本版本不宣称已完成正式ImageJevBench上榜或小于1B模型领先目标。历史v0.4证据保留在 [v0.4采用说明](release/v0.4/adoption.md)。

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

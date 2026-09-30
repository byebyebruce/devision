# devision

视觉决策模型：图片 + 英文问题 → Jev 格式的结构化答案（带校准概率）。SigLIP2 视觉编码 + Laya 初始化的 ModernBERT-large 决策头。

- 设计与范围：`docs/spec/vision-decision-mvp.md`（改设计先改 spec）
- 调研：`docs/research/jev-api.md`（Jev 协议）、`docs/research/encoder-data.md`（编码器与数据集）

## 不可违背的约束

- **API 与 Jev `/v1/systemone` 兼容**：唯一扩展是 `state` 数组可含 `{type:"image", base64|url}`。响应结构不得偏离 Jev。
- **`confidence = (n·p_max−1)/(n−1)`**（Jev 口径），不用 Laya 的熵式定义。
- 推理只跑 **CPU**；训练 1×A100。
- 当前仅：英文、单图、letterbox 到 256×256、题型 `noul`/`choice`（`score` 返回 422）。
- **非商用项目**：可用 laya-vision（CC BY-NC-SA）做 baseline；许可相关改动需先确认。

## 测试

- 只从两个 seam 测外部行为：`decide(state, questions) → answers` 和数据转换器（原始记录 → Jev 格式样本）。不测张量形状/层结构。
- HTTP 层是 `decide` 的薄封装，不单测。
- 端到端冒烟（转换几十条 → 训几十步 → decide）需要 GPU，按需跑。
- 评测集图片按 image id 从所有训练源剔除（COCO/VG 跨数据集泄漏）。

## 命令

<!-- 代码骨架完成后补充：安装、测试、数据转换、训练、评测、启动服务 -->

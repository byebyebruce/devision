# 第 16 轮起：在本地 CUDA 上从零重训，目标 v0.4（2026-10-20）

## 决定（负责人 2026-10-20）

- 不再使用旧 Mac，也不复现其上封存的结果。`TODO.md` 原有的 EXP76 V 阶段、EXP76 完整运行、v0.4-k 两份输出补测都不再执行；`runs/v04-k` 的读数组合不作为 v0.4 候选。
- 在本机（WSL2，RTX 5070 12 GB，torch 2.14+cu130，31 GB 内存）重建数据、从零训练。v0.4 的目标是**在全部评测集上整体超过已发布的 v0.3**。
- 不要求逐字节复现：数据脚本可以改路径、字体等，只要规则不变并写清楚。

## 基线

- v0.3 = Hugging Face `lukbit/devision`（本机缓存 snapshot af353dc…，`model_name: devision-v0.3`）。结构：64 个视觉 token，没有局部视图、问题条件和读数器；最后一段解冻了 SigLIP 顶部 4 层（`release/v0.3/notes.md`）。
- 比较只用本机数字：在本机重建的每个评测集上评 v0.3（`devision-eval --checkpoint lukbit/devision`），再和新模型逐题配对（`devision-compare`）。旧封存数字只作参考，不能直接相减（数据是重建的，题目与图片可能不同）。
- 已有一个参考数：v0.3 在本机小试集 `data/trial_small/test_cocoqa.jsonl`（1000 题，val2014）上是 0.865。从零的小模型 `exp-cuda-small` 是 0.653，配错图后 0.641，说明对齐数据不足时模型不看图。

## 数据（本机重建）

下载实测：huggingface.co 直连可用（hf-mirror 目前只是 308 重定向到 huggingface.co，设 `HF_ENDPOINT` 没有收益）；COCO 官网单张约 10 张/秒（16 线程）；github raw 约 3 MB/s，或经 gh-proxy.com。

| 阶段 | 内容 | 规模 | 做法 |
|---|---|---|---|
| A | COCO train2014 全部图片 | 13.5 GB | 从 HF `GAIA-URJC/COCO_2014` 下 `train2014.zip`（字节数与官方一致），解压到 `data/coco/train2014/`；captions 等脚本跳过已存在的图 |
| A | `build_all.sh`（data-v2 到 v9、lv_bench、v11 之前的全部） | 约 10 GB（GQA、POPE、A-OKVQA、ScienceQA、Cauldron 的 AI2D/TQA、VQAv2、VG 图片逐张） | `UV_NO_SYNC=1 bash scripts/data/build_all.sh data`（脚本内部用 `uv run`，不加会联网同步） |
| B | `ext/`（15 个来源）、`r13` 来源、exp14 PixMo、r13b / r14 / r14b 训练混合 | 约 130 GB | A 跑训练时在后台下载；VisMin 39 GB、CoGenT 24.7 GB 最大；Flickr / Zenodo 上的图预计有失败，按脚本已有的失败统计如实记录 |

注意：
- VG 图片不要整包解压进 `gqa/images/`（GQA 场景图题用的是磁盘上已有的 GQA 图，整包会改变输出），按 id 逐张取。
- 合成图脚本（`exp21_reading.py`、`real_layout.py`、`v10_diagrams.py`、`r13/gqa_bind.py`、`r13/sources/base.py`、`release/confirm.py`）写死了 macOS 字体路径 `/System/Library/Fonts/Supplemental/Arial.ttf`；需要时改成可配置，本机用 Windows 的 Arial（`/mnt/c/Windows/Fonts/arial.ttf`）或 DejaVu。
- 小试数据已移到 `data/trial_small/`，避免被当作评测集参与剔除扫描。

## 训练路线

v0.3 的链路是 10 段、在 MPS 上约 43 小时（align → v3 → v5 → v6 → v7 → v8 → v9b → v11 → v12-ext → r15-vision）。本机按两步走：

1. **r16（数据 A 就绪即可开始）**：`configs/scratch.yaml` 的结构（align 2 轮 + data-v2 混合 + data-v3 续训），再接 v6 翻转、v7 SQA、v8 VSR/V7W、v9b 真实左右、v11 计数恢复，各段沿用原来的数据与学习率（5e-5 / 5e-5 / 1e-4，预热 300–500）。目的是在 CUDA 上先得到一个与 v0.3 前半段同级的模型，并核对速度、显存与精度设置。
2. **r17（数据 B 就绪后）**：v12-ext 扩展包（15 个来源，一遍）→ r14b 混合 + 解冻 SigLIP 顶部 4 层（2e-5 / 1e-5 / 5e-5，`lr_vision` 5e-6）。这是 v0.3 收益最大的两段。在此基础上加已知的改进：
   - 存在题 yes = no 配平（r13 只有 15% 是 yes，POPE −6.5）；
   - ScienceQA 回放只用必须看图的题；
   - 计数题避免“选项围绕正确答案”的格式，并加选项倒序增强（CountBench 倒序翻转 39.5%）；
   - 温度类 ScienceQA 图的确定性 K 规则（K = 质量 × 速度²）可后加，作为单独的实验。

不采纳：v0.4-k 的读数组合（局部视图 12 + 问题条件 + EXP71 读数器 + exp30 到 protect-k 的长链）：成本高，带来 position −2.0、size −1.6、vqa_choice −1.5，新图上读数 1/240，dev16 从未达标。EXP26 面板绑定对齐是唯一学会 dev16 的分支，留作 r17 之后的候选实验。

## 设备与精度

- `rlcd.py` 的 `amp` 默认 `"cuda"`：在 CUDA 上前向用 bf16 autocast。v0.3 链路里 v11 之前的段在 MPS 上实际是 fp32，之后显式 `amp: off`。r16 第一段先用短程对照（同数据、同 seed，bf16 与 off 各约 2,000 步，比较 val 曲线与速度、显存）再定，结论写进训练日志。
- 显存：12 GB；micro_batch 8 在冻结视觉时预计足够，解冻 SigLIP 顶层后在 r17 开训前实测，必要时改 micro_batch 并单独记录。
- 延迟：发布前用 `--device cpu` 单独测（CLAUDE.md 约束）。

## v0.4 验收

- 评测集：`test_*`（data-v2、v5、v6 翻转、v11 计数）、`bench_pope`、四个 `bench_lv_*`（含对照），r13 之后的 `test_bind_gqa`、`bench_countbench`、`bench_whatsup` 等在数据 B 就绪后加入。
- 判据：每个集合上新模型 − v0.3 的配对差值（`devision-compare`，按图片重采样 95% 区间）。
  - 不允许任何集合出现上界 < 0 的退步；
  - 多数集合点估计为正，且 v0.3 的已知弱项（ScienceQA 看图题、计数、VSR、位置翻转）至少两项下界 > 0。
- 每轮按 CLAUDE.md：训练日志一节、与 laya-vision 比较（`scripts/lv_report.py`）。

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# deVision (Decision + Vision)

视觉决策模型：图片 + 英文问题 → Jev 格式的结构化答案（带校准概率）。SigLIP2 视觉编码 + Laya 初始化的 ModernBERT-large 决策头。

- 设计与范围：`docs/spec/vision-decision-mvp.md`（改设计先改 spec）
- 调研：`docs/research/jev-api.md`（Jev 协议）、`docs/research/encoder-data.md`（编码器与数据集）、`docs/research/data-quality.md`（训练数据质量与 v2）、`docs/research/laya-vision-gap.md`（与 laya-vision 的差距）
- 实验过程与结论：`docs/experiments/2026-09-30-rlcd-plateau.md`（按时间追加；提新实验前先看，避免重复已失败的做法）
- 训练日志：`docs/training-log.md`（每一轮一节：改了什么、结果、相对上一轮的增长；顶部标明当前最佳。每轮训练评测完后更新）

## 代码结构

- `src/devision/model/`：网络、预处理、checkpoint、`Decider.decide`；不依赖其他子包。数据流：图片 letterbox 256 → 冻结的 SigLIP2 → 2×2 pixel shuffle + MLP 投影层 → 64 个视觉 token 插在 `[CLS]` 之后 → ModernBERT-large（Laya 权重）→ Laya 决策头在每个选项的 `[MASK]` 上打分 → 按题型温度 softmax。训练和推理共用 `question_item` / `image_tensor`，序列构造不要另写一份。
- `src/devision/train/`：样本格式（`samples.py`）、阶段 1 对齐（`align.py`，带图完形填空，训投影层，`--lora-r` 时 ModernBERT 也加 LoRA）、阶段 2 RLCD 训练（`rlcd.py`）、评测（`evaluate.py`，走 `decide`）、SwanLab 上报（`tracking.py`）、YAML 实验编排（`pipeline.py`），依赖 model。
- 准备训练数据的代码（下载、转换、转换器测试）**不进仓库**：放在 `scripts/data/`（已 gitignore）。仓库只认 `samples.py` 里约定的 JSONL 格式。
- `src/devision/serve/`：`/v1/systemone` API，依赖 model。
- `src/devision/demo/`：web demo 页面和示例图路由，只通过 HTTP 调 API，不 import model、train、serve。
- 依赖只能单向，不要让 model 反向引用 train/serve/demo，也不要让 train、serve、demo 互相引用。
- 测试按包放：`tests/model/`、`tests/train/`；共享的 tiny 模型在 `tests/conftest.py`。

## 不可违背的约束

- **API 与 Jev `/v1/systemone` 兼容**：唯一扩展是 `state` 数组可含 `{type:"image", base64|url}`。响应结构不得偏离 Jev。
- **`confidence = (n·p_max−1)/(n−1)`**（Jev 口径），不用 Laya 的熵式定义。
- 推理只跑 **CPU**；训练在本地 Mac（MPS）。
- 当前仅：英文、单图、letterbox 到 256×256、题型 `noul`/`choice`（`score` 返回 422）。
- **非商用项目**：可用 laya-vision（CC BY-NC-SA）做 baseline；许可相关改动需先确认。

## 测试

- 只从 seam 测外部行为：仓库内是 `decide(state, questions) → answers`；数据转换器的测试在 `scripts/data/`（`uv run pytest scripts/data`）。不测张量形状/层结构。
- HTTP 层是 `decide` 的薄封装，不单测。
- 端到端冒烟（`tests/train/test_pipeline.py`）：tiny 模型 + 合成红/蓝图样本 → 训练 → 存盘 → 加载 → 经 decide 评测，CPU 上几秒，随 `pytest` 一起跑。真实模型的同一流程用下面的 train → eval 命令（训练在 Mac 上，`--device mps`）。
- 评测集图片按 image id 从所有训练源剔除；VG（GQA）里约一半是 COCO 图，剔除时两套 id 都要对上（`convert.same_images`，映射来自 VG 的 `image_data.json`）。
- v2 数据的图片用途互不重叠：训练文件、`dev_*` / `dev_mix`（训练中监测、选 checkpoint、拟合温度，按图片哈希约 3% 加稀疏切片补充）、`test_*` / `bench_pope`（最终测试：COCO val2014、VQAv2 val、GQA val、POPE，同时避开 v1 和 v2 的训练图片；不要放进训练中的周期性 `--eval`）。
- 评测报告：每个集合标明用途（`--role`）。YAML 按集合与被评模型调过的数据之间**实际共享的题目和图片**判定（不看文件名）：≥ 一半在温度拟合集里为 `calibration_fit`，与拟合集或训练中监测过的集合（各阶段 `--val` / `--eval`、只评测配置的 `history:`）有任何重叠为 `monitoring`，否则 `heldout`；`image_identity:` 让同一张图的 COCO / VG id 对上；也可写 `{path, role}` 直接指定。同一个文件对不同模型的用途可以不同（如 `bench_pope` 对 v1 模型是 monitoring，对 v2 模型是 heldout）；`devision-eval` 输出逐题明细，汇总（准确率按题型 / 来源 / kind / 轴 / 选项数、NLL、Brier、ECE、可靠性分箱、阈值覆盖率、POPE 的 precision / recall / F1 / yes ratio）都能从明细重算；同时给出 MANIFEST 里的「只看题目」基线、配错图和选项倒序两个对照（YAML `evaluate.controls: true`）。模型之间的差值用 `devision-compare`，不要用固定的「几个点以内不显著」。
- 评测集命名：`test_<名字>` 是我们自己数据里留出的测试划分（`data/v2/test_{exist,position,relation,size,gqa,vqa_yesno,vqa_choice}.jsonl`，与 LookFirst 的 config 名一致），`bench_<名字>` 是外部公开评测（`bench_pope` = `data/v2/bench_pope.jsonl`；laya-vision 的四个集合 `bench_lv_{vqav2_yesno,aokvqa,scienceqa,pope}`，文件仍在 `data/lv_bench/`）。YAML 里的集合名就是结果文件名（`runs/<轮次>/eval/<集合>.json`、`.details.jsonl`、`.mismatched.*`、`.reversed.*`）。旧名（eval_ 开头、第 2 轮的 v2eval 前缀、lv 开头）由 `scripts/rename_eval_sets.py` 一次改完。
- 早期评测集：`val_mix`（GQA testdev 1000 + VQAv2 val 1000，用于 `--val` 与拟合温度，指标按来源拆分）、`pope`（300，只评不拟合）。旧的 `val.jsonl`（200 题）只为和早期实验对比而保留。

## Python 与依赖

- 用 **uv** 管理 Python 与依赖（Python 版本见 `.python-version`）。加依赖用 `uv add <pkg>`（开发依赖 `uv add --dev`），不要用 pip 或手改 `uv.lock`。
- 所有命令经 `uv run ...` 执行。
- 依赖分层（参照 Laya）：核心依赖只够推理；`serve`（fastapi、uvicorn）和 `train`（peft、swanlab）是 extras，`model` 包不能 import 它们。开发依赖组也包含这些 extras，所以 `uv run` 下全部可用；新增服务或训练依赖时，`uv add --optional <extra>` 之外再 `uv add --dev` 一次。
- 对外入口：`devision.load(...)` → `Decider.predict` / `decide`；`import devision` 不加载 torch。

## 命令

```bash
uv run pytest -q                      # 全部测试（含 tiny 模型端到端冒烟，CPU 上几秒）
uv run pyright                        # 类型检查
uv run pytest tests/train/test_pipeline.py -k aligned -q   # 只跑一个测试（-k 按名字筛）
uv run pytest scripts/data -q          # 仓库外数据生成脚本的测试
uv run python scripts/data/prepare.py fetch --root data ...   # 仓库外：生成 data/{train,val,pope}.jsonl
uv run python scripts/data/captions.py --root data --exclude data/val.jsonl data/pope.jsonl   # 仓库外：生成 data/align_{train,val}.jsonl（COCO caption）
uv run python scripts/data/evalsets.py --root data --gqa 1000 --vqav2 1000   # 仓库外：生成 data/val_{gqa,vqav2,mix}.jsonl，并从 train.jsonl 剔除评测图
uv run python scripts/data/bigtrain.py --root data --gqa 60000 --vqav2 40000 --out data/train_100k.jsonl   # 仓库外：更大的阶段 2 训练集
uv run python scripts/data/cocoqa.py --root data --split train2014 --limit 60000 --out data/cocoqa_train.jsonl   # 仓库外：从 COCO 实例框出题（val2014 + --out data/val_cocoqa.jsonl 是评测集）
uv run python scripts/data/v2_build.py --root data   # 仓库外：一条命令重建 v2 数据集（data/v2/：训练、dev_*、dev_mix、test_*、bench_pope、MANIFEST.json），任何验收失败即中止；见 docs/research/data-quality.md
uv run python scripts/data/v2_mix.py --root data     # 仓库外：按能力配额混合成 data/v2/train_mix.jsonl（左右类 ≤5%，每图 ≤6 题，混合后重新配平验收）
uv run python scripts/data/lv_bench.py --root data   # 仓库外：按 laya-vision 公开的逐题预测重建他们的评测集到 data/lv_bench/（VQAv2 是非 / A-OKVQA / ScienceQA / POPE，含我们训练没见过的 *.unseen 子集），并把他们的逐题预测转成我们的明细格式；不安装、不运行他们的模型。用 configs/*-lvbench.yaml 评测我们的模型（集合名 bench_lv_<集合>），再用 devision-compare [--only *.unseen.jsonl] 逐题配对比较
uv run devision-align --data data/align_train.jsonl --val data/align_val.jsonl --data-root data --out runs/align --run-name align   # 阶段 1
uv run devision-train --init runs/align --lr-new 5e-5 --lr-head 5e-5 --lr-lora 1e-4 --warmup 500 --data data/train_100k.jsonl --val data/val_mix.jsonl --data-root data --eval pope=data/pope.jsonl --out runs/x --run-name x   # 阶段 2；Mac 上加 --device mps
uv run devision-eval --checkpoint runs/x --data data/pope.jsonl --data-root data --out runs/x/pope.json --role heldout   # 另写 pope.details.jsonl（逐题明细）；--control mismatched|reversed 为对照
uv run devision-compare runs/a/x.details.jsonl runs/b/x.details.jsonl --by source   # 同一批题上两个模型的差值，按图片配对重采样给 95% 区间
uv run devision-serve --checkpoint runs/x --port 8000   # POST /v1/systemone；浏览器打开 / 是 web demo（--no-demo 关闭）
uv run devision-pipeline configs/v3-x.yaml [--dry-run] [--from STAGE] [--force]   # 按 YAML 跑一轮训练（再跑同一个 YAML 是续跑）
```

- **实验用 YAML 编排**（`configs/*.yaml`，`src/devision/train/pipeline.py`）：每个新实验复制一份 YAML 改参数和 `name`，不要再写训练脚本。一个 YAML = `name` + 若干 `stages`（`kind: align|train`、`init` 指向前面的阶段或路径、`data` / `val` / `eval`、`params` 即训练 CLI 的参数名，`common` 是所有阶段共用的参数）+ `evaluate`（评测集和额外命令，`{checkpoint}` / `{name}` / `{eval_dir}` 会被替换）。启动前检查所有参数名、类型和数据文件。长任务用 `nohup uv run devision-pipeline configs/v3-x.yaml > runs/v3-x.out 2>&1 &`。
- **训练轮次 = YAML 的 `name`**，格式 `v<轮次>-<描述>`（如 `v3-data2`），文件名与之相同（`configs/v3-data2.yaml`）；轮次号由人来管理，不自动生成。输出在 `runs/<name>/`：`run.json`（配置、开始时间、git commit、各阶段起点）、每个阶段一个子目录（checkpoint + 报告 + `swanlab_run.json`）、`eval/`（评测结果和逐题明细）、`logs/`；SwanLab run 名 `<name>/<阶段>`。再跑同一个 YAML 就是续跑：已完成的阶段和评测跳过，`--from STAGE` 从某阶段重跑；阶段 2 每 `--save-every` 步（默认 1000）把训练状态写到阶段目录的 `resume.pt`，中断后再跑会从那里接着训（设置不同会拒绝，`--force` 会删掉它从头来），训完自动删除。同一 seed 在 CPU 上逐位可复现；MPS 上有 1e-7 量级的浮点差异，不保证逐位一致。新的一轮 = 新 YAML + 新 name。只评测的 YAML（`stages: []`）不算一轮，结果写进被评 checkpoint 所在轮次的 `eval/`。轮次（v1、v2、v3……）和数据版本（data-v1、data-v2）是两回事。第 1 轮 `v1-release-0.1`（发布的 0.1），第 2 轮 `v2-align-lora`；早期和失败的运行在 `runs/archive/`。

- `data/`（数据集）、`runs/`（checkpoint）不进 git；`examples/` 里的少量示例图进 git，供 web demo 默认加载；`examples/train/` 是格式示例数据（含合成形状图），配 `configs/example.yaml` 端到端跑通，改样本格式时要同步更新它。
- 训练默认上报 SwanLab（项目 `devision`），两个阶段同一套分组（`src/devision/train/tracking.py`），每 `--log-every` 步一行：`train/`（窗口内均值：loss、accuracy、grad_norm；阶段 2 另有 nll、sigma）、`lr/`（各参数组）、`perf/`（step_s、samples_per_s、progress_pct、eta_h）、`sys/`（cpu_pct、proc_cpu_pct、ram_used_pct、ram_available_gb、swap_used_gb、proc_rss_gb、gpu_util_pct、gpu_mem_gb、torch_mps_gb）。评测行另记：阶段 1 每 `--eval-every` 步记 val 上真实/错配图片的补词准确率与 nll 及两者之差（`val/mlm_acc_gap`）；阶段 2 从第 0 步起每 `--eval-every` 步和每个 epoch 末，在 val 和每个 `--eval` 集上记准确率（总体/noul/choice，混合来源时按来源拆分）、nll、ECE，结束时记拟合温度和套用温度后的指标（`final/...`）。只有训练（`devision-align` / `devision-train`）新建 SwanLab run；`devision-eval`、分析脚本和测试都不上报，结果写成 JSON / 文本文件。例外：`devision-pipeline` 评测完后把汇总（每个集合的准确率、NLL、ECE、配错图准确率、倒序翻转率）续写进训练这个 checkpoint 的那个 run（按阶段目录里的 `swanlab_run.json` 找到），按用途和集合名分组：heldout 的 `bench_*` 集合记在 `bench/<集合>/...`、其他 heldout 集合记在 `test/<集合>/...`，calibration_fit / monitoring 的（不论名字，如对第 2 轮的 `bench_pope`）记在 `ref/<集合>/...`（`pipeline.summary_group`）；不新建 run，逐题明细只在文件里。先 `uv run swanlab login` 登录（API key 只存在本机用户目录，**不要写进仓库**）；`--swanlab-project ""` 关闭。本地缓存 `swanlog/` 不进 git。
- 首次训练会从 Hub 下载 `convaiinnovations/laya` 与 `google/siglip2-base-patch16-256`；对齐阶段还会下载 `answerdotai/ModernBERT-large`（只取 MLM 头）。
- 不做阶段 1 直接训决策题，nll 会停在 ln2（见 `docs/experiments/2026-09-30-rlcd-plateau.md`）。
- 阶段 2 用默认学习率长训练（上万步）会塌缩成 50/50 输出；用上面命令里的低学习率加预热。发布的 checkpoint（`runs/v1-release-0.1/stage2b`）怎么训出来的见 `configs/v1-release-0.1.yaml`。
- `--checkpoint` / `--init` / `Decider.load` 都接受本地目录或 Hugging Face 模型仓库 id。

## 评论（`critic/`）

`critic/critic.md` 是另一位评审（按 `AGENTS.md` 工作，只读审查）对最近改动写的评论；`critic/` 已 gitignore。用户说「看评论」时：

1. 先读 `critic/critic.md`，只处理尚未回复的内容（文件末尾最后一条回复之后的部分）。
2. **甄别，不要被评论带着走。** 每一条分开判断三件事，不能因为事实属实就照单全收：
   - **事实对不对**：用只读检查、复算或查代码给出证据，不凭印象。
   - **影响多大**：会不会改变实验结论、训练信号或对外说法？只影响措辞或边缘情况的，就按小问题处理。
   - **现在值不值得**：成本、复杂度，以及是否挤占主线（模型训练和对照结果）。能用最小改动纠正错误（如改一句文档、显式写一个标签）时，不新造机制。
3. 据此给出决定：**现在做 / 推迟 / 不采纳**，部分同意是正当的结论。评论把 "记录过" 和 "用来做过决定" 混为一谈、把建议升级成阻断条件、或忽略了已知限制（如某数据集自带的标签噪声）时，要明确指出。
4. 现在做的：照常写测试、在本地提交（不自动 push），回复里写明改了什么、验证结果。推迟或不采纳的：写理由和证据。
5. 被评审引用的数据或 checkpoint 版本（"观察身份"）保留到这一轮评审结束；回复里的证据引用文件、提交或 MANIFEST，不引用对话内容；没跑过的检查不写成已验证。
6. 回复**追加**到 `critic/critic.md` 末尾，不改动原评论。格式：

```
====================================================================
回复 · <YYYY-MM-DD HH:MM> · 作者：Claude Code
====================================================================
1. <原评论要点>
   事实：正确 / 部分正确 / 不正确 —— 证据：...
   影响：<改变什么结论 / 只影响说法 / 很小>
   决定：现在做（commit / 文件，验证结果）/ 推迟（原因、何时做）/ 不采纳（理由）
2. ...
====================================================================
```

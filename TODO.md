# TODO

待办事项，按优先级排列。每项写明：为什么做、做什么、怎么算完成，以及出处（评审编号或实验记录）。完成的项移到文末「已完成」，附 commit。最后更新：2026-10-02。

优先级：**P0** 今晚 / 下一步必须做；**P1** 下一次长训练之前做；**P2** 有空或有证据后做；**P3** 研究性、可选。

---

## P0 — 正在进行 / 紧接着做

### 1. ~~修 MC1：CLI 种子覆盖新建模型的随机投影层~~（已完成，见文末）
- **为什么**：`train/cli.py` 先 `_decider(a)` 建模型再进 `align()` / `train()` 设种子；本机 PyTorch 每个进程的默认种子不同，所以从 Laya 新建时投影层初始值每次都不一样。已完成的 `align-full-mac` 与 `align-lora` 对比因此混入了不同的初始投影层（差距很大，大概率不是它造成的，但没量化）。
- **做什么**：CLI 在建模型之前用 `--seed` 设好 `random` / `torch` 种子；训练内部的播种保留。
- **完成标准**：同一 seed 两次新建，投影层参数完全一致；不同 seed 不同；从 checkpoint 加载不受影响。有 tiny 测试。
- 出处：critic MC1（2026-10-02 20:43）。

### 2. ~~修 MC4-1：选项倒序对照不再对是非题空跑~~（已完成，见文末）
- **为什么**：倒序只改 choice，noul 输入不变；8 个 `eval_*` 共 17,532 题中 11,790 道是 noul（POPE 全是），今晚的对照评测约 1.2 万次推理是白跑（CPU 约 40 分钟）。
- **做什么**：流水线只对含 choice 的集合生成倒序步骤；倒序评测只推理 choice 题，记录里注明 noul 未参与。
- **完成标准**：纯 noul 集合不出现 `reversed:` 步骤；混合集合的倒序明细只含 choice；`order_sensitivity` 结果不变。有测试。
- 出处：critic MC4 第 1 点。

### 3. ~~今晚 v1 流水线结束后（约 00:15）的评测~~（已完成 2026-10-03，结果见实验记录末节和 `docs/research/laya-vision-gap.md`；README Benchmark 未改，待确认）
- 后续：发布的 `stage2-cocoqa` 没有逐题明细，和新一轮只能比点估计；需要配对区间时，用同一批集合补评它（只评测，不训练）。
- `configs/eval-v1-align-lora.yaml`：v1 数据模型（`runs/align-lora-cocoqa`）在 v2 测试集上，带配错图 / 倒序对照，用途按实际重叠自动标注（`eval_pope` 对这个模型是 monitoring）。
- `configs/lv-bench-align-lora.yaml`：同一模型在 laya-vision 公开评测集上（VQAv2 是非 4,887 / A-OKVQA 1,138 / ScienceQA 2,097 / POPE 9,000）。
- 然后 `devision-compare data/lv_bench/laya-vision-201m.<set>.details.jsonl runs/align-lora-cocoqa/lv_<set>.details.jsonl`，全部题和 `--only data/lv_bench/<set>.unseen.jsonl` 各一次；POPE 他们只公开总分，只比总分。
- 结果写进 `docs/experiments/2026-09-30-rlcd-plateau.md` 和 `docs/research/laya-vision-gap.md`；README 的 Benchmark 等确认后再改。
- **注意**：ScienceQA（教科书插图）和 A-OKVQA（常识推理）我们没训练过，低分是预期，不是评测错误。

### 3a. 今晚评测汇总之后（不在评测进行中改代码）
- **训练按轮编号**：`runs/v<N>-<描述>/`，各阶段为子目录，SwanLab run 名同轮名；旧的两轮整理成 `v1-*`（0.1，`stage2-cocoqa`）和 `v2-*`（align-lora），下一轮从 `v3-` 开始。失败实验进 `runs/archive/`（只留报告、评测、日志；删权重前列清单给用户确认）。同步改文档和 YAML 里的路径。轮次编号与数据版本 `data-v1` / `data-v2` 无关。
- **评测进度**：`devision-eval` 每几百题打印一行进度和预计剩余时间。
- **评测提速**：即第 9 项（同图多题合并成一次请求），提前做。
- **评测汇总写回该轮训练的 SwanLab run**（`test/<集合>/accuracy|nll|ece`、对照结果），不新建 run；先确认 SwanLab 能续写已结束的 run。逐题明细仍只在文件里。

### 4. 回复 critic MC1–MC6
- 按 CLAUDE.md 的格式追加到 `critic/critic.md`：事实 / 影响 / 决定（现在做 MC1、MC4-1；推迟 MC2、MC3、MC4-2、MC5、RLCD 消融；暂不采纳 MC6）。

---

## P1 — 下一次长训练（v2，约 6 小时）之前

### 5. MC2 简化版：长训练可中断续训
- **为什么**：每个阶段 4–5 小时，只在结束时保存推理权重（LoRA 已合并），没有 optimizer / scheduler / 步数 / RNG；中断要整段重跑，也取不回中间权重。
- **做什么（简化版）**：每 N 步滚动保存一个续训 checkpoint：可训练参数（未合并的 LoRA、投影层、决策头）、optimizer、scheduler、global step、epoch 内位置、`random` / `torch`（含 MPS）RNG；临时文件写完再替换，只留最新 1–2 个。`devision-train --resume <dir>` 接着训。最终推理导出仍在结束时做。
- **完成标准**：tiny CPU 上连续训 N 步 vs 训 K 步中断再续到 N 步，样本顺序、学习率、sigma 一致，最终 loss / 权重在容差内一致；导出后 `Decider.load/decide` 正常。MPS 上不承诺逐位一致。
- 出处：critic MC2。

### 6. 跑 v2 对照实验
- `configs/v2-align-lora.yaml`：同一 stage 1（`runs/align-lora`）+ v2 混合数据（`data/v2/train_mix.jsonl`，134,556 题，左右类 4.07%，每图 ≤ 6 题），`dev_mix` 监测 / 选模型 / 拟温度，结束后在全部 `eval_*` 上评测，开对照。
- 和第 3 项的 v1 结果用 `devision-compare` 逐题配对比较（同一套 eval 题）。
- 前置：第 1 项（种子）、第 5 项（续训）。

### 7. 空间题占比对照（5% vs 15%）
- 来自数据计划：左右类题当前学不会，占比用受控对照决定，不预先定死。
- 做法：`scripts/data/v2_mix.py --lr-share 0.15` 生成另一份混合，其他不变，各训一次，看空间题与其他能力的变化。
- 前置：第 6 项的基线。

---

## P2 — 有空或有证据后

### 8. MC3：对齐时补词输出层只算被遮住的位置
- 现在 `align.py::_mlm_logits` 对全部 caption 位置算 5 万词表的 logits，再挑出被遮位置。抽样估计约七成输出位置可省；按我估算只占整步计算的百分之几，外加省约 100 MB 激活。
- 下次改对齐代码时做；要求固定输入下有效位置的 logits、loss、梯度和改前一致。

### 9. MC4-2：评测时同一张图的多道题合并成一次请求
- `decide` 已支持一图多问、视觉只编码一次；POPE 每图 18 题。需要给题目临时分配唯一键再还原 `sample_id/qid`，保留配错图和换序对照。
- 完成标准：分组与逐题结果在容差内一致；分别记录吞吐与单题 P50/P95，不能用吞吐代替单题延迟。

### 10. MC5：冻结视觉特征的复用
- SigLIP 冻结且预处理确定，可复用 patch 输出；但全量缓存 25–50 GB，这台 24 GB 的 Mac 放不下。
- 先测视觉塔占训练 / 评测总耗时的比例；占比够大再做 batch 内去重、dev 集缓存或有上限的磁盘缓存。缓存键要含图像内容、预处理、视觉权重版本和 dtype。

### 11. 独立的温度拟合集
- 现在 v2 用 `dev_mix` 同时选模型和拟温度（已披露）。若开始反复按 ECE / 阈值调参，再按图片身份单独留一份 calib。
- 「阈值在 dev 上定、在测试上验证」的流程工具已有（评测输出的阈值表），尚未在真实模型上执行。

### 12. 评测覆盖补充
- 人工核查的小型诊断集：物体在 256 像素下是否看得见、困难负例、按文字找目标（成对题：同图换问的目标，答案应随之变化）。
- 多问题请求、带选项说明（criteria 非 null）的选择题的质量和延迟。
- dev 中相对位置上下题只有 42 道（配平后），按需补。
- 近重复图片检查（现在只按 COCO / VG 编号和 A-OKVQA 的感知哈希查过）。

### 13. 延迟测量规范
- 固定请求、预热、线程数、单题 vs 多题请求分开记录 P50 / P95；在目标服务器上测，分出预处理、视觉塔、文本编码器、决策头各自的占比。README 里 160–190 ms 是历史 Mac 数字。

---

## P3 — 研究性 / 可选

### 14. RLCD vs 只用交叉熵的消融
- `rlcd.py` 里策略梯度项和 CE 直接相加，没有受控对照。同起点、同数据、同预算比一次，先用 dev 和几百步筛查。v2 基线之后再做。

### 15. 文字到区域的绑定（左右关系）
- 已测失败的条件见实验记录（只训投影层、LoRA r16/r64、全量放开、4 倍合成数据、带方位的完形填空）。不原样重试。
- 未试过的方向：决策头加选项标记对视觉 token 的交叉注意力（需先改 spec）；更大规模的绑定数据配合对齐阶段 LoRA。

### 16. 数据扩充
- The Cauldron 的闭式题转换（约 27 万，含 VSR、Visual7W、TallyQA），参考 laya-vision 的 `cauldron.py`；注意许可（部分子集非商用）。
- 多选题颜色 + 计数占比高（训练池 78%），混合时已限额；其余类别（肉、蔬菜、天气等）样本少。

### 17. 其他性能候选（无实测证据前不排前）
- MPS 混合精度（bf16）、长度分桶、减少逐步 `.item()` 同步、`inference_mode`、权重存 bf16（2.07 GB → 约 1.04 GB，需重测指标）、量化 / 编译。不要同时改 batch / 学习率 / 精度。
- MC6：加载 checkpoint 时跳过无用的随机初始化；一次评测共用一个 `Decider`。影响小，暂不做。

---

## 等用户决定

- **推送模型到 Hugging Face**：需要 `uv run hf auth login`、仓库名、公开 / 私有、权重许可（建议 CC BY-NC 4.0）。GitHub 仓库目前私有，模型卡里的代码链接对外打不开。
- **推送代码到 GitHub**：本地有未推送的提交，按约定只在要求时推。
- **SwanLab 上早先补测旧 checkpoint 时留下的 `eval-*` 记录**是否清理（删除不可逆）。

---

## 已完成（近期）

- MC1：CLI 在建模型前设种子（`devision.train.cli.seed_all`），测试 `tests/train/test_seeding.py` —— commit "Seed before building a fresh model; reversed-option control ..."
- MC4-1：倒序对照只对含 choice 的集合运行、只推理 choice 题（v1 / v2 评测配置由 27 / 28 步减到 24 / 25 步）

- 评测逐题明细、对照（配错图 / 选项倒序）、`devision-compare`、集合用途按实际重叠标注 —— `8146eaa`、`599f3f6`、`2bc1329`、`4fd5009`
- v2 数据集（按类修复、配平、图片身份统一、dev/eval 划分、MANIFEST、训练混合）—— 数据脚本在仓库外，文档 `docs/research/data-quality.md`
- laya-vision 公开评测集重建与逐题对齐 —— `0ff54ea`，数据 `data/lv_bench/`
- 对齐阶段加 LoRA（阶段 2a 上 VQAv2 0.634 → 0.723）—— `e3529cf`，结果见实验记录

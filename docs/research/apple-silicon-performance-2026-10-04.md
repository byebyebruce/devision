# Apple Silicon 训练与评估提速调研

调研日期：2026-10-04（Asia/Shanghai）。本节为联网核对的官方资料；未执行模型、训练、评估或性能测试，未修改实现或运行配置。源码固定到 PyTorch `v2.14.0`，避免用早期 MPS 限制判断当前版本。下列优先级是待结合本项目耗时分解验证的建议，不是已测得的加速收益。

## 官方资料与适用边界

### 1. 优先测 MPS 混合精度，不能沿用“MPS 不支持 AMP”的旧判断

PyTorch 2.14 的 autocast 接受 FP16/BF16；MPS 注册了独立的算子策略：`linear`、`matmul`、`bmm`、SDPA 等使用低精度，`layer_norm`、`softmax`、`log`、NLL 等保持 FP32。这与把整个模型直接 `.half()` 不同。前向及 loss 可放入 autocast，反向应在其外进行。[autocast 前端源码](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/amp/autocast_mode.py)、[MPS 算子精度策略](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/autocast_mode.cpp)。

**建议，较高优先级：** 先比较 MPS FP32 与 BF16 autocast，再按需要比较 FP16；保留自定义概率、奖励和敏感归约的 FP32 计算。验收同时看完整 step 的耗时、内存、有限梯度和固定样本概率差异，不能仅凭前向变快就切换训练。BF16 与 FP16 的吞吐差异需实测，不能从类型名称推断。FP16 训练还要核对 gradient scaling；2.14 的 GradScaler 已有 MPS 分支，旧版“必然不可用”的结论也不适用，但本项目完整链路未经运行验证。[GradScaler 源码](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/amp/grad_scaler.py)。

### 2. SDPA 在 MPS 有专门实现，但训练与评估走法不同

2.14 的 SDPA dispatcher 仅在 MPS 且无需对 Q/K/V 记录反向时进入专用 MPS 路径；需要梯度的计算会进入通用 math 路径。因此，评估或冻结视觉编码器的加速不能直接外推到训练中的文本骨干。[SDPA dispatcher](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/transformers/attention.cpp)。

专用实现按序列长度、head dimension、dtype、mask 类型与布局选择 kernel；长序列 prefill 支持合适的布尔 mask 或与 Q 同 dtype 的浮点 mask，不能笼统说“有 mask 就不能走快路”。专用路径要求 `dropout_p == 0`。**建议，中优先级：** 对真实模型检查实际路径，保持 padding、局部注意力和因果语义；不能为提速删掉必需 mask 或改训练 dropout。[MPS Attention 实现](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/operations/Attention.mm)。

### 3. 减少细碎的 CPU/GPU 同步

MPS 的 `.item()` 底层标量读取先把 tensor 同步拷到 CPU；MPS→CPU 的阻塞拷贝也会等待 GPU。统一内存不代表这些等待消失。**建议，较高优先级：** 把每个样本、每个参数或每个 microbatch 的标量读取集中到确实需要日志的时点；指标先在设备上归约，再批量取回。只移动诊断计算，不改变 loss、奖励或优化器更新语义。[Scalar 实现](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/operations/Scalar.mm)、[Copy 实现](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/operations/Copy.mm)。

### 4. 先用 Metal 时间线确定瓶颈

Apple 推荐用 MPS profiler 的 OS signposts 配合 Instruments / Metal System Trace，区分 GPU 算子、CPU/GPU 拷贝、unsupported op 的 CPU fallback。它能回答是在算、在等数据，还是频繁提交小任务。[Apple WWDC23：Optimize machine learning for Metal apps](https://developer.apple.com/videos/play/wwdc2023/10050/)。

`torch.mps.profiler.profile()` 默认 `wait_until_completed=False`；设为 True 会让每个 GPU 操作等待结束，官方明确它会降低性能。端到端计时需在测量区间两端同步，先预热，再测一段迭代；不能用异步提交时间冒充 GPU 完成时间，也不应把逐算子强制同步留在日常训练中。[MPS profiler API](https://docs.pytorch.org/docs/2.14/generated/torch.mps.profiler.profile.html)、[MPS synchronize API](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html)。

### 5. 环境变量适合最后做单变量对照

以下含义来自 [PyTorch 2.14 MPS 环境变量文档](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)：

| 设置 | 官方含义 | 本次建议 |
| --- | --- | --- |
| `PYTORCH_MPS_PREFER_METAL=1` | matmul 选择 Metal kernel，替代 MPS Graph | 分独立进程对比默认值；不保证更快 |
| `PYTORCH_MPS_FAST_MATH=1` | MPS kernel 启用 fast math | 同时比较概率/NLL与稳定性，再决定是否保留 |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | 不支持的 MPS 算子回退 CPU | 兼容机制；先查是否真实发生回退 |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | 分配硬上限；0 关闭保护 | 不作为通用提速开关，不推荐关闭保护 |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | 回收与 adaptive commit 的软阈值 | 仅在内存/提交时间线支持时研究 |

这些开关没有证明本项目会加速。尤其“强制 Metal”“开放全部内存”不等于提高吞吐；应先解决重复计算、批量大小与同步问题。

### 6. `torch.compile` 已有 MPS backend，适合有限试验

2.14 的 Inductor 已注册 `MetalScheduling`，并有 Metal 代码生成器。因此不能再给出“MPS 完全不支持 torch.compile”的笼统结论。源码存在不代表本项目整图可编译，更不证明编译后的完整训练更快。[backend 注册](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/codegen/common.py)、[Metal codegen](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/codegen/mps.py)。

**建议，较低优先级：** 先选固定形状、稳定控制流的热点子模块；同时记录初次编译成本、预热后收益与形状变化后的重新编译。训练需包含反向计时并校验梯度；实际运行不够长时，编译成本可能收不回来。

### 7. 评估用 `inference_mode` 要有边界

PyTorch 文档说明它比 `no_grad` 进一步去掉 view tracking 和 version counter 开销，但产生的 tensor 对后续 autograd 有更强限制，而且不会自动执行 `model.eval()`。[inference_mode API](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html)。

**建议：** 完整独立评估可比较 `eval()+inference_mode()`；训练时冻结视觉编码器的输出仍要参与投影层参数反向，不能把该局部 `no_grad()` 机械替换成 `inference_mode()`。它也不是 batching 或视觉特征复用的替代品。

## 最小验证方案（待执行）

1. 先记录固定 checkpoint、题目、输入形状、精度、线程数，以及冷启动和稳态耗时；训练额外固定有效 batch、梯度累积和优化器参数。基线指标是完成相同工作量的墙钟时间。
2. 按问题选择一次只改一项：数据/特征复用、批量处理、减少同步、BF16 autocast；确认收益后再考虑环境变量与编译。前向优化必须在含反向、日志、验证的完整链路复核。
3. 建议每个候选先做约 10 个预热迭代和 30–50 个计时迭代；真实耗时较长时缩短样本数，但重复测量，报告中位数与范围。不得与正在进行的正式训练争抢资源；由执行者安排独立时段运行。
4. 正确性门槛先于速度结论：同一 checkpoint 的概率、预测和损失应满足预先约定容差；混合精度训练还要检查梯度有限、更新非零及短程曲线。任何“提速几倍”都留待本机同口径实测。

## 未确定的部分

本资料确认的是机制与版本支持。尚未运行 Metal trace，也未测本项目 BF16、FP16、Metal matmul、fast math 或 compile 的真实收益与数值影响；不能据此断言当前瓶颈已定位。项目当前实现、硬件及优先级核查另行追加。

## deVision 本地核查与建议顺序

记录日期：2026-10-04。执行者：Codex。观察代码 `b1d545b7319e98696a621b98780930b812363287`；以下是代码、系统查询和已有结果的直接观察，未运行性能测试。正式建议编号 AS-PERF-01 至 04 同步在根目录 `critic/critic.md`；全部为可选择验证的优化项，不影响已通过的数据验收。

### 当前机器与任务

- Apple M4，24 GiB 统一内存，4 个性能核、6 个效率核；macOS 26.5.2。安装包元数据为 PyTorch 2.14.0、Transformers 5.17.0，与本次固定源码版本匹配。查询未 import torch 或加载模型。
- 查询时进程为 `devision-pipeline configs/v7-sqa-v5sets.yaml`，子进程执行 `test_vsr.mismatched`。这是 CPU 评估：[`cli.py:145`](../../src/devision/train/cli.py) 使用默认 CPU 的 `Decider.load`，评估 CLI 没有 `--device` 参数；YAML 顶层训练 `device: mps` 不会给独立评估子进程启用 MPS。CPU 是当前设计约定，不作为错误。
- V8 训练配置使用 MPS；[`rlcd.py:357`](../../src/devision/train/rlcd.py) 的 AMP 却只在 CUDA 开启，训练前模型为 FP32。这是值得首先验证的机制缺口，而非已证明的最大瓶颈。
- 当前系统 swap 快照约 5.46 GiB，memory_pressure 报 free percentage 78%；没有监测持续换页速率，不能仅凭已有 swap 判断评估被内存拖慢。

### 已经做过的优化，不重复当新方案

已经冻结 SigLIP2 并使用 no_grad，训练前完成文本 tokenization，ModernBERT 已显式使用 SDPA，AdamW 清梯度使用 `set_to_none=True`；独立评估已有同图同 state 合批、预热和完成结果跳过机制。当前与已有 V7 汇总报告的 CPU 线程数均为 4。增加 `--max-questions` 不会把不同图片合成一个 batch。

已有 CPU 结果只是规模和吞吐背景，**不同集合的数字不能组成加速 A/B**：

| 已有产物（`runs/v7-sqa/eval/`） | 题数 / 请求数 | 平均每题请求计算耗时 | 说明 |
| --- | --- | --- | --- |
| `test_vsr.json` | 904 / 552 | 115.84 ms | 本轮新集合；请求总耗时约 104.72 秒，未含加载、文件处理等完整墙钟时间 |
| `test_vqa_yesno.json` | 1,000 / 984 | 150.60 ms | 大多仍是单题请求 |
| `bench_lv_scienceqa.json` | 2,097 / 2,097 | 165.37 ms | 同图分组在这里不减少请求数 |
| `bench_lv_pope.json` | 9,000 / 500 | 54.61 ms | 每请求 18 题；纯请求总耗时约 491.46 秒 |

### 优先实施的小范围对照

| 顺序 | 候选改动与位置 | 适用原因 / 不能忽略的边界 | 执行成本与验证 |
| --- | --- | --- | --- |
| 1，训练 | MPS BF16 autocast：`rlcd.py:357,419`，若有效再同步训练内验证路径 | 当前 MPS FP32；先保留主权重与 RLCD 奖励/损失 FP32，不整模 `.half()` | 小幅实现改动；独立短跑固定 micro_batch=8，前向+反向+更新一起计时，核对梯度与 dev 数值 |
| 1，评估 | 完整评估 `inference_mode`、同 checkpoint 常驻模型：`decider.py:160`、`pipeline.py:270` | 当前每个集合与对照各启动子进程；加载占比尚未知。先测这一部分，保留 CPU 路径和对照 | 小至中等改动；固定一组含短长题、noul/choice 的样本，比较总墙钟与逐题输出 |
| 2，评估 | 多图片独立请求在共享模型核心内合批：`evaluate.py:98`、`Decider` | 当前只支持同图多问复用；不能把不同图硬塞成同一个 state。保持单请求协议与唯一计算逻辑，不绕开线上实现另写一套评估 | 中等实现工作；CPU batch 1/4/8/16 做同数据吞吐对照；单请求 CPU 延迟另测 |
| 2，训练/验证 | 缓存冻结 SigLIP2 的 patch 特征：`rlcd.py:98,165` | 优先反复验证的图片，不缓存仍在更新的 projector 输出。保留图片、镜像、错配关系和预处理身份 | 中等改动及一次预计算；缓存构建耗时计入总成本，验证 projector 梯度及缓存与重算输出 |
| 3，训练 | 窗口累计指标、减少 `.item()`：`rlcd.py:428–433`、`tracking.py:109` | 日志每 50 步写一次，但设备标量每步读取；累计必须 detach，避免保留训练图。并非每个重复读取都等价于一次完整 GPU 等待 | 小改动；保留完整每步历史和 resume 边界，测同步后的窗口耗时 |
| 3，训练/验证 | 按长度分桶，减少 padding：`rlcd.py:413`、`_val_logits` | 先保留 micro_batch=8、随机桶顺序与完整样本覆盖，不能全局按题型固定排序；样本顺序变化可能影响训练 | 中等改动；记录有效 token/padded token 比、最长 batch、内存和 samples/s，再验证学习曲线 |
| 4 | Metal matmul 开关、热点 compile、最后才 fast math | 以前面官方资料的版本/数值边界为准；不一次开多个开关 | 每项一个独立短对照；compile 冷启动单列，测到足够长才能判断是否回本 |

执行者先做约 10 步预热、30–50 步计时，必要时重复三次；这是待安排的预算建议，不是本次运行记录。训练和 CPU 评估可以分别挑最相关的一项，先投入数分钟至十余分钟测量，再决定是否继续实现其他项。批量扩大虽可能提高吞吐，但会改变每轮更新次数、有效 batch 和当前按 step 定义的 warmup/噪声/学习率进度，不能只以“每轮步数更少”证明等价提速。

### 特征缓存的实际空间与复用条件

SigLIP2 配置为 256 个 patch、每个 768 维。直接读取最终 JSONL 得到 train_mix 34,298 个不同图片路径、dev_mix 3,089 个路径；路径数不同于归一化图片数，因为原图与镜像必须保留不同输入。仅 patch 特征的理论容量为：

- train：FP32 **25.12 GiB**，FP16/BF16 **12.56 GiB**；不适合全部常驻这台 24 GiB 机器。
- dev：FP32 **2.26 GiB**，更适合先用有界缓存或磁盘映射；仍须加上模型/激活后评估总内存。
- V8 的 5,178 道 dev 在 step 0、每 1,000 步、epoch 末反复验证，末尾 LoRA 合并后还会计算校准 logits。冻结视觉分支可复用，后面的可训练层仍要重算。
- 单遍训练即使缓存也需先算所有特征；若图很少复用，纯预计算可能只把耗时搬到训练前。缓存键须包括视觉权重、图像内容及预处理、dtype；不能把镜像当同一图，也不能复用旧 projector 的输出。

### MLX 与 GPU 离线评估的定位

[MLX 统一内存说明](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)描述 CPU/GPU 共享 array 存储，[MLX compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)支持合并共同计算和算子融合。这是另一个值得研究的 Apple Silicon 方案，但**官方机制不能证明本项目迁移后更快**。deVision 的 ModernBERT 局部/全局 attention、投影层、LoRA、RLCD 和 CPU checkpoint 兼容都需要迁移与对齐，成本高于当前 PyTorch 内部优化；建议后置，不以生成式 LLM 的 token/s 报告推断本模型收益。

如果未来允许离线能力评估用 MPS，可为评估入口显式增加设备选择，在同一模型核心上先对齐 CPU/MPS 概率；这是对当前 CPU 评估约定的设计调整，尚未实施。正式 CPU 部署延迟测量仍应保留，GPU 批量吞吐不能替代它。

本次只新增研究与建议文档；所有候选优化均未实施，当前训练和评估任务未被干预。


## 后续已采纳改动验收

时间：2026-10-04 22:55:36 +08:00。执行者：Codex。代码范围 `b1d545b...2e215b4`。

**已实施的小改功能验收通过**：403abd1 已将加载/评估/服务/校准的默认设备改为 auto（CUDA > MPS > CPU），流水线传递 YAML 设备，并在评估结果记录实际设备；显式 CPU 仍可用。2e215b4 仅将 `Decider.decide` 改为 inference_mode，训练冻结视觉分支仍是 no_grad。因此上文基于 b1d545b 的“评估 CLI 只能用 CPU”是历史状态，已被本次修改取代。

AS-PERF-01 混合精度、AS-PERF-04 同步/内核等实验暂缓；AS-PERF-02 常驻模型/跨图片批处理、AS-PERF-03 特征缓存不采纳，接受控制复杂度的取舍。这些可选项不是本次验收障碍，也不记成已实施。

独立复算已有 `runs/analysis/mps_check/` 明细：同一 V7 checkpoint 的 500 题（VSR 200、Visual7W 300）CPU/MPS 预测全部相同，最大概率差 0.0004376769；NLL 为 0.68346014/0.68346144，ECE 为 0.05963689/0.05963792。数据 SHA256 为 `3d869506708f3a5b6df568bbac1a9436b6fd0a025eedbde5497aa07125a851fc`。两份产物标记 `b1d545b+dirty`，不冒充最终提交已重新执行。

**提速尚未独立验证**：同批 CPU 165.74 ms/题、MPS 227.92 ms/题，均 466 个请求。V8 任务 22:43:21 启动，训练本地日志目录时间 22:43:50，而 MPS 对照汇总写于 22:45:57；运行时间重叠，存在 GPU 争用混杂。不能依据这份记录认定“评估耗时大头已解决”，也不能据此认定独占 GPU 时 MPS 一定更慢。auto 是可用设备优先级，不是按实测吞吐择优。此限制不妨碍此次功能小改通过，未额外要求性能重测或大改。

验证范围：两条独立代码审查均无新增实质缺陷；本次执行变更 Python 语法、设备选择纯函数四种组合、diff 空白检查及既有结果复算。未启动 pytest、模型、校准或训练，未更改代码/配置和运行状态。验收完成后已按约定清空 critic/。

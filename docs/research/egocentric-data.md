# 第一人称（egocentric）日常任务决策数据：调研与选择

日期：2026-10-27。作者：Claude Code。状态：数据已生成（`data/v05/ego/`），**未训练、未评测**。代码：`scripts/data/v05_ego.py`（测试 `scripts/data/test_v05_ego.py`）。

目标：一张第一人称画面 + 英文问题 → 4–6 个选项（或是非），干扰项来自同一场景（画面里真实存在的物体、同一段视频里的其他步骤），只看文字难以猜对。

## 下载渠道

- **hf-mirror.com 当前不可用**：2026-10-27 实测，所有 `/api/...` 和 `/resolve/...` 请求都返回 308，跳转到 huggingface.co（即不再镜像）。按规定没有改走直连。
- 改用 **ModelScope**（国内）和 GitHub raw。ModelScope 的文件支持 HTTP Range，30 GB 的单个 zip 可以只读中央目录、按字节范围取需要的成员（逐个核对 CRC32），不必整包下载。
- 本机没有 ffmpeg、PyAV、OpenCV、decord：**需要从视频里抽帧的数据集目前都做不了**，只选自带画面的。

## 候选（ModelScope 上实际查到的）

| 数据集 | ModelScope 仓库 / 大小 | 内容 | 单帧可答 | 结论 |
|---|---|---|---|---|
| **EPIC-KITCHENS VISOR** + EPIC-KITCHENS-100 动作标注 | `qiup0307/EK100-VISOR`（30.6 GB 单 zip，我们只取 16 GB：标注 0.87 GB + train/val 稀疏帧 15.2 GB）；动作 CSV 来自 GitHub `epic-kitchens-100-annotations@ea8b4045` | 真实厨房第一人称帧（1920×1080），每帧人工分割：物体名称（开放词表）、左右手各自接触哪个物体；每帧落在某个 EPIC 动作片段（"cut onion"）里 | 是（帧本身就是标注对象） | **选用，训练 / dev / test**。CC BY-NC 4.0（只记录，不排除） |
| **EgoThink** | `lmms-lab/Egothink`（0.9 GB，12 个 parquet，700 题，Ego4D 画面） | 第一人称单图开放问答（活动、物体、定位、计划、推理） | 是 | **只评测** → `bench_egothink.jsonl`（开放答案改四选一，见下） |
| **EgoPlan-Bench** | `lmms-lab/EgoPlan`（题目 parquet 0.6 MB + 视频片段 zip 121 GB） | 任务目标 + 进度 → 下一步，四选一；3,343 题，2,420 题来自 EPIC-KITCHENS | 部分（依赖视频进度） | **只评测，暂不转换**：画面要从片段里解码，本机没有解码器；121 GB。只用题目 parquet 把它的时间窗排除在我们的数据之外 |
| EgoPlan-IT（EgoPlan 训练集） | 只在 HF（`ChenYi99/EgoPlan-IT`），ModelScope 未找到 | 5 万条下一步训练题 | 部分 | 镜像不可用，未取得 |
| EgoSchema | `lmms-lab/egoschema`（106 GB 视频） | 3 分钟长视频问答 | 否 | 不选 |
| HD-EPIC | `CircleRadon/hd-epic`（35 GB）等，视频 | 视频 VQA 基准 | 多为否 | 不选（需解码，且是评测集） |
| RoboVQA | `lmms-lab/RoboVQA`（只有 val 指令）、`allenai/Molmo2-ER-RoboVQA`（82 GB 视频） | 机器人 / 人的长程任务问答 | 部分 | 不选（需解码） |
| ECBench / RynnEC-Bench | ECBench 未找到；`DAMO_Academy/RynnEC-Bench`（49 GB，视频） | 具身认知评测 | 多为否 | 不选 |
| OpenEQA | 只有空的 `thomas/open-eqa-mirror` | 扫描场景的情景记忆问答 | 否（多帧） | 不选 |
| EgoTaskQA | ModelScope 未找到 | 视频任务推理 | 否 | 不选 |
| Ego4D / Ego-Exo4D 原始数据 | 官方需许可（`LibAIsii/*` 为 gated；其他转存来历不明，TB 级） | — | — | 不选（需许可） |
| EgoCross | `BruceLee0726/EgoCross`（2.3 GB） | 跨域第一人称（手术、工业、极限运动、动物），评测 | 部分 | 不是日常任务，暂不选 |

## 生成的题目（EPIC-KITCHENS VISOR）

三种题型。每帧最多 4 题，一题一个样本，`image_id = visor:<帧名>`，图片存为 640×360 JPEG（`data/v05/ego/images/visor/`）。

- **action**（"What is the person doing right now?"）：答案是包含该帧的 EPIC 动作片段的叙述（"cut onion"）；干扰项是**同一视频里其他片段的叙述**，名词不能和答案同类，也不能是任何一只手正在接触的物体；名词在画面里可见的排在前面。丢弃：该帧不在任何动作片段内、同时重叠着不同名词的动作、答案的物体在画面里没被分割出来（单帧看不出来）、干扰项不足 3 个。
- **contact**（"What is the person's left / right hand touching?"；两只手碰同一物体时只问一次）：答案是 VISOR 标的接触物体；干扰项是**同一帧里分割出来的其他物体**（不含手 / 手套、不同类、名称不互相包含）。丢弃：手没接触、接触了未分割的物体、标注不一致、目标在 256 下两边都小于 6 px、干扰项不足 3 个。
- **touching**（是非，"Is the person touching the X?"）：每帧至多一题，真假按帧名哈希各约一半，假的物体两只手都没碰。

选项数 4–6（干扰项 3–5 个），顺序按题目 id 的哈希打乱；答案位置分布、只看文字的基线写在 `MANIFEST.json`（只做诊断，不用来删题）。

**切分按参与者（= 厨房）**：test、dev 各约 10% 帧，参与者与训练集不重叠，看不到的厨房。由标注文件决定，部分下载时切分不变。

**与 EgoPlan-Bench 的重叠**：VISOR 的 158 个视频里有 134 个也是 EgoPlan-Bench 的源视频。落在 EgoPlan-Bench 题目时间窗（任务进度起点到 current_observation_frame 之后 300 帧）内的帧全部丢弃；视频层面的重叠仍在，以后若评 EgoPlan-Bench 的 EPIC 部分，对用了这批数据的模型应标为 monitoring 而不是 heldout；它的 Ego4D 部分（923 题）与我们的数据无关。

## EgoThink（只评测）

700 题都是开放答案。只保留答案 ≤ 6 个词的题；干扰项取**同一类别其他题**的答案，且与正确答案没有共同实词。干扰项不是同一画面里的，只看文字的基线会比 VISOR 题高，结果只作参考。Ego4D 画面，与 EPIC-KITCHENS 无重叠。

## 没做到的

- "下一步做什么 / 任务是否完成"：EPIC 没有任务目标标注，单帧也看不出进度；EgoPlan-Bench / EgoPlan-IT 是正对口的来源，需要视频解码器（PyAV 或 ffmpeg，经阿里云 pip 镜像可装）和 HF 镜像恢复后再做。
- 下载命令和续传：`uv run --no-sync python scripts/data/v05_ego.py fetch --root data`（日志 `data/v05/ego/fetch.log`，中断后再跑即续传），然后 `... build --root data`。

## 生成结果（2026-10-27，`data/v05/ego/MANIFEST.json`）

下载 16.1 GB（ModelScope，约 45 MB/s，7 分钟，0 错误），158 个视频全部用上（`complete: true`）。

| 文件 | 题数 | 帧 | 参与者 | action / contact / touching |
|---|---|---|---|---|
| `train.jsonl` | 43,617 | 21,325 | 19 | 19,871 / 14,191 / 9,555 |
| `dev.jsonl` | 7,367 | 3,516 | P01 P25 P26 | 3,314 / 2,374 / 1,679 |
| `test.jsonl` | 5,864 | 2,957 | P09 P13 P15 P17 P18 P20 P27 P35 P37 | 2,737 / 1,846 / 1,281 |
| `bench_egothink.jsonl`（只评测） | 604 | 604 | — | 12 个类别，四选一 |

主要丢弃（40,604 帧）：EgoPlan-Bench 时间窗 7,279 帧；action 题答案物体不可见 3,977、帧不在动作内 1,977、重叠动作 1,434；contact 题干扰项不足 20,038（只用"常被手碰"的类别，代价是少了一半 contact 题）、标注不一致 4,140、手没接触 3,442、接触未分割物体 2,537。

只看文字的基线（在 train 上拟合，test 上测）：

- action（6 选 1，机会 16.7%）：最长 15.7%、最短 17.3%、答案先验 17.9%，**基本是机会水平**。
- contact（4–6 选 1，机会 23.0%）：答案先验 **39.2%**（dev 34.0%）。剩下的是真实世界先验（刀比盘子更常被拿在手里）；干扰项已限定为至少 5% 的出现中被手碰过的类别、并优先选最常被碰的，从 43% 降到 39%，没有完全消除。用这部分数据时要看配错图对照。
- touching（是非）：train 多数类 47.9%，真假各半。
- EgoThink：最短选项 20–47%（按类别），因为干扰项来自别的画面；只作参考。

答案位置：各题型各选项数下大致均匀（见 MANIFEST `answer_position`）。

许可：VISOR / EPIC-KITCHENS 为 CC BY-NC 4.0，EgoThink 按其仓库；本项目只做研究，记录不排除。

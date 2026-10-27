# 真实手机 / 网页 GUI「下一步操作」选择题数据

日期：2026-10-27。作者：Claude Code。状态：**已生成数据，未训练**（GPU 上有别的任务，没有占用 GPU）。

题目形式：截图 + 用户目标（`instructions`）+ 已做过的操作（`state_text`，最多列最近 4 步）→ 从 4–6 个英文操作短语中选出演示者实际做的下一步，例如 `tap "Settings"`、`type "pizza" into "Search"`、`scroll down`、`go back`。干扰项优先取**同一屏上真实存在的其他元素**（AITW 用 OCR / 图标检测框，Mind2Web 用 HTML 候选元素），再补通用操作，这样只看文字没法排除。代码在 `scripts/data/v05_gui_fetch.py`（下载）和 `scripts/data/v05_gui_action.py`（转换，规则写在文件头），测试在 `scripts/data/test_v05_gui_action.py`（`uv run --no-sync pytest scripts/data -q -k gui`）。

## 1. 下载渠道

- **hf-mirror.com 当前不可用**：2026-10-27 20:57 起，API、`resolve` 文件、首页以外的所有路径都返回 308，跳到 huggingface.co（`curl -w %{redirect_url}` 验证过 gpt2、RICO-ScreenQA、FineVision）。按规定没有改走直连。
- **改用 ModelScope**（国内站点；CLAUDE.md 允许模型从 ModelScope 取，这里把同样做法用于数据集）。树 API：`https://www.modelscope.cn/api/v1/datasets/<repo>/repo/tree?Revision=master&Root=<dir>`，文件：`.../repo?Revision=master&FilePath=<path>`。实测下载速度 15–17 MB/s。如果负责人不接受用 ModelScope 下数据集，需要在使用这批数据前确认。

## 2. 候选数据集（均在 ModelScope 上实际查过）

| 数据集 | ModelScope 仓库 | 大小 | 截图 | 同屏元素标注（用来出干扰项） | 许可（只记录） | 结论 |
|---|---|---|---|---|---|---|
| **AITW**（Android in the Wild） | `russwest/AitW`：`data/general` 3 个 parquet、`data/web_shopping` 13 个 | 12.9 GB + 53.9 GB | 有 | **有**：每屏都有 OCR 文字和图标检测框（`ui_annotations_{positions,text,ui_types}`），还有点击 / 抬起坐标、输入文字、goal | 页面标 Apache-2.0；原始发布见 google-research/android_in_the_wild | **选用**。真实目标、真实手机截图；general 约 9.5k 个 episode、9.3 万步 |
| **Multimodal-Mind2Web** | `osunlp/Multimodal-Mind2Web` | 13.6 GB（47 个 parquet） | 有（整页长截图） | **有**：目标元素和负候选元素都带 `bounding_box_rect`，文字从 cleaned HTML 取 | OpenRAIL | **选用**。真实网站真实任务；官方 test_website / test_domain 的网站没在训练集出现 |
| AndroidControl | `smolagents/android-control`（67.4 GB，train 12,232 / test 3,051 个 episode）；`lizibing/Android_Control_test_aibox`（JSON + 11 GB 图片 zip） | 67 GB | 有 | **没有**：这两个版本都只有 goal、逐步指令和点击 x/y，没有无障碍树 | Apache-2.0 | 不选：没有元素文字，就给不出 `tap "X"` 和同屏干扰项（要先在 CPU 上跑 OCR，9 万张图要一天以上） |
| AitZ（Android-in-the-Zoo） | `junyuli/AITZ` | 2.65 GB | 有 | **没有**：只有 Observation / Thought / Action / Summary 文本 | Apache-2.0 | 不选。它本身是 AITW 的子集，AITW 已经覆盖 |
| GUI-Odyssey | 只有 `Voxel51/gui-odyssey-test` | 11.6 GB | 有 | 没有元素列表 | other | 不选（只有测试集） |
| AMEX | ModelScope 搜不到 | — | — | — | — | 无法获取 |
| GUIAct | `HongxinLi/GUIAct`（只有 45.8 GB 的 imgs.zip）；`smolagents/guiact-web-single`（7.6 GB，对话格式） | — | 有 | 对话格式里没有同屏元素列表 | Apache-2.0 | 不选 |
| OmniACT | `Writer-Org/omniact` | 0.42 GB | 有（桌面 / 网页） | 输出是 PyAutoGUI 脚本，没有元素列表 | MIT | 不选（量小，而且是桌面） |
| MultiUI | `neulab/MultiUI` | 954 GB | 有 | 网页 QA / grounding | ODC-BY | 不选：太大，也不是「下一步」类任务 |
| RICO-ScreenQA（完整版） | `rootsautomation/RICO-ScreenQA` | 3.3 GB | 有 | 答案带 `ui_elements` | CC-BY-4.0 | 是屏幕问答，不是操作。**本仓库的 UI 评测集（exp82v2 / exp85 的 `ui_dev`）就来自它**，这里不用 |
| RICO-SCA | `rootsautomation/RICO-SCA` | 2.1 GB | 有 | 有视图层级 | Apache-2.0 | 不选：只有控件描述，没有用户目标；而且也是 RICO 图，和 UI 评测集同源 |
| Magic-RICH / RealMobile | `FudanNLP/Magic-RICH`、`SeerRay-Lab/RealMobile` | 3.3 GB / 35 GB | 有 | — | Apache-2.0 / CC-BY-4.0 | 不选：中文；前者只用于评测，后者是模型轨迹，卡片写明不用于训练 |

## 3. 转换规则（详见 `v05_gui_action.py` 文件头）

- **AITW**：
  - 点击和滑动的区分：落点和抬起点相距 ≤ 0.04（归一化坐标）算点击，否则算滑动，按内容移动方向命名（手指上划 = `scroll down`）。
  - 点击的 gold 是包含落点的元素；没有元素包含时，用放大 1.4 倍的框再找一次（与 AITW 官方的匹配容差一致）。文字元素用文字命名，图标用检测类别命名（`tap the magnifying glass icon`）。
  - 以下情况丢弃：命中多个不同标签（ambiguous）；标签不可读（长度 < 2 或 > 40，或不含字母数字）；同屏另有同名元素（duplicate）；点在系统导航栏上（和通用操作「返回 / 回主屏」重复）；状态步骤（完成 / 不可能）；空输入。
- **Mind2Web**：
  - gold 标签取 `target_action_reprs`。整页截图裁成 1280×1000 的窗口，窗口必须包含目标，竖直位置按 id 随机，避免目标总在中间。
  - 干扰项只取窗口内的候选元素，并排除：与目标是同一个控件（嵌套或重叠）的；位于 menu / listbox / dialog / dropdown 等层里的（目标本身也在该层时除外）；框内几乎空白的（灰度标准差 < 8，即隐藏元素）；和其他标签不同的元素交叉重叠的（隐藏层）。
  - 窗口内另有同名元素时丢弃。
  - 输入 / 选择题的干扰项优先用「同一个值填进同屏另一个输入框」。
- **选项**：gold + 至多 3 个同屏干扰项 + 通用操作（`scroll down/up`、`go back`、`go to home screen`、`press enter`、本 episode 其他步骤输入过的文字），凑满 5 个。同屏干扰项少于 2 个时丢弃。「返回」和「回主屏」不会同时以 gold 和干扰项出现。选项顺序按 `sha256(id)` 固定种子打乱。
- **滑动 gold 上限**：每个来源最多占 20%。AITW 有很长的连续滑动，不加上限时约占 31%。
- **切分（dev / test 的目标或网站在 train 里都没出现过）**：
  - AITW general 按规范化后的 goal 文本分组，哈希 5% / 10% 分到 dev / test。
  - AITW web_shopping 按 goal 里的网站分组。
  - Mind2Web：官方 train + test_task 作为训练；官方 test_website + test_domain 的网站（官方训练集未见过）按网站哈希分到 dev / test。
  - dev / test 按整组截断到上限，超出上限的留出组**不进任何文件**。
  - 限制：AITW 不同 goal 之间仍会出现同一个 app 的界面（主屏、Chrome、Google 搜索页），只能做到「目标没见过」，做不到「app 没见过」。
- **与现有 UI 评测集的隔离**：每张输出图都和 `data/v05/exp82v2/ui_dev.jsonl`、`data/v05/exp85/ui_dev*.jsonl` 的图比较 dHash + pHash（与 v04_confirm 相同，≤ 6 bit 视为近重复）。这两个评测集是 RICO 图，和 AITW / Mind2Web 不同源，预期没有重复；结果见 MANIFEST。

## 4. 结果

（见下方「生成结果」，数字来自 `data/v05/gui/MANIFEST.json`）

## 5. 已知问题

- **256 下的可读性**：AITW 截图多为 270×600 到 720×1520，letterbox 到 256 后文字只有 2–4 px；Mind2Web 窗口 1280×1000 缩到 256×200。很多题模型只能靠版式、图标和位置来判断。这和 exp82–85 的 UI 题是同一类限制，适合作为切块方案（v05-training-directions 方向 2）的测试集。
- **OCR 噪声**：AITW 的标签是 OCR 结果，会有错字，例如 `Grmail`、`Mediterrangarn restaurart`。gold 和干扰项用的是同一套 OCR，所以题本身是一致的，只是读起来不够规范。
- **单条演示的歧义**：每一步只有一条人工演示，其他也能完成目标的操作（例如先滑动再点）会被当作错。已去掉最明显的等价对（返回 ↔ 返回箭头图标、导航栏 ↔ 通用操作）。
- **只看文字的基线偏高**（goal 里常直接写出要点的词或要输入的文字），见下方 text_only。看图的增益要用配错图对照来测，不能只看准确率。
- 没有做是非题变体。

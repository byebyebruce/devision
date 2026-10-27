# exp108：官方图像评测与 score 契约定位

2026-10-25 17:04 +08:00 预写。承接exp107固定发布者commit，CPU元数据实验，零训练、不占GPU、不生成checkpoint。假设：公开页面和原生执行代码能区分ImageJevBench专用契约与纯文本JevBench的score规则；不能将后者直接套用到前者。

最小对照：同一commit的ImageJevV03Page、官方API页面、native_image和official_scoring源码，记录URL/哈希，逐一核对适用范围、请求/响应与评分。仅下载公开源文件，不执行外部代码、不提交模型，不访问封存题。每请求30秒、总240秒退出，独立runs/exp108-official-contract，失败保留。成功条件是形成带行号的明确契约或最小缺口；没有明确映射不改产品API，不猜造score概率。

停止条件：时间预算到、网络异常或源码仅描述其他榜单时停止扩展。18:00冻结训练；23:00汇总可行回归/校准与缺口，周一08:00缓冲、09:00截止。当前科学图示退化、五用途与正式public300仍阻断达标发布，不用接入分析代替能力验收。

## 2026-10-25 17:05:25 +08:00：完成

实际下载进程exit0，EXIT_CODE=0；固定commit `4365ca7ec2ee12ea952079185cbf68f37bbf7ff1`的5份公开源码原件、URL及SHA256在`runs/exp108-official-contract/manifest.json`，全部哈希独立复核通过（verification.json）。零GPU调用、零训练、无新权重。

- **阻断，直接观察**：`components__ImageJevV03Page.tsx:78`明确sealed draw为600 choice / 300 Noul / 300 score，不能用choice专项代表正式全题型。当前产品拒绝score符合现有规格，但不足以证明完整官方兼容。最小解除证据仍为完整官方可执行接入及score语义，不能用自定义ECE替代Calibration。
- **重要，直接观察**：`ops__priority-evaluation__pod_drivers__native_image.py:1-10`明示proposed/not an official method admission，`64-83`只生成choice。它的存在不解除接入阻塞。`161`附近将图片置于state.image.data，按本项目spec该嵌套对象是文字，不是图像片段；若未来接入需显式传入本项目image参数并验证真实视觉输入。当前未执行该外部代码、未改产品API，不能称已修复。
- **适用范围**：`ops__priority-evaluation__official_scoring.py:40-49`对文本jevbench检查三题型support枚举，对imagejevbench检查endpoint kind；该检查不是题型评分实现，不可据此推断图像榜不需要score。完整评分仍未取得，本轮不下载或执行封存数据。

决定：不将choice-only proposed适配器直接产品化，不重复失败训练或小面板数值实验。下一轮检查同commit official_score.py及其公开依赖是否给出图像score数学定义与运行要求；只检查公开源码，20分钟CPU预算，无明确语义则输出最小接入需求及发布缺口。18:00训练冻结，23:00完成已有证据的发布就绪矩阵；当前不具备合格最终候选，不做无依据的发布校准。py_compile、哈希/契约静态断言、git diff --check通过；不是正式模型评测或全仓测试。

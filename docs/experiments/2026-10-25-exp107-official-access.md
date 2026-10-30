# exp107：正式评测接入材料核对

2026-10-25 16:05 +08:00预写。零训练、CPU/网络元数据实验，不占GPU、不生成checkpoint。目的：区分当前公开发布渠道是否已新增可验证的public300材料，或仍仅汇总和8示例；固定发布者main的实际commit再查完整tree，保存原始响应与哈希，不能把文件名搜索无结果外推成题集不存在。

输入：官方版本化v0.3.0 results、发布者GitHub commit/tree、同commit示例manifest。每请求30秒、总240秒硬超时；异常记录并退出，不重试猜测路径、不访问封存数据、不发提交。独立目录runs/exp107-official-access，禁止覆盖。成功标准为证据归档完整、tree未截断、示例计数与聚合身份可核实；只有得到明确完整题集/执行入口才解除接入阻塞。另运行现有decide非法score契约测试，核实当前发布接口能否执行官方全题型；测试通过表示拒绝行为符合当前规格，不表示正式评测兼容。

无合格模型时不提交未达标版本、不在monitoring面板拟合发布温度。18:00训练冻结、23:00回归/独立校准可行项和缺口材料、周一08:00缓冲、09:00截止；五用途与科学图示退化仍需单独解决。

## 2026-10-25 16:05:51 +08:00：完成

实际工具会话exit0、目录EXIT_CODE=0。归档results/commit/tree/examples原始JSON及fetches哈希；固定commit `4365ca7ec2ee12ea952079185cbf68f37bbf7ff1`，完整tree 43352条且truncated=false，未发现public/core路径；manifest明确8题public_examples_only。官方source_sha256仍为`b7e1ba2ee58baba1f442924db9d4760934425e2061fdee88dc94e45d78150bb5`，built_at未变。这只证明本次公开渠道核查结果，不能证明完整题集不存在。原始产物`runs/exp107-official-access/`。

现有decide非法请求测试实际16 passed/21 deselected、exit0，包含score拒绝；实现`src/devision/infer.py:204`与当前规格一致，但正式全题型接入仍缺失。官方data card仅提供系统汇总，普通提交队列免费且审核/接受时间由维护者决定，本轮没有发送提交。

决定：当前无合格整体替换候选，不启动没有新假设的训练、不复跑exp106、不拟合虚假发布温度。下一轮优先从已归档固定tree定位官方score schema/评分与执行适配代码，形成可验证契约再实现，预算30分钟CPU；缺少明确语义时保留阻塞，不猜测Capability。public300合法归档/正式执行入口、五用途提升、独立校准及完整外部回归继续阻断v0.5达标发布。

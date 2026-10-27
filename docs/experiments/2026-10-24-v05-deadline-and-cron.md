# v0.5发布期限与CC-Connect调度

记录日期：2026-10-24，Asia/Shanghai。负责人明确要求：“你用cc connect的Cron来给自己定时，你的目标就是要在周一早上之前发布为0.5版本。一直朝着这个目标努力。”

执行解释：发布期限暂按2026-10-26（星期一）09:00 +08:00倒排；本条明确授权本项目v0.5正式发布相关操作，无需就同一版本重复请求授权。既定质量与正式验收要求仍有效，未达到目标时不得伪称发布完成。不增加付费算力、不擅改许可。具体发布渠道及步骤沿用项目发布约定。

已创建CC-Connect Cron：任务ID `064e9c68`，`0 * * * *`，每小时整点；project `proj-codex_ws_a6815f35`，绑定当前deVision工作区Codex会话，reuse模式，每次50分钟超时。使用运行中CC-Connect的实际数据目录 `/home/bruce/app/cc-connect/data`，而非未使用的默认目录。通过指定project的cron list核实任务已启用。

任务要求每轮检查实际进程/GPU/日志、分析上一实验、修复故障；无活跃正常GPU任务时基于证据启动下一项，训练运行时准备数据和下一项。保留旧权重/失败证据，禁止重复抢卡和占卡空跑。进展写训练日志，按正常CC-Connect会话返回。

已移除系统crontab中的hourly_engineer执行入口，并用 `runs/v05-monitor/engineer/PAUSE` 禁用旧执行器；原系统只读小时监控保留。停止本轮自行启动的两个独立Codex工程执行进程，避免与当前会话及CC-Connect调度并发改动；未停止其他执行者任务。

倒排优先级：周六解决候选推理适配与同题基线；周日尽早确定路线并完成有效候选训练；周日晚完成回归、校准、性能和模型卡，留出周一前正式提交/发布缓冲。实际进度与瓶颈写入训练日志。ImageJevBench访问及五用途完整证据仍为关键路径，不能用局部开发集代替；遇到影响期限的实质阻塞应提前报告。

维护：`cc-connect cron list --project proj-codex_ws_a6815f35 --data-dir /home/bruce/app/cc-connect/data`。目标完成或负责人暂停后停止新实验并撤销/停用此推进任务；不无限创建递归任务。

## 19:37 调度故障修正

用户反馈未见定时推进，实际last_error为project "proj-codex_ws_a6815f35" not found：此前误用工作区会话名为配置项目名。已通过CC-Connect cron edit将同一任务064e9c68改为真实项目proj-codex，保留原session_key用于工作区/消息路由。首次补触发复用会话报busy，随后改session_mode=new_per_run并再次补触发；系统已观察到新Codex执行进程370376（内部370383），参数--cd /home/bruce/proj/devision。证明执行已实际启动，不代表该轮实验已完成；last_run/last_error在运行完成前可能仍显示上一失败记录。原整点0 * * * *与50分钟超时保留。原先“只核实enabled”不足以证明调度有效，本次补做运行验证。

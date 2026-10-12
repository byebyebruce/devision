# 从命令行触发 cc-connect 当前会话

记录时间：2026-10-08 23:20:21 +08:00。本轮为使用方式调研，不是项目评审；没有创建任务、发送消息、启动模型、修改配置或重启服务。

## 结论

本机可用的一条命令方案是 **`cc-connect timer add --delay 1s --session-mode reuse --prompt '任务'`**，同时指定项目、会话和实际数据目录。它只执行一次，能解析本机的多工作区绑定，沿用这个聊天当前激活的模型会话，结果走原平台回复；不需要创建循环 cron 再禁用。但它仍然是一次性调度，忙时会失败，不能当作忙时排队的实时注入接口。[v1.5.0 CLI](https://github.com/chenhg5/cc-connect/blob/v1.5.0/cmd/cc-connect/timer.go#L402)、[工作区及会话解析](https://github.com/chenhg5/cc-connect/blob/v1.5.0/core/engine.go#L1771)

更符合理想接口的 `cc-connect send --as-prompt` 已有上游 [PR #575](https://github.com/chenhg5/cc-connect/pull/575) 和 [PR #593](https://github.com/chenhg5/cc-connect/pull/593)，但查询时均为 **open、未合并**；本机 v1.5.0 和查询到的最新预发布均没有该参数。不能把 PR 中的命令当作已发布功能。

## 本机已核实的状态

以下由主审进程在本轮只读核查，并与上游源码交叉确认：

- 已安装 cc-connect 1.5.0；顶层帮助没有列出 timer，但 `cc-connect timer add --help` 可以正常显示。
- 实际数据目录为 `/Users/tu/Documents/app/cc-connect/data`，项目是 `proj-codex`；项目使用 `base_dir=/Users/tu/Documents/proj` 的多工作区模式。
- 对实际 Unix socket 执行只读 `GET /timer/list?project=proj-codex`，返回 HTTP 200、JSON `null`，说明当前服务提供 timer API，查询时没有这个项目的 timer 记录。
- 当前工作区会话文件 `sessions/proj-codex_ws_ce268a6f.json` 的激活记录为 `s2`；其模型会话 ID 与当前 `CODEX_THREAD_ID` 一致。2026-10-08 23:20 +08:00 再次核对一致。
- 当前配置未配置 `[webhook]` 或 `[management]`；未为了调研启用它们。没有输出凭据。

可供用户在当前回复结束后执行的完整命令：

```bash
cc-connect timer add \
  --data-dir /Users/tu/Documents/app/cc-connect/data \
  --project proj-codex \
  --session-key 'feishu:oc_9d5b5ec3c5a5bacfb8d9fe411f968d4b:ou_0f8aed6eabaf1898a25afa6e1b411fd6' \
  --session-mode reuse \
  --delay 1s \
  --prompt '方案已经修改，请重新验收'
```

`reuse` 选择的是触发时该聊天的激活会话，不是永久固定某个 Codex UUID；执行前若切换了聊天会话，会使用切换后的上下文。任务只触发一次，执行后保存 fired 状态及错误历史，不会自动删除记录。[会话选择](https://github.com/chenhg5/cc-connect/blob/v1.5.0/core/engine.go#L1820)、[执行与状态持久化](https://github.com/chenhg5/cc-connect/blob/v1.5.0/core/timer.go#L393)

如果触发时会话忙，`TryLock()` 失败后返回错误并记录为已触发失败，不自动排队或重试。CLI 成功创建 timer 也不等于任务已执行成功。本轮没有实际触发，因此结论是帮助、运行中只读 API、会话映射和源码共同支持该方式，尚无端到端执行证据。

## 最新上游核查

访问日期为 2026-10-08。GitHub releases API 返回最新稳定版 **v1.5.0**（2026-08-16），最新预发布 **v1.5.1-beta.3**（2026-09-29）。查询的 main 为 `dfad19415a38b00b2c5c288610784d1a7eef337f`；v1.5.1-beta.3 为 `e27354b2365ee8c8e9a1b6fc4ea57cd71cdf9ead`。本轮分别读取 main、稳定版、最新预发布的相关源码。[稳定版](https://github.com/chenhg5/cc-connect/releases/tag/v1.5.0)、[预发布](https://github.com/chenhg5/cc-connect/releases/tag/v1.5.1-beta.3)

| 入口 | 实际语义 | 对当前目标的判断 |
| --- | --- | --- |
| `cc-connect send` / Unix socket `POST /send` | 把内容发送到聊天平台 | 不触发模型 |
| Management `POST /api/v1/projects/{name}/send` | 实现同样调用 `SendToSession`，最终 `p.Send` | 不触发模型；文档草案描述与实现不符 |
| `cc-connect timer add ... --delay 1s --session-mode reuse` | 一次性任务，解析工作区后获取激活会话并进入正常模型处理流程 | 本机最简现成方案；忙时失败 |
| Webhook `POST /hook` | 真正触发模型，异步执行并向原平台回复 | 已发布，但未正确路由本机多工作区会话；不直接推荐启用 |
| `cc-connect send --as-prompt` | PR 提议的入站提示词入口 | 未合并，不可直接使用 |

### Management API 文档不能代替实现

[官方 API 文档](https://github.com/chenhg5/cc-connect/blob/dfad19415a38b00b2c5c288610784d1a7eef337f/docs/management-api.md)标注 `1.1-draft`，把 `/projects/{name}/send` 描述为向 agent 发送用户输入，但实际 [handler](https://github.com/chenhg5/cc-connect/blob/dfad19415a38b00b2c5c288610784d1a7eef337f/core/management.go#L1328) 调用 `e.SendToSession`，再由 [SendToSessionWithOptions](https://github.com/chenhg5/cc-connect/blob/dfad19415a38b00b2c5c288610784d1a7eef337f/core/engine.go#L11488) 调用平台 `p.Send`。v1.5.0、最新预发布和 main 均是这个行为。换成 curl 调这个管理端点，仍然只是发平台消息。

### Webhook 确实入站，但缺少当前工作区路由

已发布的标准入口是 `POST /hook`（默认端口 9111）。请求可包含 `project`、必填 `session_key`、`prompt`、可选 `event`、`payload` 和 `silent`；`prompt` 与执行 shell 的 `exec` 互斥。需显式启用 `[webhook]`。有 token 时支持 Bearer、`X-Webhook-Token` 或 query token；空 token 不鉴权，监听地址为 `:port`，因此若未来启用，应配置认证并限定可访问范围。[请求与认证源码](https://github.com/chenhg5/cc-connect/blob/dfad19415a38b00b2c5c288610784d1a7eef337f/core/webhook.go#L27)、[官方配置示例](https://github.com/chenhg5/cc-connect/blob/dfad19415a38b00b2c5c288610784d1a7eef337f/config.example.toml#L388)

仅表示协议、不是本机可直接执行的示例：

```json
{"project":"my-project","session_key":"feishu:chat-id:user-id","prompt":"检查最新提交","event":"manual"}
```

它先返回 HTTP 200 和 `{"status":"accepted","event":"manual"}`，随后后台执行。accepted 不证明会话存在、成功获取锁或模型完成；忙时直接丢弃，并在非 silent 模式向平台提示跳过。[执行入口](https://github.com/chenhg5/cc-connect/blob/dfad19415a38b00b2c5c288610784d1a7eef337f/core/webhook.go#L88)

关键限制是 `executePrompt` 使用 `engine.sessions.GetOrCreateActive(sessionKey)`，随后 `processInteractiveMessage` 使用基座 `e.agent/e.sessions`，没有解析 `workspaceContext`。本机当前上下文存放在独立工作区 SessionManager，因此这条路径不能保证触发现有 devision 会话，而可能新建另一条上下文。此缺口在 v1.5.0、最新预发布和 main 均可从源码确认；本轮未实际发送请求复现。[Webhook 会话选择](https://github.com/chenhg5/cc-connect/blob/dfad19415a38b00b2c5c288610784d1a7eef337f/core/webhook.go#L244)、[基座处理入口](https://github.com/chenhg5/cc-connect/blob/dfad19415a38b00b2c5c288610784d1a7eef337f/core/engine.go#L3923)

### 真正的注入入口仍在提案中

[PR #575](https://github.com/chenhg5/cc-connect/pull/575)直接针对外部脚本触发工作，提议 `send --as-prompt`；[PR #593](https://github.com/chenhg5/cc-connect/pull/593)提议与新线程发送组合。GitHub API 均返回 `state=open`、`merged=false`。它们适合跟踪，不能直接作为升级后可用功能，也未在本轮验收其实现质量。

[Issue #1542](https://github.com/chenhg5/cc-connect/issues/1542)要求已有会话的本地 inject、忙时排队和可选等待结果；该 issue 最后由 stale bot 因无活动关闭，不代表需求已实现。现阶段无需为了单次触发更换到未合并分支：一次性 timer 已覆盖本机空闲会话的任务投递；若将来需要忙时可靠排队、幂等及同步返回结果，再评估正式 inject 接口。

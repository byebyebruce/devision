# 发布 Python 包到 PyPI

推送 `v<包版本>` 标签（如 `v0.2.0`）会触发 [`.github/workflows/publish.yml`](../.github/workflows/publish.yml)：
校验标签与包版本一致 → 构建 wheel 和源码包 → `twine check` → 在不装模型依赖的环境里装 wheel，确认 `import devision` 不加载 torch、demo 文件和 `devision-serve` 都在 → 确认源码包只含 `src/`、README、LICENSE、pyproject → 上传 PyPI。

上传用 PyPI 的 Trusted Publishing：GitHub 不保存任何 PyPI 密码或 token。

版本号的分工：

| 标签 | 在哪 | 是什么 |
|---|---|---|
| `v0.2.0` | GitHub | PyPI 上的 Python 包 `devision`（触发本流程） |
| `model-v0.2` | GitHub | 发布模型时用的代码 |
| `v0.2` | Hugging Face | 模型权重 |

## 首次设置（只做一次）

1. **PyPI**：登录 PyPI → Your projects → Publishing → *Add a new pending publisher*（项目还不存在时用这个）：
   - PyPI Project Name：`devision`
   - Owner：`byebyebruce`，Repository name：`devision`
   - Workflow name：`publish.yml`
   - Environment name：`pypi`
2. **GitHub**：仓库 Settings → Environments → New environment，名字 `pypi`。建议在 *Deployment protection rules* 里勾选 *Required reviewers* 并加上自己：标签推上去后，要在 Actions 页面点一下批准才会真正上传。

## 发一个版本

1. 改 `pyproject.toml` 的 `project.version`（唯一的版本号；`devision.__version__` 从安装信息读取），再 `uv lock`，提交并推送。
2. 打标签并推送：

   ```bash
   git tag -a v0.2.0 -m "devision 0.2.0"
   git push origin v0.2.0
   ```

3. 在 [Actions](https://github.com/byebyebruce/devision/actions/workflows/publish.yml) 里看结果（设了审批就先批准）。成功后 `pip install devision==0.2.0`。

PyPI 上的版本不能覆盖或重传；内容有误就升版本号、打新标签。只有 `v<数字>.<数字>.<数字>` 形式的标签会触发，`model-v0.2` 等不会。

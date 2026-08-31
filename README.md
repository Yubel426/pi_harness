# pi-terminal-harness

一个用 Python 实现的最小 Pi-like terminal harness。它只内置 `bash` 工具，但保留后续 3D vision agent 真正需要的核心扩展面：任何 Python 函数都可以注册成 JSON Schema tool。

运行方式是 **一个 prompt → 自主工具循环 → 最终结果 → 退出**，不提供交互式多轮对话或跨任务会话历史。

参考的是 [Pi coding-agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent) 及其 agent loop（核对版本：`853a80d26c90a14c1886f0ebb8ffaae133ca2185`）。本项目刻意不移植 Pi 的 TUI、session 文件、compaction、skills、extensions、主题和 provider registry。

## 保留的核心语义

- 每次 `Agent.run(prompt)` 是独立任务，只接收一次用户输入。
- 任务内部保留 Responses API 的 `model → tool → tool output → model` 循环；单次输入不等于只调用一次模型。
- 同一响应内的所有 function call 都会执行并回填，然后再请求模型。
- `store=False`，仅在当前任务内重放完整 response output items（包括 reasoning）；上下文是 `run()` 的局部变量，不保存在 Agent 上。
- 流式输出模型文本和 shell 输出。
- 未知工具、非法 JSON 参数、Python tool 异常都会作为 tool output 返回模型，供当前任务处理。
- 不执行来自 incomplete response 的工具调用，避免截断参数导致误操作。
- `bash` 合并 stdout/stderr，支持超时、Ctrl-C/进程组终止和非零退出码。
- `bash` 输出保留最后 2000 行或 50KB；截断时把完整输出写入临时文件。
- `OPENAI_API_KEY` 和 `PI_API_KEY` 不会传给模型启动的子 shell。

## 安装与运行

推荐使用 `uv`：

```bash
uv sync
```

凭据只通过环境变量提供，不要写入源码或提交到 Git：

```bash
export PI_API_KEY='sk-...'
export PI_BASE_URL='https://api.monkeycoding.club'
export PI_MODEL='gpt-5.6-sol'

uv run pi-harness -p '运行 pwd，并只告诉我目录名'
```

`PI_BASE_URL` 缺少 `/v1` 时会自动补齐。默认模型是 `gpt-5.6-sol`，默认 reasoning effort 是 `medium`。

`-p/--prompt` 必填。未提供或传入空白 prompt 时会直接报错退出，不会进入输入等待。

常用参数：

```text
--cwd PATH
--model MODEL
--base-url URL
--reasoning-effort none|low|medium|high|xhigh|max|off
--max-output-tokens N
--max-tool-rounds N
--no-stream
```

没有 REPL、斜杠命令或 `Agent.reset()`。成功、失败或中断后均不会向下一次 `run()` 传递任务上下文；工具产生的文件等外部副作用不会自动撤销。

## 注册 Python / 3D 工具

`Tool` 的 executor 接收解析后的参数和 `ToolContext`。返回值可以是字符串、mapping 或 `ToolResult`：

```python
import os
from pathlib import Path

from openai import OpenAI

from pi_harness import Agent, Tool, ToolContext


def inspect_scene(args, context: ToolContext):
    scene_path = (context.cwd / args["path"]).resolve()
    # 在这里调用 Open3D、PyTorch3D、Blender Python API 等。
    return {"scene": str(scene_path), "objects": 12}


scene_tool = Tool(
    name="inspect_scene",
    description="Inspect a local 3D scene and return a compact summary.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    executor=inspect_scene,
)

client = OpenAI(
    api_key=os.environ["PI_API_KEY"],
    base_url="https://example/v1",
)
agent = Agent(
    client=client,
    model="gpt-5.6-sol",
    instructions="You are a 3D vision agent.",
    tools=[scene_tool],
    cwd=Path.cwd(),
)
print(agent.run("Inspect scene.glb").text)
```

CLI 为了保持最小，只注册 `bash`。3D agent 可以直接复用 `Agent`，把自己的 Python tools 传入 `tools`；无需让模型通过 shell 间接调用 Python。

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

真实端点 smoke test 会产生 API 用量，因此不放进自动化单元测试。可以用上面的单次执行命令验证完整的 streaming + bash tool loop。

## 安全边界

这是本地 coding/3D agent harness，不是 sandbox。`bash` 拥有当前用户对 `--cwd` 及系统的权限；只应在可信工作区运行。API 凭据虽然不会传给 `bash`，但工作区已有的 `.env`、云凭据文件和其他敏感文件仍需要由运行环境自行隔离。

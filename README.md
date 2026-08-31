# pi-harness

A minimal Python terminal agent harness backed by [Pi's unified LLM API](https://github.com/earendil-works/pi/tree/main/packages/ai). Python owns the agent loop and tools; a small Node.js bridge uses the upstream `@earendil-works/pi-ai` library for providers, authentication, model catalogs, and streaming.

Each run takes a single prompt, executes an autonomous tool loop, returns a result, and exits. There is no interactive chat or cross-task conversation history.

## Quick start

Requires Python 3.11+, Node.js 22.19+, npm, and `uv`.

```bash
uv sync
uv run pi-harness setup
uv run pi-harness login  # Select a provider and API key or OAuth (where supported)
uv run pi-harness -p 'Run pwd and tell me the directory name'
```

For OpenAI, select `openai` for API keys or `openai-codex` for OAuth. Credentials are saved locally. For OAuth runs, add `--provider openai-codex` to the run command.

Use `uv run pi-harness providers` to list supported providers and `--provider` / `--model` to select one.


## Custom Python / 3D tools

The CLI registers `bash`. Pass Python tools to `Agent` to extend it directly:

```python
from pathlib import Path

from pi_harness import Agent, PiClient, Tool, ToolContext


def inspect_scene(args, context: ToolContext):
    scene_path = (context.cwd / args["path"]).resolve()
    # Call Open3D, PyTorch3D, or the Blender Python API here.
    return {"scene": str(scene_path)}


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

with PiClient(provider="openai") as client:
    agent = Agent(
        client=client,
        model="gpt-5.6-sol",
        instructions="You are a 3D vision agent.",
        tools=[scene_tool],
        cwd=Path.cwd(),
    )
    print(agent.run("Inspect scene.glb").text)
```

Existing `Agent(client=OpenAI(...), ...)` integrations remain supported through the optional `openai` extra (`uv sync --extra openai`).

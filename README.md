# pi-harness

A lightweight agent runtime for AI tools written in Python, with tool calling, per-task state management, and [Pi's unified LLM API](https://github.com/earendil-works/pi/tree/main/packages/ai). Python owns the agent loop and tools; a small Node.js bridge uses the upstream `@earendil-works/pi-ai` library for providers, authentication, model catalogs, and streaming.

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

## Custom Python AI tools

Wrap existing Python functions, models, or pipelines as tools for inference, computer vision, text processing, retrieval, data analysis, or 3D workflows. Tools run directly in Python, so they can reuse your libraries and loaded models without going through shell commands.

The CLI registers `bash`. Pass custom `Tool` objects to `Agent` to integrate your own Python AI tools. This minimal text-processing example shows the interface; replace the function body with your own implementation:

```python
from pathlib import Path

from pi_harness import Agent, PiClient, Tool, ToolContext


def analyze_text(args, context: ToolContext):
    text = args["text"]
    return {"characters": len(text), "words": len(text.split())}


text_tool = Tool(
    name="analyze_text",
    description="Count the characters and words in a text.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    executor=analyze_text,
)

with PiClient(provider="openai") as client:
    agent = Agent(
        client=client,
        model="gpt-5.6-sol",
        instructions="Use the available Python tools to complete the task.",
        tools=[text_tool],
        cwd=Path.cwd(),
    )
    print(agent.run("Analyze this text: Python tools power AI workflows.").text)
```

Existing `Agent(client=OpenAI(...), ...)` integrations remain supported through the optional `openai` extra (`uv sync --extra openai`).

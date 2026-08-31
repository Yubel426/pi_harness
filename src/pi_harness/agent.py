"""Stateless Responses API agent loop with local Python function tools."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools import Tool, ToolContext, ToolResult


class AgentError(RuntimeError):
    """Raised when the provider or agent loop cannot complete safely."""


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def add_response(self, response: Any) -> None:
        usage = _field(response, "usage")
        if usage is None:
            return
        self.input_tokens += int(_field(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(_field(usage, "output_tokens", 0) or 0)
        self.total_tokens += int(_field(usage, "total_tokens", 0) or 0)
        input_details = _field(usage, "input_tokens_details")
        output_details = _field(usage, "output_tokens_details")
        self.cached_tokens += int(_field(input_details, "cached_tokens", 0) or 0)
        self.reasoning_tokens += int(_field(output_details, "reasoning_tokens", 0) or 0)


@dataclass(slots=True)
class RunResult:
    text: str
    usage: Usage
    model: str
    response_id: str | None
    api_rounds: int
    tool_calls: int


def _noop_text(_: str) -> None:
    return None


def _noop_tool_start(_: str, __: Mapping[str, Any]) -> None:
    return None


def _noop_tool_output(_: str, __: str) -> None:
    return None


def _noop_tool_end(_: str, __: ToolResult, ___: bool) -> None:
    return None


@dataclass(slots=True)
class AgentCallbacks:
    on_text_delta: Callable[[str], None] = _noop_text
    on_tool_start: Callable[[str, Mapping[str, Any]], None] = _noop_tool_start
    on_tool_output: Callable[[str, str], None] = _noop_tool_output
    on_tool_end: Callable[[str, ToolResult, bool], None] = _noop_tool_end


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _dump_item(item: Any) -> Any:
    if isinstance(item, Mapping):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)
    return item


def _extract_text(response: Any) -> str:
    output_text = _field(response, "output_text")
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for item in _field(response, "output", []) or []:
        if _field(item, "type") != "message":
            continue
        for content in _field(item, "content", []) or []:
            if _field(content, "type") == "output_text":
                text = _field(content, "text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


class Agent:
    """Single-prompt task runner modeled after Pi's core loop.

    Each run keeps its own Responses output items and tool results until the task
    finishes. This preserves reasoning for ``store=False`` tool continuation without
    retaining conversation history between independent tasks.
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        instructions: str,
        tools: Sequence[Tool],
        cwd: str | Path,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int | None = None,
        max_tool_rounds: int = 100,
        stream: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.instructions = instructions
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")
        self.cwd = Path(cwd).expanduser().resolve()
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_tool_rounds = max_tool_rounds
        self.stream = stream

    def run(self, prompt: str, callbacks: AgentCallbacks | None = None) -> RunResult:
        """Execute one independent task; never carry context over from earlier runs."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        callbacks = callbacks or AgentCallbacks()
        input_items: list[Any] = [{"role": "user", "content": prompt}]

        usage = Usage()
        tool_call_count = 0

        for api_round in range(1, self.max_tool_rounds + 2):
            response, streamed_text = self._create_response(input_items, callbacks)
            usage.add_response(response)
            response_text = _extract_text(response)
            if response_text and not streamed_text:
                callbacks.on_text_delta(response_text)

            output = list(_field(response, "output", []) or [])
            input_items.extend(_dump_item(item) for item in output)
            calls = [item for item in output if _field(item, "type") == "function_call"]

            status = _field(response, "status", "completed")
            if status != "completed":
                reason = _field(
                    _field(response, "incomplete_details"), "reason", status
                )
                if not calls:
                    raise AgentError(f"Response did not complete: {reason}")
                for call in calls:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": _field(call, "call_id"),
                            "output": (
                                "Error: tool call was not executed because the model response "
                                "was incomplete and its arguments may be truncated. Re-issue it."
                            ),
                        }
                    )
                continue

            if not calls:
                return RunResult(
                    text=response_text,
                    usage=usage,
                    model=_field(response, "model", self.model),
                    response_id=_field(response, "id"),
                    api_rounds=api_round,
                    tool_calls=tool_call_count,
                )

            if api_round > self.max_tool_rounds:
                raise AgentError(
                    f"Exceeded maximum tool rounds ({self.max_tool_rounds})"
                )

            for call in calls:
                tool_call_count += 1
                result = self._execute_call(call, callbacks)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": _field(call, "call_id"),
                        "output": result.output,
                    }
                )

        raise AgentError(f"Exceeded maximum tool rounds ({self.max_tool_rounds})")

    def _create_response(
        self, input_items: list[Any], callbacks: AgentCallbacks
    ) -> tuple[Any, str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": self.instructions,
            "input": input_items,
            "tools": [tool.openai_schema() for tool in self.tools.values()],
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        if not self.stream:
            return self.client.responses.create(**payload), ""

        event_stream = self.client.responses.create(**payload, stream=True)
        final_response: Any = None
        streamed: list[str] = []
        try:
            for event in event_stream:
                event_type = _field(event, "type")
                if event_type == "response.output_text.delta":
                    delta = _field(event, "delta", "")
                    if delta:
                        streamed.append(delta)
                        callbacks.on_text_delta(delta)
                elif event_type in {"response.completed", "response.incomplete"}:
                    final_response = _field(event, "response")
                elif event_type == "response.failed":
                    failed = _field(event, "response")
                    error = _field(failed, "error")
                    message = _field(
                        error, "message", "Provider returned response.failed"
                    )
                    raise AgentError(str(message))
                elif event_type == "error":
                    raise AgentError(
                        str(_field(event, "message", "Provider stream error"))
                    )
        finally:
            close = getattr(event_stream, "close", None)
            if callable(close):
                close()

        if final_response is None:
            raise AgentError("Provider stream ended without a terminal response event")
        return final_response, "".join(streamed)

    def _execute_call(self, call: Any, callbacks: AgentCallbacks) -> ToolResult:
        name = _field(call, "name")
        raw_arguments = _field(call, "arguments", "{}")
        try:
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else raw_arguments
            )
            if not isinstance(arguments, Mapping):
                raise TypeError("arguments must decode to a JSON object")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            arguments = {}
            callbacks.on_tool_start(str(name), arguments)
            result = ToolResult(
                f"Error: invalid arguments for {name}: {exc}", is_error=True
            )
            callbacks.on_tool_end(str(name), result, False)
            return result

        callbacks.on_tool_start(str(name), arguments)
        tool = self.tools.get(str(name))
        if tool is None:
            result = ToolResult(f"Error: tool {name!r} not found", is_error=True)
            callbacks.on_tool_end(str(name), result, False)
            return result

        streamed_output = False

        def on_output(chunk: str) -> None:
            nonlocal streamed_output
            streamed_output = True
            callbacks.on_tool_output(str(name), chunk)

        try:
            result = tool.execute(
                arguments, ToolContext(cwd=self.cwd, on_output=on_output)
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - tool failures are model observations.
            result = ToolResult(f"Error: {type(exc).__name__}: {exc}", is_error=True)
        callbacks.on_tool_end(str(name), result, streamed_output)
        return result

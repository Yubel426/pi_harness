"""Python access to Pi's real unified LLM API through a small JSONL bridge."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Self

_ASSETS = Path(__file__).with_name("_bridge")
_FILES = ("package.json", "package-lock.json", "bridge.mjs", "credentials.mjs")


class PiError(RuntimeError):
    """A runtime, authentication, or provider error."""


def runtime_directory() -> Path:
    if value := os.environ.get("PI_HARNESS_RUNTIME_DIR"):
        return Path(value).expanduser().resolve()
    digest = hashlib.sha256((_ASSETS / "package-lock.json").read_bytes()).hexdigest()[
        :12
    ]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "pi-harness" / digest


def credential_path() -> Path:
    if value := os.environ.get("PI_HARNESS_AUTH_FILE"):
        return Path(value).expanduser().resolve()
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config / "pi-harness" / "auth.json"


def _node() -> str:
    node = shutil.which("node")
    if not node:
        raise PiError(
            "Node.js 22.19+ is required; install Node.js, then run pi-harness setup"
        )
    version = subprocess.run(
        [node, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if tuple(int(part) for part in version.lstrip("v").split(".")[:3]) < (22, 19, 0):
        raise PiError("Pi requires Node.js 22.19 or newer")
    return node


def setup_runtime(directory: str | Path | None = None) -> Path:
    """Install pinned upstream dependencies explicitly, never during a model call."""
    _node()
    npm = shutil.which("npm")
    if not npm:
        raise PiError("npm is required to install the Pi runtime")
    target = (
        Path(directory).expanduser().resolve() if directory else runtime_directory()
    )
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in _FILES:
        shutil.copyfile(_ASSETS / name, target / name)
    subprocess.run(
        [npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
        cwd=target,
        check=True,
        timeout=300,
    )
    return target


class PiClient:
    """Pi Models API. Contexts, models, messages, and events use Pi's JSON schema.

    stream/complete accept provider-specific options; their *_simple variants
    accept Pi's provider-neutral reasoning and tool options. No history is kept.
    Use as a context manager, and close streaming iterators when stopping early.
    """

    def __init__(
        self,
        *,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        auth_file: str | Path | None = None,
        runtime_dir: str | Path | None = None,
        timeout: float = 600.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.auth_file = (
            Path(auth_file).expanduser().resolve() if auth_file else credential_path()
        )
        self.runtime_dir = (
            Path(runtime_dir).expanduser().resolve()
            if runtime_dir
            else runtime_directory()
        )
        self.timeout = timeout
        self._processes: set[subprocess.Popen] = set()
        self._streams: set[Iterator[dict[str, Any]]] = set()
        self._closed = False
        self.responses = _PiResponses(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _command(self) -> list[str]:
        if self._closed:
            raise PiError("PiClient is closed")
        if not (
            self.runtime_dir / "node_modules/@earendil-works/pi-ai/dist/index.js"
        ).is_file():
            raise PiError("Pi runtime is not installed. Run: pi-harness setup")
        return [_node(), str(self.runtime_dir / "bridge.mjs"), str(self.auth_file)]

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()

    def close(self) -> None:
        self._closed = True
        for stream in list(self._streams):
            stream.close()
        self._streams.clear()
        for process in list(self._processes):
            self._stop(process)

    def _frames(self, request: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            start_new_session=True,
        )
        self._processes.add(process)
        incoming: queue.Queue[str | None] = queue.Queue()

        def read() -> None:
            try:
                for line in process.stdout:
                    incoming.put(line)
            finally:
                incoming.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        deadline = time.monotonic() + self.timeout
        try:
            process.stdin.write(json.dumps(dict(request), ensure_ascii=False) + "\n")
            process.stdin.close()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PiError("Pi request timed out")
                try:
                    line = incoming.get(timeout=remaining)
                except queue.Empty as exc:
                    raise PiError("Pi request timed out") from exc
                if line is None:
                    break
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PiError("Invalid response from Pi bridge") from exc
                if "error" in frame:
                    raise PiError(frame["error"])
                yield frame
            if process.wait(timeout=max(0.1, deadline - time.monotonic())):
                raise PiError(
                    "Pi bridge failed; run pi-harness setup to verify the runtime"
                )
        finally:
            self._stop(process)
            if not process.stdin.closed:
                process.stdin.close()
            reader.join(timeout=2)
            process.stdout.close()
            self._processes.discard(process)

    def _call(self, op: str, **request: Any) -> Any:
        frames = self._frames({"op": op, **request})
        found = False
        result: Any = None
        try:
            for frame in frames:
                if "result" in frame:
                    found, result = True, frame["result"]
        finally:
            frames.close()
        if not found:
            raise PiError("Pi bridge ended without a result")
        return result

    def get_providers(self) -> list[dict[str, Any]]:
        return self._call("providers")

    def get_models(
        self,
        provider: str | None = None,
        *,
        available: bool = False,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        return self._call(
            "models", provider=provider, available=available, refresh=refresh
        )

    def get_model(self, provider: str, model_id: str) -> dict[str, Any]:
        return self._call("model", provider=provider, model=model_id)

    def auth_status(self, provider: str | None = None) -> list[dict[str, Any]]:
        return self._call("status", provider=provider)

    def logout(self, provider: str) -> None:
        self._call("logout", provider=provider)

    def set_api_key(
        self, provider: str, key: str, *, env: Mapping[str, str] | None = None
    ) -> None:
        """Persist an API key without putting it in argv or printing it."""
        self._call("api_key", provider=provider, key=key, env=env)

    def login(self, provider: str | None = None, method: str | None = None) -> None:
        if not sys.stdin.isatty():
            raise PiError(
                "Interactive login requires a terminal; use login PROVIDER --api-key-stdin or provider environment variables"
            )
        if method not in (None, "oauth", "api_key"):
            raise ValueError("method must be oauth or api_key")
        if method and not provider:
            raise ValueError("Specify a provider when selecting a login method")
        command = [*self._command(), "login"]
        if provider:
            command.append(provider)
        if method:
            command.append(method)
        process = subprocess.Popen(command, start_new_session=False)
        try:
            if process.wait():
                raise PiError("Provider login did not complete")
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

    def _request(
        self,
        model: str | Mapping[str, Any],
        context: Mapping[str, Any],
        options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        resolved = dict(options or {})
        if self.api_key:
            resolved.setdefault("apiKey", self.api_key)
        return {
            "provider": self.provider,
            "model": model,
            "baseUrl": self.base_url,
            "context": context,
            "options": resolved,
        }

    def complete(
        self,
        model: str | Mapping[str, Any],
        context: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._call("complete", **self._request(model, context, options))

    def complete_simple(
        self,
        model: str | Mapping[str, Any],
        context: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._call("complete_simple", **self._request(model, context, options))

    def _stream(
        self,
        op: str,
        model: str | Mapping[str, Any],
        context: Mapping[str, Any],
        options: Mapping[str, Any] | None,
    ) -> Iterator[dict[str, Any]]:
        frames = self._frames({"op": op, **self._request(model, context, options)})

        def iterate() -> Iterator[dict[str, Any]]:
            terminal = False
            try:
                for frame in frames:
                    if event := frame.get("event"):
                        terminal |= event.get("type") in {"done", "error"}
                        yield event
            finally:
                frames.close()
                self._streams.discard(events)
            if not terminal:
                raise PiError("Pi stream ended without a terminal event")

        events = iterate()
        self._streams.add(events)
        return events

    def stream(
        self,
        model: str | Mapping[str, Any],
        context: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        return self._stream("stream", model, context, options)

    def stream_simple(
        self,
        model: str | Mapping[str, Any],
        context: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        return self._stream("stream_simple", model, context, options)


class _PiResponses:
    """Adapter for the existing Agent loop; Pi messages are preserved verbatim.

    Mirrored function-call entries are only for Python tool dispatch. They are
    never used to reconstruct provider messages or erase thinking signatures.
    """

    def __init__(self, client: PiClient) -> None:
        self.client = client

    @staticmethod
    def _context(payload: dict[str, Any]) -> dict[str, Any]:
        messages, names = [], {}
        for item in payload["input"]:
            if item.get("type") == "pi_message":
                message = item["message"]
                messages.append(message)
                names.update(
                    {
                        c["id"]: c["name"]
                        for c in message["content"]
                        if c["type"] == "toolCall"
                    }
                )
            elif item.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "toolResult",
                        "toolCallId": item["call_id"],
                        "toolName": names[item["call_id"]],
                        "content": [{"type": "text", "text": item["output"]}],
                        "isError": item.get("is_error", False),
                        "timestamp": item.setdefault(
                            "timestamp", int(time.time() * 1000)
                        ),
                    }
                )
            elif item.get("role") == "user":
                item.setdefault("timestamp", int(time.time() * 1000))
                messages.append(dict(item))
        return {
            "systemPrompt": payload["instructions"],
            "messages": messages,
            "tools": [
                {k: v for k, v in tool.items() if k != "type"}
                for tool in payload["tools"]
            ],
        }

    @staticmethod
    def _response(message: dict[str, Any]) -> dict[str, Any]:
        if message["stopReason"] in {"error", "aborted"}:
            raise PiError(message.get("errorMessage") or message["stopReason"])
        usage = message.get("usage", {})
        output = [{"type": "pi_message", "message": message}]
        output.extend(
            {
                "type": "function_call",
                "call_id": c["id"],
                "name": c["name"],
                "arguments": c["arguments"],
            }
            for c in message["content"]
            if c["type"] == "toolCall"
        )
        return {
            "id": message.get("responseId"),
            "model": message["model"],
            "status": "completed"
            if message["stopReason"] in {"stop", "toolUse"}
            else "incomplete",
            "incomplete_details": {"reason": message["stopReason"]},
            "output": output,
            "output_text": "".join(
                c["text"] for c in message["content"] if c["type"] == "text"
            ),
            "usage": {
                "input_tokens": usage.get("input", 0)
                + usage.get("cacheRead", 0)
                + usage.get("cacheWrite", 0),
                "output_tokens": usage.get("output", 0),
                "total_tokens": usage.get("totalTokens", 0),
                "input_tokens_details": {"cached_tokens": usage.get("cacheRead", 0)},
            },
        }

    def create(self, **payload: Any) -> Any:
        context = self._context(payload)
        options: dict[str, Any] = {}
        effort = payload.get("reasoning", {}).get("effort")
        if effort and effort not in {"none", "off"}:
            options["reasoning"] = effort
        if "max_output_tokens" in payload:
            options["maxTokens"] = payload["max_output_tokens"]
        if not payload.get("stream"):
            return self._response(
                self.client.complete_simple(payload["model"], context, options)
            )

        def events() -> Iterator[dict[str, Any]]:
            upstream = self.client.stream_simple(payload["model"], context, options)
            try:
                for event in upstream:
                    if event["type"] == "text_delta":
                        yield {
                            "type": "response.output_text.delta",
                            "delta": event["delta"],
                        }
                    elif event["type"] == "done":
                        response = self._response(event["message"])
                        yield {
                            "type": f"response.{response['status']}",
                            "response": response,
                        }
                    elif event["type"] == "error":
                        self._response(event["error"])
            finally:
                upstream.close()

        return events()

"""Tool abstraction and the built-in Pi-like bash tool."""

from __future__ import annotations

import codecs
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MAX_LINES = 2_000
DEFAULT_MAX_BYTES = 50 * 1024
MAX_TIMEOUT_SECONDS = 2_147_483.647

_ANSI_ESCAPE = re.compile(r"(?:\x1B[@-_]|\x1B\[[0-?]*[ -/]*[@-~])")
_HARNESS_SECRET_ENV = {"OPENAI_API_KEY", "PI_API_KEY"}


def _noop_output(_: str) -> None:
    return None


@dataclass(slots=True)
class ToolContext:
    """Per-call context passed to Python tool implementations."""

    cwd: Path
    on_output: Callable[[str], None] = _noop_output


@dataclass(slots=True)
class ToolResult:
    """Normalized result returned by a local tool."""

    output: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


ToolExecutor = Callable[
    [Mapping[str, Any], ToolContext], ToolResult | str | Mapping[str, Any]
]


@dataclass(frozen=True, slots=True)
class Tool:
    """A JSON-schema function tool backed by a Python callable."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    executor: ToolExecutor

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }

    def execute(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        value = self.executor(arguments, context)
        if isinstance(value, ToolResult):
            return value
        if isinstance(value, str):
            return ToolResult(value)
        return ToolResult(json.dumps(value, ensure_ascii=False, default=str))


def sanitize_output(text: str) -> str:
    """Strip terminal escapes and control characters unsafe for prompts/UI."""

    text = _ANSI_ESCAPE.sub("", text).replace("\r", "")
    return "".join(char for char in text if char in {"\n", "\t"} or ord(char) >= 0x20)


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def truncate_tail(
    raw_tail: bytes,
    *,
    total_bytes: int,
    total_lines: int,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_lines: int = DEFAULT_MAX_LINES,
) -> tuple[str, bool, str | None, int]:
    """Return a UTF-8-safe tail bounded by independent byte and line limits."""

    truncated_by: str | None = None
    selected = raw_tail

    if total_bytes > max_bytes:
        truncated_by = "bytes"
        selected = selected[-max_bytes:]
        # Avoid showing a partial first line unless the retained data is one huge line.
        newline = selected.find(b"\n")
        if newline >= 0 and newline + 1 < len(selected):
            selected = selected[newline + 1 :]

    text = sanitize_output(selected.decode("utf-8", errors="replace"))
    lines = text.split("\n")
    trailing_newline = text.endswith("\n")
    if trailing_newline:
        lines.pop()

    if len(lines) > max_lines:
        truncated_by = truncated_by or "lines"
        lines = lines[-max_lines:]

    content = "\n".join(lines)
    if trailing_newline and content:
        content += "\n"
    output_lines = len(lines)
    truncated = truncated_by is not None or total_lines > max_lines
    if truncated_by is None and total_lines > max_lines:
        truncated_by = "lines"
    return content, truncated, truncated_by, output_lines


def _resolve_shell(shell_path: str | None) -> str:
    if shell_path:
        path = Path(shell_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Shell not found: {path}")
        return str(path)
    if Path("/bin/bash").is_file():
        return "/bin/bash"
    found = shutil.which("bash") or shutil.which("sh")
    if not found:
        raise RuntimeError("No bash or sh executable found")
    return found


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=0.5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass


class BashExecutor:
    """Execute shell commands with streaming, timeout, and bounded prompt output."""

    def __init__(
        self,
        cwd: Path,
        *,
        shell_path: str | None = None,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.cwd = cwd.expanduser().resolve()
        if not self.cwd.is_dir():
            raise ValueError(f"Working directory does not exist: {self.cwd}")
        self.shell = _resolve_shell(shell_path)
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    def __call__(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolResult:
        command = arguments.get("command")
        timeout = arguments.get("timeout")
        if not isinstance(command, str) or not command:
            return ToolResult(
                "Error: command must be a non-empty string", is_error=True
            )
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                return ToolResult(
                    "Error: timeout must be a number of seconds", is_error=True
                )
            if timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
                return ToolResult(
                    f"Error: timeout must be between 0 and {MAX_TIMEOUT_SECONDS} seconds",
                    is_error=True,
                )

        env = os.environ.copy()
        for name in _HARNESS_SECRET_ENV:
            env.pop(name, None)

        output_queue: queue.Queue[bytes | None] = queue.Queue()
        process = subprocess.Popen(
            [self.shell, "-c", command],
            cwd=context.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        assert process.stdout is not None
        stop_reader = threading.Event()
        try:
            os.set_blocking(process.stdout.fileno(), False)
            nonblocking_reader = True
        except (AttributeError, OSError):
            nonblocking_reader = False

        def read_output() -> None:
            try:
                if nonblocking_reader:
                    while not stop_reader.is_set():
                        try:
                            chunk = os.read(process.stdout.fileno(), 8192)
                        except BlockingIOError:
                            stop_reader.wait(0.02)
                            continue
                        if not chunk:
                            break
                        output_queue.put(chunk)
                else:
                    while chunk := process.stdout.read1(8192):
                        output_queue.put(chunk)
            except (OSError, ValueError):
                # The main thread closes the pipe when a detached descendant keeps it open.
                pass
            finally:
                output_queue.put(None)

        reader = threading.Thread(
            target=read_output, name="pi-bash-output", daemon=True
        )
        reader.start()

        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        tail = bytearray()
        total_bytes = 0
        total_newlines = 0
        last_byte: int | None = None
        started = time.monotonic()
        timed_out = False
        output_done = False
        process_exited_at: float | None = None
        execution_failed = False
        temp_fd, temp_name = tempfile.mkstemp(prefix="pi-bash-", suffix=".log")
        temp = os.fdopen(temp_fd, "wb")
        full_output_path = Path(temp_name)

        try:
            while not output_done:
                if timeout is not None and time.monotonic() - started >= float(timeout):
                    timed_out = True
                    _terminate_process_tree(process)

                try:
                    chunk = output_queue.get(timeout=0.05)
                except queue.Empty:
                    if process.poll() is not None:
                        process_exited_at = process_exited_at or time.monotonic()
                        # A background descendant can inherit the pipe and otherwise keep
                        # this tool waiting forever after the requested shell has exited.
                        if time.monotonic() - process_exited_at >= 0.2:
                            stop_reader.set()
                            output_done = True
                    continue

                if chunk is None:
                    output_done = True
                    continue

                temp.write(chunk)
                total_bytes += len(chunk)
                total_newlines += chunk.count(b"\n")
                last_byte = chunk[-1]
                tail.extend(chunk)
                if len(tail) > self.max_bytes * 2:
                    del tail[: len(tail) - self.max_bytes * 2]
                rendered = sanitize_output(decoder.decode(chunk, final=False))
                if rendered:
                    context.on_output(rendered)

            remainder = sanitize_output(decoder.decode(b"", final=True))
            if remainder:
                context.on_output(remainder)
            return_code = process.wait()
        except BaseException:
            execution_failed = True
            _terminate_process_tree(process)
            raise
        finally:
            stop_reader.set()
            temp.close()
            reader.join(timeout=1)
            process.stdout.close()
            if execution_failed:
                full_output_path.unlink(missing_ok=True)

        total_lines = total_newlines + (
            0 if total_bytes == 0 or last_byte == ord("\n") else 1
        )
        output, truncated, truncated_by, output_lines = truncate_tail(
            bytes(tail),
            total_bytes=total_bytes,
            total_lines=total_lines,
            max_bytes=self.max_bytes,
            max_lines=self.max_lines,
        )

        metadata: dict[str, Any] = {
            "exit_code": return_code,
            "duration_seconds": time.monotonic() - started,
            "truncated": truncated,
        }
        if truncated:
            metadata["full_output_path"] = str(full_output_path)
            metadata["truncated_by"] = truncated_by
            metadata["total_lines"] = total_lines
            metadata["output_lines"] = output_lines
            start_line = max(1, total_lines - output_lines + 1)
            output = (
                f"{output.rstrip()}\n\n"
                f"[Showing lines {start_line}-{total_lines} of {total_lines} "
                f"({_format_size(self.max_bytes)} limit). Full output: {full_output_path}]"
            )
        else:
            full_output_path.unlink(missing_ok=True)

        if timed_out:
            status = f"Command timed out after {timeout} seconds"
            output = f"{output.rstrip()}\n\n{status}" if output else status
            return ToolResult(output, is_error=True, metadata=metadata)
        if return_code != 0:
            status = f"Command exited with code {return_code}"
            output = f"{output.rstrip()}\n\n{status}" if output else status
            return ToolResult(output, is_error=True, metadata=metadata)
        return ToolResult(output or "(no output)", metadata=metadata)


def create_bash_tool(
    cwd: str | Path,
    *,
    shell_path: str | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Tool:
    """Create the only built-in tool exposed by the minimal harness."""

    executor = BashExecutor(
        Path(cwd), shell_path=shell_path, max_lines=max_lines, max_bytes=max_bytes
    )
    return Tool(
        name="bash",
        description=(
            "Execute a bash command in the current working directory. Returns combined "
            f"stdout and stderr. Output keeps the last {max_lines} lines or "
            f"{max_bytes // 1024}KB; truncated full output is saved to a temporary file. "
            "Optionally provide a timeout in seconds."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in seconds; by default there is no timeout",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        executor=executor,
    )

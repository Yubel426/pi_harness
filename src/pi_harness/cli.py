"""Single-prompt command-line interface for the minimal terminal harness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import __version__
from .agent import Agent, AgentCallbacks, AgentError, RunResult
from .config import REASONING_EFFORTS, HarnessConfig
from .llm import PiClient, PiError, setup_runtime
from .tools import ToolResult, create_bash_tool


def build_system_prompt(cwd: Path) -> str:
    prompt_cwd = str(cwd).replace("\\", "/")
    return f"""You are an expert coding assistant operating inside a minimal terminal agent harness.

Available tools:
- bash: Execute commands in the current working directory. Use it for reading, searching, editing, running code, and verification.

Guidelines:
- Work autonomously on safe, in-scope local tasks requested by the user.
- Inspect relevant files before changing them and preserve existing behavior outside the request.
- Do not claim completion until you have run relevant checks and inspected their results.
- Keep responses concise and show file paths clearly.

Current working directory: {prompt_cwd}"""


class TerminalRenderer:
    def __init__(self) -> None:
        self._at_line_start = True

    def text_delta(self, delta: str) -> None:
        print(delta, end="", flush=True)
        self._at_line_start = delta.endswith("\n")

    def tool_start(self, name: str, arguments: Mapping[str, Any]) -> None:
        if not self._at_line_start:
            print()
        if name == "bash":
            print(f"[{name}] $ {arguments.get('command', '')}")
        else:
            print(f"[{name}] {arguments}")
        self._at_line_start = True

    def tool_output(self, _name: str, chunk: str) -> None:
        print(chunk, end="", flush=True)
        self._at_line_start = chunk.endswith("\n")

    def tool_end(self, _name: str, result: ToolResult, streamed: bool) -> None:
        if streamed and not self._at_line_start:
            print()
        if not streamed:
            print(result.output)
        if result.is_error:
            print("[tool error]")
        self._at_line_start = True

    def finish(self, result: RunResult | None = None) -> None:
        if not self._at_line_start:
            print()
        if result is not None:
            usage = result.usage
            print(
                f"[model={result.model} rounds={result.api_rounds} tools={result.tool_calls} "
                f"tokens={usage.total_tokens}]"
            )
        self._at_line_start = True

    def callbacks(self) -> AgentCallbacks:
        return AgentCallbacks(
            on_text_delta=self.text_delta,
            on_tool_start=self.tool_start,
            on_tool_output=self.tool_output,
            on_tool_end=self.tool_end,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi-harness",
        description="Single-prompt Pi-like terminal agent harness in Python",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("setup", "login", "logout", "auth-status", "providers", "models"),
    )
    parser.add_argument(
        "target", nargs="?", help="Provider for login/logout/status/models"
    )
    parser.add_argument("-p", "--prompt", help="Run one task and exit")
    parser.add_argument(
        "--provider", help="Provider ID (default: PI_PROVIDER or openai)"
    )
    parser.add_argument("--auth", choices=("oauth", "api_key"), help="Login method")
    parser.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="Read an API key from stdin for login",
    )
    parser.add_argument(
        "--auth-file", help="Separate credential file (default: user config directory)"
    )
    parser.add_argument("--runtime-dir", help="Pi Node.js runtime directory")
    parser.add_argument(
        "--json", action="store_true", help="Print catalog/status as JSON"
    )
    parser.add_argument(
        "--available", action="store_true", help="Only list configured models"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh dynamic provider catalogs"
    )
    parser.add_argument("--cwd", default=".", help="Working directory exposed to bash")
    parser.add_argument("--model", help="Model ID (default: PI_MODEL or gpt-5.6-sol)")
    parser.add_argument(
        "--base-url", help="API-key endpoint override (/v1 is added for OpenAI)"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=(*REASONING_EFFORTS, "off"),
        help="Reasoning effort (default: PI_REASONING_EFFORT or medium)",
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-tool-rounds", type=int, default=100)
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable response text streaming"
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _create_agent(args: argparse.Namespace) -> Agent:
    config = HarnessConfig.from_environment(
        base_url=args.base_url,
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        request_timeout=args.request_timeout,
        max_tool_rounds=args.max_tool_rounds,
        cwd=args.cwd,
    )
    client = PiClient(
        provider=config.provider,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.request_timeout,
        auth_file=args.auth_file,
        runtime_dir=args.runtime_dir,
    )
    return Agent(
        client=client,
        model=config.model,
        instructions=build_system_prompt(config.cwd),
        tools=[create_bash_tool(config.cwd)],
        cwd=config.cwd,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=config.max_output_tokens,
        max_tool_rounds=config.max_tool_rounds,
        stream=not args.no_stream,
    )


def _run_prompt(agent: Agent, prompt: str) -> bool:
    renderer = TerminalRenderer()
    try:
        result = agent.run(prompt, renderer.callbacks())
    except KeyboardInterrupt:
        renderer.finish()
        print("[interrupted]", file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001 - the CLI renders provider/tool failures.
        renderer.finish()
        print(f"error: {exc}", file=sys.stderr)
        return False
    renderer.finish(result)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command:
        if args.prompt is not None:
            parser.error("Management commands do not accept --prompt")
        if args.command == "logout" and not (args.target or args.provider):
            parser.error("logout requires a provider")
        if args.api_key_stdin and (
            args.command != "login"
            or not (args.target or args.provider)
            or args.auth == "oauth"
        ):
            parser.error(
                "--api-key-stdin requires login PROVIDER and cannot be combined with OAuth"
            )
        try:
            return _manage(args)
        except (PiError, ValueError, OSError, subprocess.SubprocessError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("[interrupted]", file=sys.stderr)
            return 130
    if args.prompt is None or not args.prompt.strip():
        parser.error("--prompt cannot be empty")
    try:
        agent = _create_agent(args)
    except (ValueError, AgentError, PiError, ImportError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        return 0 if _run_prompt(agent, args.prompt) else 1
    finally:
        agent.client.close()


def _manage(args: argparse.Namespace) -> int:
    if args.command == "setup":
        print(f"Pi runtime installed at {setup_runtime(args.runtime_dir)}")
        return 0
    provider = args.target or args.provider
    with PiClient(
        auth_file=args.auth_file,
        runtime_dir=args.runtime_dir,
        timeout=args.request_timeout,
    ) as client:
        if args.command == "login":
            if args.api_key_stdin:
                client.set_api_key(provider, sys.stdin.read().strip())
                print(f"API key saved for {provider}")
            else:
                client.login(provider, args.auth)
            return 0
        if args.command == "logout":
            client.logout(provider)
            print(
                f"Removed stored credentials for {provider}; environment credentials are unchanged"
            )
            return 0
        if args.command == "providers":
            rows = client.get_providers()
            render = lambda row: (
                f"{row['id']:<28} {', '.join(row['authMethods']) or 'ambient credentials'}"
            )
        elif args.command == "models":
            rows = client.get_models(
                provider, available=args.available, refresh=args.refresh
            )
            render = lambda row: f"{row['provider']}/{row['id']}  [{row['api']}]"
        else:
            rows = client.auth_status(provider)
            render = lambda row: (
                f"{row['provider']:<28} {row['source'] or ('stored ' + row['stored'] if row['stored'] else 'not configured')}"
            )
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                print(render(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

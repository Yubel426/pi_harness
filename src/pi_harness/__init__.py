"""Minimal Pi-like terminal agent harness."""

from .agent import Agent, AgentCallbacks, AgentError, RunResult, Usage
from .config import HarnessConfig, normalize_base_url
from .llm import PiClient, PiError, setup_runtime
from .tools import Tool, ToolContext, ToolResult, create_bash_tool

__all__ = [
    "Agent",
    "AgentCallbacks",
    "AgentError",
    "HarnessConfig",
    "PiClient",
    "PiError",
    "RunResult",
    "Tool",
    "ToolContext",
    "ToolResult",
    "Usage",
    "create_bash_tool",
    "normalize_base_url",
    "setup_runtime",
]

__version__ = "0.2.0"
